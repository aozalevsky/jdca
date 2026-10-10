"""Ridge-regularised Gaussian DCA, scored by partial correlations.

The covariance of the 21-state one-hot (the gap is a state) with sequence weights w = 1 / #{sequences with identity > 0.8 L},
normalised by N = sum(w) - sqrt(mean(w)), is regularised by a diagonal ridge -- per column, lambda_i = c / sqrt(n_i) with n_i the
column's weighted non-gap count (or one lambda = c / sqrt(Meff)) -- and inverted (Woodbury or Cholesky, as in `gauss`). Per position
pair, the score is the Frobenius norm of the 20 x 20 non-gap block of the partial correlations P_ij,ab / sqrt(P_ii,aa P_jj,bb)
(P = C^-1; or of the raw block, score='fn'), then APC. Defaults: c = 4.07 with the column ridge, 5.11 with the single ridge.

`humphreys_dca` is the same estimator with the settings of the GPU DCA used by Humphreys et al. (Science 2021), as published by
Zhang, Humphreys, Pei et al. (Science 2025; script and APC formula in the preprint's supplementary Methods M5.3, bioRxiv
2024.10.01.615885): one ridge 4.5 / sqrt(Meff), the raw block, their APC.
"""
import numpy as np

from . import _jax as J
from .alignment import Q, as_codes, is_path
from .gauss import DCA, _DROWS, _TILE, apc
from .weights import identity_counts


def _programs(Lb, ns):
    j = J.jax()
    jnp, lax = j.numpy, j.lax
    HI = lax.Precision.HIGHEST
    Db = Lb * ns
    tile = min(_TILE, Lb)
    Lp = -(-Lb // tile) * tile

    def rows(codes, w, mu, npnt):
        X = j.nn.one_hot(codes, Q, dtype=jnp.float32)[:, :, :ns].reshape(codes.shape[0], Db)
        return jnp.sqrt(w / npnt)[:, None] * (X - mu[None, :])

    def woodbury(codes, w, mu, npnt, lam):
        R = rows(codes, w, mu, npnt)
        RA = R / lam[None, :]
        G = jnp.matmul(RA, R.T, precision=HI) + jnp.eye(R.shape[0], dtype=jnp.float32)
        Y = j.scipy.linalg.solve_triangular(jnp.linalg.cholesky(G), RA, lower=True)
        d = 1.0 / lam - jnp.sum(Y * Y, 0)                          # diag(C^-1)
        return jnp.pad(Y, ((0, 0), (0, (Lp - Lb) * ns))), jnp.pad(d, (0, (Lp - Lb) * ns), constant_values=1.0)

    def direct(codes, w, mu, npnt, lam):
        # the sequences stream through in 256-row chunks: memory for D x D matrices, whatever M
        def gram(S, cw):
            R = rows(cw[0], cw[1], mu, npnt)
            return S + jnp.matmul(R.T, R, precision=HI), None
        C = lax.scan(gram, jnp.zeros((Db, Db), jnp.float32), (codes.reshape(-1, 256, Lb), w.reshape(-1, 256)))[0] + jnp.diag(lam)
        Z = j.scipy.linalg.solve_triangular(jnp.linalg.cholesky(C), jnp.eye(Db, dtype=jnp.float32), lower=True)
        d = jnp.sum(Z * Z, 0)
        return jnp.pad(Z, ((0, (-Db) % 256), (0, (Lp - Lb) * ns))), jnp.pad(d, (0, (Lp - Lb) * ns), constant_values=1.0)

    def tiles(Z, d, i0, j0, k0, k1, pc):
        # as gauss: chunks k0 .. k1 - 1 (the others are zero for these columns); only the 20 non-gap states enter the product
        nongap = lambda X: X.reshape(256, tile, ns)[:, :, :20].reshape(256, tile * 20)

        def body(k, acc):
            Zi = nongap(lax.dynamic_slice(Z, (k * 256, i0 * ns), (256, tile * ns)))
            Zj = nongap(lax.dynamic_slice(Z, (k * 256, j0 * ns), (256, tile * ns)))
            return acc + jnp.matmul(Zi.T, Zj, precision=HI)
        T = lax.fori_loop(k0, k1, body, jnp.zeros((tile * 20, tile * 20), jnp.float32)).reshape(tile, 20, tile, 20)
        if pc:
            di = lax.dynamic_slice_in_dim(d, i0 * ns, tile * ns).reshape(tile, ns)[:, :20]
            dj = lax.dynamic_slice_in_dim(d, j0 * ns, tile * ns).reshape(tile, ns)[:, :20]
            T = T / jnp.sqrt(di[:, :, None, None] * dj[None, None, :, :])
        return jnp.sqrt((T ** 2).sum((1, 3)))
    return woodbury, direct, tiles, tile, Lp


def _scores(Z, w, c, column_ridge, pc, dev):
    """Symmetric (L, L) ridge-DCA scores (diagonal 0, no APC) for raw weights w."""
    j = J.jax()
    M, L = Z.shape
    ns = Q                                                                  # the gap is a state
    meff = float(w.sum())
    npnt = meff - np.sqrt(w.mean())
    # sum_m w_m x_m, per column and state, one column at a time (no M x L temporaries)
    mu = np.stack([np.bincount(col, weights=w, minlength=Q) for col in np.ascontiguousarray(Z.T)])
    Lb = J.lbucket(L)
    Db = Lb * ns
    mub = np.zeros((Lb, ns), np.float32)
    mub[:L] = (mu[:, :ns] / npnt).astype(np.float32)
    lamc = np.ones(Lb)                                                       # padded positions: any positive value (decoupled)
    lamc[:L] = c / np.sqrt(np.maximum(mu[:, :Q - 1].sum(1), 1.0)) if column_ridge else c / np.sqrt(meff)
    lamv = np.repeat(lamc, ns).astype(np.float32)
    woodbury, direct, tiles, tile, Lp = _programs(Lb, ns)
    f32, i32 = np.float32, np.int32
    route = 'woodbury' if M + 1 < Db / np.sqrt(2.0) else 'direct'
    rows = J.rbucket(M) if route == 'woodbury' else -(-M // _DROWS) * _DROWS
    codes = np.full((rows, Lb), Q, i32)
    codes[:M, :L] = Z
    wv = np.zeros(rows, f32)
    wv[:M] = w
    put = lambda a: j.device_put(a, dev)
    e_z = J.aot(('r' + route[0], Lb, rows, ns), woodbury if route == 'woodbury' else direct,
                [((rows, Lb), i32), ((rows,), f32), ((Db,), f32), ((), f32), ((Db,), f32)], dev)
    Zm, d = e_z(put(codes), put(wv), put(mub.reshape(-1)), put(f32(npnt)), put(lamv))
    zr = rows if route == 'woodbury' else Db + (-Db) % 256
    k1 = -(-(M if route == 'woodbury' else L * ns) // 256)                    # the rest is zero: padded rows / padded positions
    e_t = J.aot(('rt', Lb, zr, tile, ns, pc), lambda Z_, d_, a, b, k, kk: tiles(Z_, d_, a, b, k, kk, pc),
                [((zr, Lp * ns), f32), ((Lp * ns,), f32)] + [((), i32)] * 4, dev)
    S = np.zeros((Lp, Lp))
    for i0 in range(0, L, tile):
        for j0 in range(i0, L, tile):
            k0 = 0 if route == 'woodbury' else (j0 * ns) // 256
            S[i0:i0 + tile, j0:j0 + tile] = np.asarray(e_t(Zm, d, put(i32(i0)), put(i32(j0)), put(i32(k0)), put(i32(k1))))
    S = S[:L, :L]
    if not np.isfinite(S).all():
        raise np.linalg.LinAlgError(f'non-finite ridge scores ({route} route)')
    S = np.triu(S, 1)
    return S + S.T


def ridge_dca(alignment, c=None, column_ridge=True, score='pc', identity=0.8, max_gap_fraction=None, device='auto', use_apc=True):
    """Ridge DCA contact scores.

    alignment: a FASTA / A3M path (read keeping every sequence, or with `max_gap_fraction`) or an (M, L) matrix of codes 0..20 (used
    as given: no gap filter, and passing `max_gap_fraction` with a matrix is an error).
    c: the ridge constant (default 4.07 with the column ridge, 5.11 without). column_ridge: lambda_i = c / sqrt(n_i) (default) or
    c / sqrt(Meff). score: 'pc' (partial correlations, default) or 'fn' (the raw coupling block). identity: sequences with more than
    identity * L identical columns are neighbours for the reweighting (0 <= identity < 1). device: as in `gaussdca`.
    Returns DCA(scores (L, L) float64 with APC and a zero diagonal, meff, theta = 1 - identity).
    """
    if score not in ('pc', 'fn'):
        raise ValueError(f"score={score!r}: expected 'pc' or 'fn'")
    c = (4.07 if column_ridge else 5.11) if c is None else float(c)
    if c <= 0:
        raise ValueError(f'c={c!r}: the ridge constant must be positive')
    if not 0.0 <= float(identity) < 1.0:
        raise ValueError(f'identity={identity!r}: expected a fraction 0 <= identity < 1 (at 1 no sequence is its own neighbour)')
    if max_gap_fraction is not None and not is_path(alignment):
        raise ValueError('max_gap_fraction applies when reading a file; a matrix is used as given (filter it with '
                         'jdca.read_alignment(path, max_gap_fraction))')
    Z = as_codes(alignment, 1.0 if max_gap_fraction is None else max_gap_fraction)
    if len(Z) < 2:
        raise ValueError('ridge DCA needs at least 2 sequences (its normaliser sum(w) - sqrt(mean(w)) is 0 for one)')
    if device in ('auto', 'server'):
        from . import server
        out = server.request('ridge_dca', Z, dict(c=c, column_ridge=column_ridge, score=score, identity=identity, use_apc=use_apc),
                             required=device == 'server')
        if out is not None:
            return DCA(*out)
        device = 'auto'
    dev = J.device(device)
    M, L = Z.shape
    w = 1.0 / identity_counts(Z, int(np.floor(identity * L)) + 1, dev)      # identity > 0.8 L, strictly
    S = _scores(Z, w, c, column_ridge, score == 'pc', dev)
    return DCA(apc(S) if use_apc else S, float(w.sum()), round(1.0 - identity, 12))


def humphreys_dca(alignment, device='auto', use_apc=True):
    """The GPU DCA used by Humphreys et al. (Science 2021), with the settings of the script Zhang, Humphreys, Pei et al. published
    (Science 2025; bioRxiv 2024.10.01.615885, supplementary Methods M5.3): every sequence kept, weights 1 / #{sequences with
    identity > 0.8 L}, the 21-state covariance plus one ridge 4.5 / sqrt(Meff), and per pair the Frobenius norm of the raw 20 x 20
    non-gap block of C^-1. Their script returns these scores; their APC, S - S_i. S_.j / S.. (sums over all residues), follows.
    use_apc=False returns the scores as their script does.

    alignment, device: as in `ridge_dca`. Returns DCA(scores, meff, theta = 0.2).
    """
    r = ridge_dca(as_codes(alignment, 1.0), c=4.5, column_ridge=False, score='fn', identity=0.8, device=device, use_apc=False)
    return DCA(apc(r.scores, offdiagonal=False), r.meff, r.theta) if use_apc else r
