"""GaussDCA (Baldassi et al., PLoS ONE 2014): the Frobenius-norm contact score of the inverse pseudocounted covariance.

The estimator is GaussDCA.jl's. With q states per column (the last one is the reference state, left out of the one-hot) the
pseudocounted covariance C = Pij - Pi Pi^T (pseudocount pc, weights w normalised to sum 1) is exactly the covariance of the mixture
(1 - pc) x alignment + pc x independent uniform columns:

    C = A + R^T R,  A = blockdiag(pc (I/q - 11^T/q^2)),
    R = [ sqrt((1 - pc) w_m) (x_m - f) ;  sqrt(pc (1 - pc)) (f - 1/q) ]      (x_m the one-hot of sequence m, f the weighted frequencies)

so C is never formed by subtracting nearly equal numbers. With r = M + 1 rows and D = (q - 1) L:
  r < D / sqrt(2): Woodbury, C^-1 = A^-1 - Y^T Y with Y = chol(I + R A^-1 R^T)^-1 R A^-1 (an r x r factorisation, no D x D inverse);
  otherwise:       C^-1 = Z^T Z with Z = chol(C)^-1.
The off-diagonal (q - 1) x (q - 1) blocks of C^-1 (up to sign) are zero-sum gauged and their Frobenius norms taken, then APC.
"""
from collections import namedtuple

import numpy as np

from . import _jax as J
from .alignment import Q, as_codes, is_path
from .weights import identity_counts, sequence_weights

DCA = namedtuple('DCA', 'scores meff theta')


def apc(S, offdiagonal=True):
    """Average-product correction. offdiagonal=True: means over the off-diagonal entries (GaussDCA.jl's `correct_APC`),
    S - S_i. S_.j / (S.. (1 - 1/L)); False: means over all entries, the zero diagonal included, S - S_i. S_.j / S.. (the adaptive
    mode, and the formula of Humphreys et al.)."""
    L = S.shape[0]
    if not S.any():
        raise ValueError('every score is zero (the alignment carries no coupling signal): APC is undefined')
    if offdiagonal:
        s = S.sum(axis=0)
        out = S - np.outer(s, s) / (S.sum() * (1.0 - 1.0 / L))
    else:
        m = S.mean(axis=1)
        out = S - np.outer(m, m) / (S.mean() + 1e-10)
    np.fill_diagonal(out, 0.0)
    return out


def _route(M, L, s):
    return 'woodbury' if M + 1 < L * s / np.sqrt(2.0) else 'direct'


# ------------------------------------------------------------------------------------------------------------------------------ CPU
def _factor(Z, w, pc, q):
    M, L = Z.shape
    s = q - 1
    D = L * s
    x = Z.astype(np.int64)
    fr = np.zeros((L, Q))
    np.add.at(fr, (np.broadcast_to(np.arange(L), (M, L)), x), np.broadcast_to(w[:, None], (M, L)))
    f = fr[:, :s].reshape(D).astype(np.float32)
    sw = np.sqrt((1.0 - pc) * w).astype(np.float32)
    R = np.empty((M + 1, D), np.float32)
    np.multiply(-sw[:, None], f[None, :], out=R[:M])
    m, l = np.nonzero(x < s)
    R[m, l * s + x[m, l]] += sw[m]
    R[M] = np.float32(np.sqrt(pc * (1.0 - pc))) * (f - np.float32(1.0 / q))
    return R


def _inverse_cpu(Z, w, pc, eps, q):
    """The upper triangle of T (D x D float32) whose off-diagonal blocks equal those of C^-1 up to sign (BLAS / LAPACK, in place)."""
    from scipy.linalg import blas, lapack
    M, L = Z.shape
    s = q - 1
    R = _factor(Z, w, pc, q)
    a, b = pc / q + eps, pc / q ** 2
    if _route(M, L, s) == 'direct':
        T = blas.ssyrk(1.0, R.T, trans=0, lower=0)
        del R
        blk = (a * np.eye(s) - b).astype(np.float32)
        for i in range(L):
            T[i * s:(i + 1) * s, i * s:(i + 1) * s] += blk
        cf, info = lapack.spotrf(T, lower=0, overwrite_a=1)
        if info:
            raise np.linalg.LinAlgError(f'spotrf info={info}')
        T, info = lapack.spotri(cf, lower=0, overwrite_c=1)
        if info:
            raise np.linalg.LinAlgError(f'spotri info={info}')
        return T
    c = b / (a * (a - s * b))                            # (a I - b 11^T)^-1 = I/a + c 11^T
    Rb = R.reshape(M + 1, L, s)
    RA = (Rb / np.float32(a) + np.float32(c) * Rb.sum(-1, keepdims=True)).reshape(M + 1, L * s)
    G = blas.sgemm(1.0, RA, R, trans_b=1)
    del R, Rb
    G[np.diag_indices_from(G)] += 1.0
    Lc, info = lapack.spotrf(G, lower=1, overwrite_a=1)
    if info:
        raise np.linalg.LinAlgError(f'spotrf info={info}')
    Y = blas.strsm(1.0, Lc, RA, lower=1, overwrite_b=1)
    del RA
    return blas.ssyrk(1.0, Y.T, trans=0, lower=0)


def _block_norms_cpu(T, L, s):
    """Frobenius norms of the zero-sum-gauged off-diagonal blocks of T's upper triangle, in float64 row bands of ~64 MB."""
    S = np.zeros((L, L))
    ch = max(1, (1 << 26) // (s * T.shape[1] * 4))
    for i0 in range(0, L, ch):
        i1 = min(L, i0 + ch)
        B = T[i0 * s:i1 * s, i0 * s:].reshape(i1 - i0, s, L - i0, s).astype(np.float64)
        m1, m3, m13 = B.mean(1, keepdims=True), B.mean(3, keepdims=True), B.mean((1, 3), keepdims=True)
        B -= m1
        B -= m3
        B += m13
        np.square(B, out=B)
        S[i0:i1, i0:] = np.sqrt(B.sum((1, 3)))
    S = np.triu(S, 1)
    return S + S.T


# ------------------------------------------------------------------------------------------------------------------------------ GPU
_TILE = 128
_DROWS = 2048                  # the direct route's rows are padded to a multiple of this (zero weight): few programs per length bucket


def _programs(Lb, q):
    """The bucketed programs for Lb columns (`J.lbucket`) and q states: rows padded with zero weight and code Q, padded columns all-zero
    one-hots, the mean-difference row masked to the real columns. The contractions over sequences (the Gram matrix of the deep route and
    the tile products) are summed over 256-row chunks: one long float32 GPU contraction is ~50x less accurate than LAPACK's blocked
    sums, 256-row chunks match it at the same speed."""
    j = J.jax()
    jnp, lax = j.numpy, j.lax
    HI = lax.Precision.HIGHEST
    s = q - 1
    Db = Lb * s
    tile = min(_TILE, Lb)
    Lp = -(-Lb // tile) * tile

    def consts(pc, eps):
        a = pc / q + eps
        b = pc / q ** 2
        return a, b, b / (a * (a - s * b)), jnp.sqrt(pc * (1.0 - pc))

    def rows(codes, w, pc):
        X = j.nn.one_hot(codes, Q, dtype=jnp.float32)[:, :, :s].reshape(codes.shape[0], Db)
        f = jnp.matmul(w, X, precision=HI)
        return jnp.sqrt((1.0 - pc) * w)[:, None] * (X - f[None, :]), f

    def woodbury(codes, w, colm, pc, eps, mi):
        # row `mi` = M (zero weight) receives the mean-difference row
        a, b, c, su = consts(pc, eps)
        R, f = rows(codes, w, pc)
        R = lax.dynamic_update_slice(R, (su * (f - 1.0 / q) * colm)[None, :], (mi, jnp.int32(0)))
        r = R.shape[0]
        Rb = R.reshape(r, Lb, s)
        RA = (Rb / a + c * Rb.sum(-1, keepdims=True)).reshape(r, Db)
        G = jnp.matmul(RA, R.T, precision=HI) + jnp.eye(r, dtype=jnp.float32)
        Y = j.scipy.linalg.solve_triangular(jnp.linalg.cholesky(G), RA, lower=True)
        return jnp.pad(Y, ((0, 0), (0, (Lp - Lb) * s)))

    def direct(codes, w, colm, pc, eps):
        # the sequences stream through in 256-row chunks (one-hot built per chunk): memory for D x D matrices, whatever M
        a, b, _, su = consts(pc, eps)
        chunks = (codes.reshape(-1, 256, Lb), w.reshape(-1, 256))
        onehot = lambda c: j.nn.one_hot(c, Q, dtype=jnp.float32)[:, :, :s].reshape(256, Db)
        f = lax.scan(lambda F, cw: (F + jnp.matmul(cw[1], onehot(cw[0]), precision=HI), None), jnp.zeros(Db, jnp.float32), chunks)[0]

        def gram(S, cw):
            Y = jnp.sqrt((1.0 - pc) * cw[1])[:, None] * (onehot(cw[0]) - f[None, :])
            return S + jnp.matmul(Y.T, Y, precision=HI), None
        C = lax.scan(gram, jnp.zeros((Db, Db), jnp.float32), chunks)[0]
        m = su * (f - 1.0 / q) * colm
        C = C + jnp.outer(m, m)
        idx = jnp.arange(Lb)
        C = C.reshape(Lb, s, Lb, s).at[idx, :, idx, :].add((a * jnp.eye(s, dtype=jnp.float32) - b)[None]).reshape(Db, Db)
        Z = j.scipy.linalg.solve_triangular(jnp.linalg.cholesky(C), jnp.eye(Db, dtype=jnp.float32), lower=True)
        return jnp.pad(Z, ((0, (-Db) % 256), (0, (Lp - Lb) * s)))

    def tiles(Z, i0, j0, k0, k1):
        # (Z^T Z) for a tile x tile block of positions, summed over the 256-row chunks k0 .. k1 - 1. The chunks outside hold exact
        # zeros for these columns and are skipped: on the direct route Z = chol(C)^-1 is lower-triangular (no rows above j0*s) and
        # block-diagonal between real and padded positions (no rows past L*s); on the Woodbury route the padded rows are zero.
        def body(k, acc):
            Zi = lax.dynamic_slice(Z, (k * 256, i0 * s), (256, tile * s))
            Zj = lax.dynamic_slice(Z, (k * 256, j0 * s), (256, tile * s))
            return acc + jnp.matmul(Zi.T, Zj, precision=HI)
        T = lax.fori_loop(k0, k1, body, jnp.zeros((tile * s, tile * s), jnp.float32)).reshape(tile, s, tile, s)
        T = T - T.mean(1, keepdims=True) - T.mean(3, keepdims=True) + T.mean((1, 3), keepdims=True)
        return jnp.sqrt((T ** 2).sum((1, 3)))
    return woodbury, direct, tiles, tile, Lp


def _norms_jax(Z, w, pc, eps, q, dev):
    """Symmetric (L, L) block norms (diagonal 0, no APC) on a JAX device with bucketed, ahead-of-time compiled programs."""
    j = J.jax()
    M, L = Z.shape
    s = q - 1
    Lb = J.lbucket(L)
    Db = Lb * s
    woodbury, direct, tiles, tile, Lp = _programs(Lb, q)
    f32, i32 = np.float32, np.int32
    colm = np.zeros(Db, f32)
    colm[:L * s] = 1.0
    route = _route(M, L, s)
    rows = J.rbucket(M + 1) if route == 'woodbury' else -(-M // _DROWS) * _DROWS
    codes = np.full((rows, Lb), Q, i32)
    codes[:M, :L] = Z
    wv = np.zeros(rows, f32)
    wv[:M] = w
    put = lambda x: j.device_put(x, dev)
    if route == 'woodbury':
        e_z = J.aot(('gw', Lb, rows, q), woodbury, [((rows, Lb), i32), ((rows,), f32), ((Db,), f32), ((), f32), ((), f32), ((), i32)],
                    dev)
        Zm = e_z(put(codes), put(wv), put(colm), put(f32(pc)), put(f32(eps)), put(i32(M)))
        zr, k1 = rows, -(-(M + 1) // 256)                                         # rows past M + 1 are zero
    else:
        e_z = J.aot(('gd', Lb, rows, q), direct, [((rows, Lb), i32), ((rows,), f32), ((Db,), f32), ((), f32), ((), f32)], dev)
        Zm = e_z(put(codes), put(wv), put(colm), put(f32(pc)), put(f32(eps)))
        zr, k1 = Db + (-Db) % 256, -(-(L * s) // 256)                             # rows of padded positions are zero
    e_t = J.aot(('gt', Lb, zr, tile, q), tiles, [((zr, Lp * s), f32)] + [((), i32)] * 4, dev)
    S = np.zeros((Lp, Lp))
    for i0 in range(0, L, tile):
        for j0 in range(i0, L, tile):
            k0 = 0 if route == 'woodbury' else (j0 * s) // 256
            S[i0:i0 + tile, j0:j0 + tile] = np.asarray(e_t(Zm, put(i32(i0)), put(i32(j0)), put(i32(k0)), put(i32(k1))))
    del Zm
    S = S[:L, :L]
    S = np.triu(S, 1)
    return S + S.T


def _norms(Z, w, pc, eps, q, dev):
    """Symmetric (L, L) block norms on `dev`; a map that does not fit in GPU memory is computed on the CPU."""
    S = None
    if dev.platform != 'cpu':
        try:
            S = _norms_jax(Z, w, pc, eps, q, dev)
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' not in str(e) and 'out of memory' not in str(e).lower():
                raise
            import sys
            print(f'jdca: the map (M={Z.shape[0]}, L={Z.shape[1]}) does not fit in GPU memory; computing it on the CPU',
                  file=sys.stderr, flush=True)
    if S is None:
        S = _block_norms_cpu(_inverse_cpu(Z, w, pc, eps, q), Z.shape[1], q - 1)
    if not np.isfinite(S).all():
        raise np.linalg.LinAlgError('non-finite scores')
    return S


# ------------------------------------------------------------------------------------------------------------------------------ API
MODES = {
    # GaussDCA.jl's estimator and defaults; q = the largest state code in the alignment (GaussDCA.jl's `q = maximum(Z)`)
    'auto': dict(pseudocount=0.8, theta='auto', max_gap_fraction=0.9, eps=0.0, apc_offdiagonal=True, states='max'),
    # adaptive: adaptive pseudocount, neighbours at identity >= 0.8 L, a 1e-4 diagonal, every sequence kept, APC over all entries, q = 21
    'adaptive': dict(pseudocount='adaptive', theta='identity0.8', max_gap_fraction=1.0, eps=1e-4, apc_offdiagonal=False, states=Q),
}


def adaptive_pseudocount(M, L):
    """The adaptive mode's pseudocount: 0.5 when the alignment has more sequences than covariance columns (M > 20 L), else
    max(0.5, 0.95 - 0.45 M / (20 L))."""
    r = M / float(L * (Q - 1))
    return 0.5 if r > 1.0 else max(0.5, 0.95 - 0.45 * r)


def _check(cfg, window):
    pc, th, g = cfg['pseudocount'], cfg['theta'], cfg['max_gap_fraction']
    if pc != 'adaptive' and not 0.0 < float(pc) <= 1.0:
        raise ValueError(f'pseudocount={pc!r}: expected 0 < pseudocount <= 1 (0 leaves the covariance singular)')
    if th not in ('auto', 'identity0.8') and not 0.0 <= float(th) <= 1.0:
        raise ValueError(f"theta={th!r}: expected 'auto' or a fraction in [0, 1]")
    if not 0.0 <= float(g) <= 1.0:
        raise ValueError(f'max_gap_fraction={g!r}: expected a fraction in [0, 1]')
    if window is not None and (isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1):
        raise ValueError(f'window={window!r}: expected a positive integer')


def _informative(Z, q):
    """The number of columns holding a state other than the reference state (the gap, or code q - 1 when there is no gap). A column
    without one has exactly zero couplings; with fewer than two such columns every score is zero."""
    return int((Z < q - 1).any(axis=0).sum())


def gaussdca(alignment, mode='auto', pseudocount=None, theta=None, max_gap_fraction=None, window=None, device='auto', use_apc=True):
    """GaussDCA Frobenius contact scores.

    alignment: a FASTA / A3M path (read with the mode's `max_gap_fraction`) or an (M, L) matrix of codes 0..20 (used as given: no gap
      filter, and passing `max_gap_fraction` with a matrix is an error).
    mode: 'auto' (default) = GaussDCA.jl's estimator and defaults (pseudocount 0.8, theta 'auto', max_gap_fraction 0.9, q = the largest
      state code present); 'adaptive' = adaptive pseudocount (`adaptive_pseudocount`), sequences with identity >= 0.8 L as neighbours,
      a 1e-4 diagonal, no gap filter, APC over all entries, q = 21.
    pseudocount, theta, max_gap_fraction: override the mode's value ('auto' mode: theta = 'auto' or a fraction).
    window: None (default) = the whole-chain inverse; an integer w = the windowed map for chains too long for it (`_windowed`):
      tiles of 2w + 1 columns, each inverted on its own, scoring only pairs with |i - j| <= w (the rest NaN). A chain no longer than
      2w + 1 gets the whole-chain map.
    device: 'auto' (the jdca server if one is running, else a local GPU, else the CPU), 'server', 'gpu', 'cpu'.
    Returns DCA(scores (L, L) float64 with APC and a zero diagonal, meff, theta).
    """
    if mode not in MODES:
        raise ValueError(f'mode={mode!r}: expected one of {sorted(MODES)}')
    cfg = dict(MODES[mode])
    for k, v in (('pseudocount', pseudocount), ('theta', theta), ('max_gap_fraction', max_gap_fraction)):
        if v is not None:
            cfg[k] = v
    _check(cfg, window)
    if max_gap_fraction is not None and not is_path(alignment):
        raise ValueError('max_gap_fraction applies when reading a file; a matrix is used as given (filter it with '
                         'jdca.read_alignment(path, max_gap_fraction))')
    Z = as_codes(alignment, cfg['max_gap_fraction'])
    q = int(Z.max()) + 1 if cfg['states'] == 'max' else int(cfg['states'])
    if q < 2:
        raise ValueError('the alignment has a single state: no couplings to estimate')
    if _informative(Z, q) < 2:
        raise ValueError('fewer than two columns hold an amino acid (or any state but the reference state): no couplings to estimate')
    if device in ('auto', 'server'):
        from . import server
        out = server.request('gaussdca', Z, dict(mode=mode, pseudocount=pseudocount, theta=theta, window=window, use_apc=use_apc),
                             required=device == 'server')
        if out is not None:
            return DCA(*out)
        device = 'auto'
    dev = J.device(device)
    M, L = Z.shape
    if cfg['theta'] == 'identity0.8':
        w = 1.0 / identity_counts(Z, int(np.ceil(0.8 * L - 1e-9)), dev)
        meff, th = float(w.sum()), 0.2
    else:
        w, meff, th = sequence_weights(Z, cfg['theta'], dev)
    w = w / meff
    if window is not None and L > 2 * int(window) + 1:
        return DCA(_windowed(Z, w, cfg, q, int(window), dev, use_apc), meff, th)
    pc = adaptive_pseudocount(M, L) if cfg['pseudocount'] == 'adaptive' else float(cfg['pseudocount'])
    S = _norms(Z, w, pc, cfg['eps'], q, dev)
    return DCA(apc(S, cfg['apc_offdiagonal']) if use_apc else S, meff, th)


def _windowed(Z, w, cfg, q, w0, dev, use_apc=True):
    """The windowed map for chains too long for the whole-chain inverse: GPU memory for the inverse independent of L.

    Tiles of nw = 2 w0 + 1 columns start every w0 columns. The weights come from the whole alignment and a tile's covariance is built
    from its own columns: with a fixed pseudocount ('auto' mode) it is exactly the sub-block of the whole-chain covariance, so only the
    inverse is local (the adaptive mode recomputes its pseudocount for the tile's width). Each tile's scores get APC on their own (the
    tiles' inverses are on different scales, so a correction mixing tiles would not match the scores it corrects). A pair (i, j) with
    |i - j| <= w0 lies in at least one tile; it is read from the tile whose centre is nearest to the pair's midpoint. Pairs further
    apart are NaN."""
    M, L = Z.shape
    nw = 2 * w0 + 1
    starts = np.arange(0, L - nw + 1, w0, dtype=np.int64)
    if starts[-1] != L - nw:
        starts = np.append(starts, L - nw)
    pc = adaptive_pseudocount(M, nw) if cfg['pseudocount'] == 'adaptive' else float(cfg['pseudocount'])
    tiles = []
    for s in starts:
        Zt = np.ascontiguousarray(Z[:, s:s + nw])
        if _informative(Zt, q) < 2:                     # no coupling in this tile: zeros (APC undefined)
            tiles.append(np.zeros((nw, nw)))
            continue
        T = _norms(Zt, w, pc, cfg['eps'], q, dev)
        tiles.append(apc(T, cfg['apc_offdiagonal']) if use_apc else T)
    tiles = np.stack(tiles)
    ar = np.arange(L)
    S = np.full((L, L), np.nan)
    np.fill_diagonal(S, 0.0)
    for d in range(1, w0 + 1):
        i = ar[:L - d]
        j = i + d
        want = (i + j) // 2 - w0
        k = np.clip(np.searchsorted(starts, want), 0, len(starts) - 1)
        for kk in (k - 1, k + 1):
            kk = np.clip(kk, 0, len(starts) - 1)
            better = (np.abs(starts[kk] - want) < np.abs(starts[k] - want)) & (starts[kk] <= i) & (starts[kk] + nw > j)
            k = np.where(better, kk, k)
        k = np.where((starts[k] <= i) & (starts[k] + nw > j), k, np.clip(np.searchsorted(starts, j - nw + 1), 0, len(starts) - 1))
        S[i, j] = S[j, i] = tiles[k, i - starts[k], j - starts[k]]
    return S
