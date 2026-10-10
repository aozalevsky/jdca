"""A direct float64 NumPy transcription of GaussDCA.jl (gDCA with score=:frob) -- the correctness reference for the tests.

It forms the pseudocounted frequencies and the covariance exactly as DCAUtils.jl does (no factorisation, no float32), so it is slow
(O(M^2 L) reweighting, a dense (20 L)^3 inverse) and only meant for small alignments."""
import numpy as np


def theta_auto(Z):
    M, L = Z.shape
    ids = sum(int((Z[a + 1:] == Z[a]).sum()) for a in range(M - 1))
    return 0.5 if ids == 0 else min(0.5, 0.38 * 0.32 / (ids / L / (0.5 * M * (M - 1))))


def weights(Z, theta):
    M, L = Z.shape
    thresh = np.floor(theta * L)
    if thresh <= 0:
        return np.ones(M)
    return 1.0 / np.array([((Z != Z[a]).sum(1) < thresh).sum() for a in range(M)], float)


def gaussdca_reference(Z, pseudocount=0.8, theta='auto', eps=0.0, apc_offdiagonal=True, q='max'):
    """GaussDCA.jl's gDCA (score=:frob). q = 'max' is GaussDCA.jl's `q = maximum(Z)` (here the largest 0-based code + 1); the last
    state is the reference state. theta = 'identity0.8' gives the adaptive mode's neighbours (identity >= 0.8 L)."""
    Z = np.asarray(Z, np.int64)
    M, L = Z.shape
    q = int(Z.max()) + 1 if q == 'max' else int(q)
    if theta == 'auto':
        theta = theta_auto(Z)
    if theta == 'identity0.8':                         # adaptive mode: neighbours share >= 0.8 L columns
        W = 1.0 / np.array([((Z == Z[a]).sum(1) >= 0.8 * L - 1e-9).sum() for a in range(M)], float)
        theta = 0.2
    else:
        W = weights(Z, theta)
    return _apc(fn_norms(Z, W, pseudocount, q, eps), apc_offdiagonal), W.sum(), theta


def fn_norms(Z, W, pc, q, eps=0.0):
    """The Frobenius norms of the zero-sum-gauged (q-1) x (q-1) blocks of C^-1, before APC, for raw weights W."""
    Z = np.asarray(Z, np.int64)
    M, L = Z.shape
    s = q - 1
    meff = W.sum()
    X = np.zeros((M, L * s))
    m, l = np.nonzero(Z < s)
    X[m, l * s + Z[m, l]] = 1.0
    Pi = W @ X / meff
    Pij_true = X.T @ (W[:, None] * X) / meff
    # DCAUtils.add_pseudocount: off-diagonal blocks (1 - pc) Pij + pc/q^2, diagonal blocks (1 - pc) Pij + (pc/q) I, Pi (1 - pc) Pi + pc/q
    Pij = (1 - pc) * Pij_true + pc / q / q
    for i in range(L):
        xr = slice(i * s, (i + 1) * s)
        Pij[xr, xr] = (1 - pc) * Pij_true[xr, xr] + np.eye(s) * (pc / q)
    Pi = (1 - pc) * Pi + pc / q
    C = Pij - np.outer(Pi, Pi) + eps * np.eye(L * s)
    J = np.linalg.inv(C)
    FN = np.zeros((L, L))
    for i in range(L - 1):
        for j in range(i + 1, L):
            B = J[i * s:(i + 1) * s, j * s:(j + 1) * s]
            B = B - B.mean(0, keepdims=True) - B.mean(1, keepdims=True) + B.mean()
            FN[i, j] = FN[j, i] = np.sqrt((B ** 2).sum())
    return FN


def windowed_reference(Z, w0, pseudocount=0.8, adaptive=False):
    """The windowed map: whole-alignment weights and q, tiles of 2 w0 + 1 columns every w0 columns, each tile's own inverse and APC;
    a pair (|i - j| <= w0) is read from the containing tile whose start is nearest to (i + j) // 2 - w0 (ties: the later tile).
    adaptive=True: the adaptive mode (identity >= 0.8 L neighbours, q = 21, the pseudocount for the tile's width, a 1e-4 diagonal,
    APC over all entries)."""
    Z = np.asarray(Z, np.int64)
    M, L = Z.shape
    nw = 2 * w0 + 1
    if adaptive:
        q, eps, offdiagonal = 21, 1e-4, False
        W = 1.0 / np.array([((Z == Z[a]).sum(1) >= 0.8 * L - 1e-9).sum() for a in range(M)], float)
        r = M / (nw * 20.0)
        pseudocount = 0.5 if r > 1.0 else max(0.5, 0.95 - 0.45 * r)
    else:
        q, eps, offdiagonal = int(Z.max()) + 1, 0.0, True
        W = weights(Z, theta_auto(Z))
    starts = list(range(0, L - nw + 1, w0))
    if starts[-1] != L - nw:
        starts.append(L - nw)
    tiles = [_apc(fn_norms(Z[:, s:s + nw], W, pseudocount, q, eps), offdiagonal) for s in starts]
    S = np.full((L, L), np.nan)
    np.fill_diagonal(S, 0.0)
    for i in range(L):
        for j in range(i + 1, min(L, i + w0 + 1)):
            want = (i + j) // 2 - w0
            k = min((k for k, s in enumerate(starts) if s <= i and s + nw > j), key=lambda k: (abs(starts[k] - want), -starts[k]))
            S[i, j] = S[j, i] = tiles[k][i - starts[k], j - starts[k]]
    return S


def _apc(FN, offdiagonal=True):
    L = FN.shape[0]
    if offdiagonal:
        Si = FN.sum(0)
        S = FN - np.outer(Si, Si) / (FN.sum() * (1 - 1 / L))
    else:
        m = FN.mean(1)
        S = FN - np.outer(m, m) / (FN.mean() + 1e-10)
    np.fill_diagonal(S, 0.0)
    return S


def ridge_reference(Z, c=4.07, column_ridge=True, score='pc', identity=0.8):
    """float64 transcription of the ridge DCA: weights 1 / #{identity > identity * L}, 21-state one-hot, N = sum(w) - sqrt(mean(w)),
    C = sum(w (x - mu)(x - mu)^T) / N + diag(lambda), P = C^-1, Frobenius norms of the 20 x 20 non-gap blocks (of the partial
    correlations for score='pc'), APC."""
    Z = np.asarray(Z, np.int64)
    M, L = Z.shape
    ns = 21
    w = 1.0 / np.array([((Z == Z[a]).sum(1) > identity * L).sum() for a in range(M)], float)
    X = np.zeros((M, L * ns))
    X[np.arange(M)[:, None], np.arange(L)[None, :] * ns + Z] = 1.0
    N = w.sum() - np.sqrt(w.mean())
    mu = (w[:, None] * X).sum(0) / N
    Xc = (X - mu) * np.sqrt(w)[:, None]
    C = Xc.T @ Xc / N
    if column_ridge:
        n = np.maximum((w[:, None] * (Z < 20)).sum(0), 1.0)
        lam = np.repeat(c / np.sqrt(n), ns)
    else:
        lam = np.full(L * ns, c / np.sqrt(w.sum()))
    P = np.linalg.inv(C + np.diag(lam))
    if score == 'pc':
        d = np.sqrt(np.diag(P))
        P = P / np.outer(d, d)
    FN = np.zeros((L, L))
    for i in range(L - 1):
        for j in range(i + 1, L):
            FN[i, j] = FN[j, i] = np.sqrt((P[i * ns:i * ns + 20, j * ns:j * ns + 20] ** 2).sum())
    return _apc(FN), w.sum()


def humphreys_reference(Z):
    """float64 transcription of the TensorFlow 1 graph of Humphreys et al. (Zhang, Humphreys, Pei et al., bioRxiv 2024.10.01.615885,
    supplementary Methods M5.3), step by step: 21-state one-hot, identities over all 21 states, weights 1 / #{identity > 0.8 L},
    tf_cov with weights, + I * 4.5 / sqrt(sum(w)), inverse, Frobenius norm of the [:-1, :-1] block, zero diagonal. Returns the raw
    scores (their output, before APC) and sum(w)."""
    Z = np.asarray(Z, np.int64)
    nr, nc = Z.shape
    ns = 21
    msa = np.eye(ns)[Z]                                                     # tf.one_hot(x, 21)
    pw = np.tensordot(msa, msa, [[1, 2], [1, 2]])
    weights = 1.0 / (pw > nc * 0.8).sum(-1)
    feat = msa.reshape(nr, nc * ns)
    num_points = weights.sum() - np.sqrt(weights.mean())                 # tf_cov(x, w)
    x_mean = (feat * weights[:, None]).sum(0, keepdims=True) / num_points
    x = (feat - x_mean) * np.sqrt(weights[:, None])
    c = x.T @ x / num_points + np.eye(nc * ns) * 4.5 / np.sqrt(weights.sum())
    w = np.linalg.inv(c).reshape(nc, ns, nc, ns)
    wi = np.sqrt((w[:, :-1, :, :-1] ** 2).sum((1, 3))) * (1 - np.eye(nc))
    return wi, weights.sum()


def humphreys_apc(s):
    """Their APC (supplementary Methods M5.3, one protein): dca_ij = s_ij - (sum_m s_im)(sum_n s_nj) / (sum_m sum_n s_mn), zero diagonal."""
    out = s - np.outer(s.sum(1), s.sum(0)) / s.sum()
    np.fill_diagonal(out, 0.0)
    return out
