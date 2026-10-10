"""Sequence reweighting, exact: identity counts between all pairs of sequences.

S[a, b] = the number of columns where sequences a and b carry the same state (a gap matching a gap counts). S is an integer and is
computed exactly, over the upper triangle of row-block pairs only, with one of two encodings:
  GPU: int8 one-hots, S = their dot product with int32 accumulation (tensor cores);
  CPU: sequences packed in 5-bit fields, six per uint32, compared with XOR; the per-field mismatch flags of up to 31 words are added
       in place and collapsed once (DCAUtils.jl's trick, no popcount) -- about twice as fast as the int8 products on a CPU.
Rows are visited sorted by their gap count: sequences a, b differ in at least |g_a - g_b| columns, so a block pair whose gap counts are
too far apart to reach the identity threshold is skipped without changing any count.
"""
import numpy as np

from . import _jax as J
from .alignment import GAP, Q

_BLOCK = {'gpu': 2048, 'cpu': 1024}         # rows per block
_BLOCK_BYTES = 256 << 20                    # max bytes of one block's int8 one-hot (GPU)
_CACHE_BYTES = 2 << 30                      # one-hots of all blocks are kept on the device while they fit (GPU)
_FIELD, _PER = 5, 6                         # bits per residue, residues per uint32 (CPU)
_FLAGS = sum(1 << (_FIELD * k) for k in range(_PER))


def _onehot(B, Lk):
    j = J.jax()

    def f(codes):
        return j.nn.one_hot(codes, Q, dtype=j.numpy.int8).reshape(B, Lk * Q)      # code Q (padding) -> all zeros
    return f


def _matches(encoding):
    """(a, b blocks, L) -> (B, B) int32 S."""
    j = J.jax()
    jnp = j.numpy
    if encoding == 'onehot':
        return lambda a, b, L: j.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32)

    def bits(a, b, L):
        y = a[:, None, :] ^ b[None, :, :]
        t = (y | (y >> 1) | (y >> 2) | (y >> 3) | (y >> 4)) & jnp.uint32(_FLAGS)     # bit 0 of each field = "fields differ"
        W = t.shape[-1]
        t = jnp.pad(t, ((0, 0), (0, 0), (0, (-W) % 31))).reshape(t.shape[0], t.shape[1], -1, 31).sum(-1, dtype=jnp.uint32)
        d = sum((t >> (_FIELD * k)) & jnp.uint32(31) for k in range(_PER)).astype(jnp.int32).sum(-1)
        return L - d                                                                   # padding columns are 0 in both: equal
    return bits


def _pair(mode, matches):
    j = J.jax()
    jnp = j.numpy

    def hits(a, b, va, vb, thr, L):          # per row: #partners with S >= thr (both directions of the block pair)
        h = (matches(a, b, L) >= thr) & va[:, None] & vb[None, :]
        return h.sum(axis=1, dtype=jnp.int32), h.sum(axis=0, dtype=jnp.int32)

    def sums(a, b, va, vb, thr, L):          # per row: sum of S over the block pair (thr unused)
        s = jnp.where(va[:, None] & vb[None, :], matches(a, b, L), 0)
        return s.sum(axis=1), s.sum(axis=0)
    return hits if mode == 'hits' else sums


def _block_pairs(Z, dev, mode, min_matches=0):
    """Accumulate the per-row results of `_pair(mode)` over the upper triangle of row-block pairs. Returns int64 per original row
    (for 'sums' the totals over ALL partners b, each block pair counted from both sides)."""
    j = J.jax()
    Z = np.asarray(Z)
    M, L = Z.shape
    cpu = dev.platform == 'cpu'
    Lk = J.lbucket(L)
    B = int(max(64, min(_BLOCK['cpu' if cpu else 'gpu'], _BLOCK_BYTES // (Lk * Q))))
    nblk = -(-M // B)
    gaps = (Z == GAP).sum(axis=1)
    order = np.argsort(gaps, kind='stable')
    gs = gaps[order]
    lo = [int(gs[i * B]) for i in range(nblk)]
    hi = [int(gs[min((i + 1) * B, M) - 1]) for i in range(nblk)]
    valid = np.zeros(nblk * B, bool)
    valid[:M] = True
    i32 = np.int32
    if cpu:
        W = -(-Lk // _PER)
        P = np.zeros((nblk * B, W * _PER), np.uint32)
        P[:M, :L] = Z[order]
        P[M:] = 31                                                 # padding rows: a field value no residue has
        P = (P.reshape(-1, W, _PER) << (_FIELD * np.arange(_PER, dtype=np.uint32))).sum(-1, dtype=np.uint32)
        spec = ((B, W), np.uint32)
        block = lambda i: j.device_put(P[i * B:(i + 1) * B], dev)
    else:
        codes = np.full((nblk * B, Lk), Q, i32)
        codes[:M, :L] = Z[order]
        e_oh = J.aot(('oh', B, Lk), _onehot(B, Lk), [((B, Lk), i32)], dev)
        spec = ((B, Lk * Q), np.int8)
        block = lambda i: e_oh(j.device_put(codes[i * B:(i + 1) * B], dev))
    enc = 'bits' if cpu else 'onehot'
    e_pr = J.aot((mode, enc, B, Lk), _pair(mode, _matches(enc)), [spec, spec, ((B,), np.bool_), ((B,), np.bool_), ((), i32), ((), i32)],
                 dev)
    thr, Ld = j.device_put(i32(min_matches), dev), j.device_put(i32(L), dev)
    keep = cpu or nblk * B * Lk * Q <= _CACHE_BYTES
    cache = {}

    def blk(i):
        if i in cache:
            return cache[i]
        x = (block(i), j.device_put(valid[i * B:(i + 1) * B], dev))
        if keep:
            cache[i] = x
        return x
    acc = [None] * nblk
    for i in range(nblk):
        a, va = blk(i)
        for k in range(i, nblk):
            if mode == 'hits' and min_matches > L - (lo[k] - hi[i]):
                break                                # gap counts only drift further apart for larger k
            b, vb = (a, va) if k == i else blk(k)
            r, c = e_pr(a, b, va, vb, thr, Ld)
            if mode == 'sums':                       # identity sums can exceed int32 over many block pairs: add them on the host
                r, c = np.asarray(r, np.int64), np.asarray(c, np.int64)
            acc[i] = r if acc[i] is None else acc[i] + r
            if k != i:
                acc[k] = c if acc[k] is None else acc[k] + c
    n = np.concatenate([np.asarray(x, np.int64) if x is not None else np.zeros(B, np.int64) for x in acc])
    out = np.empty(M, np.int64)
    out[order] = n[:M]
    return out


def identity_counts(Z, min_matches, dev=None):
    """For each sequence a: the number of sequences b (a included) with at least `min_matches` identical columns."""
    if min_matches <= 0:                                  # every pair qualifies (and padding rows must not)
        return np.full(len(Z), len(Z), np.int64)
    return _block_pairs(Z, dev or J.device(), 'hits', int(min_matches))


def mean_identity(Z, dev=None):
    """The mean fraction of identical columns over all pairs of distinct sequences (GaussDCA's θ = 'auto' statistic)."""
    M, L = np.shape(Z)
    if M < 2:
        return 1.0
    total = int(_block_pairs(Z, dev or J.device(), 'sums').sum())        # sum over ordered pairs incl. a == b
    return (total - M * L) / 2 / (0.5 * M * (M - 1)) / L


def sequence_weights(Z, theta='auto', dev=None):
    """GaussDCA.jl's reweighting. θ = 'auto' is min(0.5, 0.38 * 0.32 / mean identity); a sequence's weight is 1 / #{sequences
    (itself included) at Hamming distance < floor(θ L)}. Returns (weights, Meff = their sum, θ)."""
    Z = np.asarray(Z)
    M, L = Z.shape
    dev = dev or J.device()
    if theta == 'auto':
        mi = mean_identity(Z, dev)
        theta = 0.5 if mi == 0 else min(0.5, 0.38 * 0.32 / mi)        # GaussDCA.jl: min(0.5, 0.1216 / 0) = 0.5
    theta = float(theta)
    thresh = np.floor(theta * L)
    if thresh <= 0:
        w = np.ones(M)
    else:
        w = 1.0 / identity_counts(Z, L - int(thresh) + 1, dev)            # distance < thresh  <=>  matches >= L - thresh + 1
    return w, float(w.sum()), theta
