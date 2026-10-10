"""Alignment input: FASTA / A3M -> an (M, L) int8 matrix of state codes, with the conventions of GaussDCA.jl."""
import gzip

import numpy as np

AMINO = b'ACDEFGHIKLMNPQRSTVWY'
Q = 21                      # 20 amino acids + one state for the gap and every other symbol
GAP = Q - 1

_CODE = np.full(256, GAP, np.int8)
for _i, _c in enumerate(AMINO):
    _CODE[_c] = _i
_INSERT = bytes(range(ord('a'), ord('z') + 1)) + b'.'


def _records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    seq = None
    with opener(path, 'rb') as f:
        for line in f:
            line = line.strip()
            if line.startswith(b'>'):
                if seq is not None:
                    yield b''.join(seq)
                seq = []
            elif line and seq is not None:
                seq.append(line)
    if seq is not None:
        yield b''.join(seq)


def read_alignment(path, max_gap_fraction=0.9):
    """Read a FASTA or A3M alignment (optionally gzipped).

    Lowercase letters and '.' are insertions and are dropped; the 20 standard amino acids map to codes 0..19 (in the order
    ACDEFGHIKLMNPQRSTVWY) and the gap '-' and every other symbol to code 20. A sequence whose fraction of '-' exceeds
    `max_gap_fraction` is discarded (1.0 keeps all). These are the conventions of GaussDCA.jl's `read_fasta_alignment`.
    Returns the (M, L) int8 matrix; the first row is the first sequence of the file (the query of an A3M) if it passes the filter.
    """
    rows, L = [], None
    for s in _records(path):
        s = s.translate(None, _INSERT)
        if L is None:
            L = len(s)
            if L == 0:
                raise ValueError(f'{path}: the first sequence has no match columns (empty, or only insertions)')
        elif len(s) != L:
            raise ValueError(f'{path}: sequences are not aligned ({len(s)} match states vs {L})')
        if s.count(b'-') / L <= max_gap_fraction:          # the same float test as GaussDCA.jl
            rows.append(_CODE[np.frombuffer(s, np.uint8)])
    if L is None:
        raise ValueError(f'{path}: no FASTA / A3M record (no line starting with ">")')
    if not rows:
        raise ValueError(f'{path}: no sequence passed max_gap_fraction={max_gap_fraction}')
    return np.stack(rows)


def is_path(alignment):
    return isinstance(alignment, (str, bytes)) or hasattr(alignment, '__fspath__')


def as_codes(alignment, max_gap_fraction):
    """A FASTA / A3M path -> `read_alignment(path, max_gap_fraction)`; an array -> the (M, L) int8 code matrix, used as given (no gap
    filter) after checking it holds jdca's codes (0..19 the amino acids, 20 the gap; 0-based, unlike GaussDCA.jl's 1..21)."""
    if is_path(alignment):
        return read_alignment(alignment, max_gap_fraction)
    a = np.asarray(alignment)
    if a.ndim != 2 or a.size == 0 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f'expected a non-empty (M, L) integer matrix of state codes, got shape {a.shape}, dtype {a.dtype}')
    lo, hi = int(a.min()), int(a.max())
    if lo < 0 or hi > GAP:
        raise ValueError(f'state codes must lie in 0..{GAP} (0-based, gap = {GAP}); this matrix holds {lo}..{hi}')
    return a.astype(np.int8, copy=False)
