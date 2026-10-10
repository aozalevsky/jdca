"""The windowed map against the whole-chain map (docs/windowed.md): rank correlations, contact precision on the pairs both score, times.

Data as for benchmark.py: bench/data/<UniProt>/ with the AFDB model and MSA. Writes bench/out/windowed.json.

    python bench/windowed.py P0A9S1 P57740 P49588 P53396 P29474 P17301 [--window 100]
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')      # as in benchmark.py: no huge-page stalls in the timed calls
import numpy as np                                              # noqa: E402
from scipy.stats import spearmanr                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import jdca                                                     # noqa: E402
from benchmark import model, precision, strip_a3m              # noqa: E402


def timed(fn, n=3):
    """The result and the fastest of n calls."""
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--window', type=int, default=100)
    a = ap.parse_args()
    w0 = a.window
    path = os.path.join(HERE, 'out', 'windowed.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    for uid in a.ids:
        d, out = os.path.join(HERE, 'data', uid), os.path.join(HERE, 'out', uid)
        os.makedirs(out, exist_ok=True)
        fasta = os.path.join(out, 'aln.fasta')
        if not os.path.exists(fasta):
            strip_a3m(sorted(glob.glob(os.path.join(d, '*.a3m')))[0], fasta)
        seq, xyz, plddt = model(sorted(glob.glob(os.path.join(d, 'AF-*-model_v*.cif')))[-1])
        for mode in ('auto', 'adaptive'):
            Z = jdca.read_alignment(fasta, 0.9 if mode == 'auto' else 1.0)
            M, L = Z.shape
            # both maps timed the same way: a first call (compilation, or loading from the persistent cache), then the fastest of 3
            jdca.gaussdca(Z, mode=mode, window=w0, device='gpu')
            win, t_win = timed(lambda: jdca.gaussdca(Z, mode=mode, window=w0, device='gpu'))
            jdca.gaussdca(Z, mode=mode, device='gpu')                                  # falls back to the CPU if it does not fit
            whole, t_whole = timed(lambda: jdca.gaussdca(Z, mode=mode, device='gpu'))
            sep = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
            iu = np.triu_indices(L, 1)
            band = lambda k: (iu[1] - iu[0]) <= k
            rho = {k: float(spearmanr(whole.scores[iu][band(k)], win.scores[iu][band(k)]).statistic) for k in (12, w0)}
            G = np.where(sep <= w0, whole.scores, np.nan)
            r = res.setdefault(uid, {}).setdefault(mode, {})
            r.update(L=L, M=M, meff=whole.meff, meff_per_L=whole.meff / L, window=w0, t_window=t_win, t_whole=t_whole,
                     rho_12=rho[12], rho_window=rho[w0],
                     precision_whole=precision(jdca.ranking(G, 5), xyz, plddt, L),
                     precision_window=precision(jdca.ranking(win.scores, 5), xyz, plddt, L))
            print(f'{uid} {mode:8s} L={L} M={M} Meff/L={r["meff_per_L"]:.2f} rho<=12 {rho[12]:.3f} rho<={w0} {rho[w0]:.3f} '
                  f'whole {t_whole:.1f}s window {t_win:.2f}s | top-L/5 @6 {r["precision_whole"]["L/5@6"]:.2f} -> '
                  f'{r["precision_window"]["L/5@6"]:.2f}, @24 {r["precision_whole"]["L/5@24"]:.2f} -> '
                  f'{r["precision_window"]["L/5@24"]:.2f}', flush=True)
            json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
