"""jdca ALIGNMENT [-o OUT]: contact scores of a FASTA / A3M alignment, written as 'i j score' lines (1-based, best first)."""
import argparse
import os
import sys
import time

from . import gaussdca, humphreys_dca, ranking, read_alignment, ridge_dca
from .gauss import MODES


def fraction_or_auto(s):
    try:
        return s if s == 'auto' else float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a fraction, not {s!r}") from None


def main(argv=None):
    ap = argparse.ArgumentParser(prog='jdca', description=__doc__)
    ap.add_argument('alignment')
    ap.add_argument('-o', '--out', help='output file (default: stdout)')
    ap.add_argument('--method', choices=['gauss', 'ridge', 'humphreys'], default='gauss',
                    help='GaussDCA (default), ridge DCA, or the DCA of Humphreys et al. with their settings')
    ap.add_argument('--mode', choices=sorted(MODES), default='auto',
                    help="GaussDCA settings: 'auto' (default, GaussDCA.jl's defaults) or 'adaptive' (see README)")
    ap.add_argument('--pseudocount', type=float, default=None, help="GaussDCA pseudocount (default: the mode's)")
    ap.add_argument('--theta', type=fraction_or_auto, default=None, help="GaussDCA reweighting threshold, a fraction or 'auto' (auto mode default)")
    ap.add_argument('--max-gap-fraction', type=float, default=None,
                    help='drop sequences with more gaps (default: 0.9 in auto mode, 1.0 in adaptive mode and for ridge)')
    ap.add_argument('--ridge', type=float, default=None, help='ridge constant c (default 4.07, or 5.11 with --single-ridge)')
    ap.add_argument('--single-ridge', action='store_true', help='one ridge c / sqrt(Meff) instead of c / sqrt(n_i) per column')
    ap.add_argument('--score', choices=['pc', 'fn'], default='pc', help='ridge score: partial correlations (default) or raw block')
    ap.add_argument('--window', type=int, default=None,
                    help='windowed GaussDCA for very long chains: tiles of 2W+1 columns, pairs up to W apart (default: whole chain)')
    ap.add_argument('--min-separation', type=int, default=5, help='smallest |i - j| written (default 5)')
    ap.add_argument('--device', choices=['auto', 'server', 'gpu', 'cpu'], default='auto')
    a = ap.parse_args(argv)
    if 'jax' not in sys.modules:                       # JAX not imported yet (a command, not a call)
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')     # XLA's informational lines on loading cached programs
        if a.device == 'cpu':
            os.environ['JAX_PLATFORMS'] = 'cpu'       # keep it off the GPU entirely
    gauss_opts = (('--mode', a.mode != 'auto'), ('--pseudocount', a.pseudocount is not None), ('--theta', a.theta is not None),
                  ('--window', a.window is not None))
    ridge_opts = (('--ridge', a.ridge is not None), ('--single-ridge', a.single_ridge), ('--score', a.score != 'pc'))
    ignored = [o for o, v in {'gauss': ridge_opts, 'ridge': gauss_opts,
                              'humphreys': gauss_opts + ridge_opts + (('--max-gap-fraction', a.max_gap_fraction is not None),)}[a.method] if v]
    if ignored:
        print(f'jdca: {", ".join(ignored)} do(es) not apply to --method {a.method}; ignored', file=sys.stderr)
    if a.method == 'gauss' and a.window is not None and a.window < a.min_separation:
        print(f'jdca: --window {a.window} < --min-separation {a.min_separation}: no pair can be written', file=sys.stderr)
    t = time.perf_counter()
    try:
        if a.method == 'gauss':
            Z = read_alignment(a.alignment, MODES[a.mode]['max_gap_fraction'] if a.max_gap_fraction is None else a.max_gap_fraction)
            r = gaussdca(Z, mode=a.mode, pseudocount=a.pseudocount, theta=a.theta, window=a.window, device=a.device)
        elif a.method == 'ridge':
            Z = read_alignment(a.alignment, 1.0 if a.max_gap_fraction is None else a.max_gap_fraction)
            r = ridge_dca(Z, c=a.ridge, column_ridge=not a.single_ridge, score=a.score, device=a.device)
        else:
            Z = read_alignment(a.alignment, 1.0)
            r = humphreys_dca(Z, device=a.device)
        i, j, s = ranking(r.scores, a.min_separation)
        if len(i) == 0 and Z.shape[1] > a.min_separation:
            print('jdca: no pair could be scored', file=sys.stderr)
            return 1
        out = open(a.out, 'w') if a.out else sys.stdout
    except (OSError, ValueError, RuntimeError) as e:     # unreadable input, invalid settings, no GPU for --device gpu / server
        print(f'jdca: error: {e}', file=sys.stderr)
        return 1
    for x, y, v in zip(i + 1, j + 1, s):
        out.write(f'{x} {y} {v:e}\n')
    if a.out:
        out.close()
    print(f'jdca: M={Z.shape[0]} L={Z.shape[1]} Meff={r.meff:.1f} theta={r.theta:.4f} in {time.perf_counter() - t:.2f} s',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
