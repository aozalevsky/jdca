"""GaussDCA.jl vs jdca (GaussDCA, adaptive, ridge, humphreys_dca) vs the DCA of Humphreys et al. (their TF1 script, model code
unchanged) on the benchmark targets.

Run bench/benchmark.py first (it writes the aligned FASTA and GaussDCA.jl's rankings). Humphreys et al.'s script runs in the NGC TF1
image on the GPU (bench/humphreys/run_tf1.sh, strict FP32), optionally also on the CPU (stock TF 1.15.5, --cpu-env).

Every program is timed the same three ways, with nothing else of ours on the GPU:
  one_off  a single command from process start to exit, one map per process (fastest of 3);
  batch    the per-map cost in a process that has already computed one (fastest of 3);
  server   (jdca only) a single command while jdca-server is running (fastest of 3).
Their script is timed first for all targets; then jdca (one-off, batch, server) per target; then, untimed, the agreement and the
contact precision of every method. Their scores get their own APC (S - S_i. S_.j / S.., supplementary Methods M5.3).

    python bench/compare.py P34884 P0A9S1 P57740 [--cpu-env /path/to/tf1/python] [--cores 0-7]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import jdca                                                     # noqa: E402
from benchmark import ENV, PY, best, ensure_server, model, precision, timed      # noqa: E402

TF1 = os.path.join(HERE, 'humphreys', 'run_tf1.sh')
SIF = os.path.join(os.path.dirname(HERE), 'containers', 'tf1_gpu.sif')


def humphreys(npz, out, cmd, n=3):
    """Their script: one_off = one call per process, fastest of n; batch = the fastest of n later calls in one session (after a
    first call on the same input). The batch run also records where the key operations ran (one traced call, outside the timing)."""
    run = lambda repeat, check: ['env', f'REPEAT={repeat}', f'PLACEMENT_CHECK={check}'] + cmd + [npz, out]
    one_off = min(timed(run(1, 0))[0] for _ in range(n))
    timed(run(n + 1, 1))
    t = json.load(open(out[:-4] + '.timing.json'))
    k = t['keys']['msa']
    return dict(one_off=one_off, batch=min(k['later_runs_s']), first_call=k['first_run_s'], session_init=t['session_init_s'],
                import_s=t['import_s'], placement=t.get('placement_key_ops'))


def target(uid):
    out = os.path.join(HERE, 'out', uid)
    fasta = os.path.join(out, 'aln.fasta')
    cif = [f for f in os.listdir(os.path.join(HERE, 'data', uid)) if f.endswith('.cif')][0]
    seq, xyz, plddt = model(os.path.join(HERE, 'data', uid, cif))
    return out, fasta, xyz, plddt, jdca.read_alignment(fasta, 1.0)         # Humphreys et al. and the ridge DCA keep every sequence


def time_humphreys(uid, a, res):
    out, fasta, xyz, plddt, Z = target(uid)
    M, L = Z.shape
    r = res[uid] = dict(M=M, L=L)
    npz = os.path.join(out, 'humphreys_in.npz')
    np.savez(npz, msa=Z.astype(np.uint8))
    r['humphreys_gpu'] = humphreys(npz, os.path.join(out, 'humphreys_gpu.npz'), [TF1])
    if a.cpu_env:
        r['humphreys_cpu'] = humphreys(npz, os.path.join(out, 'humphreys_cpu.npz'), n=1,
                                       cmd=['taskset', '-c', a.cores, a.cpu_env, os.path.join(HERE, 'humphreys', 'humphreys_dca_tf1.py')])
    h = r['humphreys_gpu']
    print(f'== {uid}: L={L}, M={M}\n  Humphreys et al. (TF1, GPU): one-off command '
          f'{h["one_off"]:.2f} s (TensorFlow import {h["import_s"]:.2f} s, session start {h["session_init"]:.2f} s, first call '
          f'{h["first_call"]:.2f} s), batch {h["batch"]:.3f} s per map', flush=True)


CALLS = {'gauss': ('jdca.gaussdca(Z, device="gpu")', 0.9, []),                  # (call, max_gap_fraction, command options)
         'adaptive': ('jdca.gaussdca(Z, mode="adaptive", device="gpu")', 1.0, ['--mode', 'adaptive']),
         'humphreys': ('jdca.humphreys_dca(Z, device="gpu")', 1.0, ['--method', 'humphreys']),
         'ridge': ('jdca.ridge_dca(Z, device="gpu")', 1.0, ['--method', 'ridge'])}


def batch(fasta, call, gap):
    """Per-map cost in one process: a first call, then the fastest of 3 (in a subprocess, so this script holds no GPU memory)."""
    code = (f'import sys, time, jdca\nZ = jdca.read_alignment(sys.argv[1], {gap})\nf = lambda: {call}\nf()\nts = []\n'
            'for _ in range(3):\n    t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)\nprint(min(ts))')
    return float(timed([PY, '-c', code, fasta])[1].stdout.split()[-1])


def time_jdca(uid, res):
    """One-off commands first, with nothing else on the GPU; then batch; then commands with jdca-server running (stopped after)."""
    out, fasta, xyz, plddt, Z = target(uid)
    r = res[uid]
    cmd = lambda key: [PY, '-m', 'jdca.cli', fasta, '-o', os.path.join(out, f'jdca_{key}.txt')] + CALLS[key][2]
    t = {key: r.setdefault(f'jdca_{key}_time', {}) for key in CALLS}
    for key in CALLS:
        t[key]['one_off'] = best(cmd(key) + ['--device', 'gpu'])
    for key in CALLS:
        t[key]['batch'] = batch(fasta, *CALLS[key][:2])
    ensure_server()
    for key in CALLS:
        timed(cmd(key) + ['--device', 'server'])                               # the server's first call of this setting
        t[key]['server'] = best(cmd(key) + ['--device', 'server'])
    subprocess.run([PY, '-m', 'jdca.server', 'stop'], env=ENV, capture_output=True)
    for key in CALLS:
        print(f'  jdca {key}: one-off command {t[key]["one_off"]:.2f} s, with jdca-server {t[key]["server"]:.2f} s, '
              f'batch {t[key]["batch"] * 1e3:.0f} ms per map', flush=True)


def score(uid, res):
    """Agreement of jdca.humphreys_dca with their script's output, and contact precision of every method (no timing)."""
    out, fasta, xyz, plddt, Z = target(uid)
    L = Z.shape[1]
    r = res[uid]
    raw = np.load(os.path.join(out, 'humphreys_gpu.npz'))['msa'].astype(np.float64)
    theirs = jdca.apc(raw, offdiagonal=False)                                  # their APC, S - S_i. S_.j / S..
    mine = jdca.humphreys_dca(Z, device='gpu')
    top = lambda S: set(zip(*[x[:L] for x in jdca.ranking(S, 5)[:2]]))
    r['jdca_vs_humphreys'] = dict(max_rel=float(np.abs(jdca.humphreys_dca(Z, device='gpu', use_apc=False).scores - raw).max()
                                                / np.abs(raw).max()),
                                  topL_shared=len(top(mine.scores) & top(theirs)) / L)
    maps = {'humphreys': theirs, 'jdca_humphreys': mine.scores,
            'jdca_gauss': jdca.gaussdca(jdca.read_alignment(fasta), device='gpu').scores,
            'jdca_adaptive': jdca.gaussdca(Z, mode='adaptive', device='gpu').scores,
            'jdca_ridge': jdca.ridge_dca(Z, device='gpu').scores}
    for k, S in maps.items():
        r[f'precision_{k}'] = precision(jdca.ranking(S, 5), xyz, plddt, L)
    print(f'  {uid}: jdca.humphreys_dca vs their scores: max rel {r["jdca_vs_humphreys"]["max_rel"]:.1e}, '
          f'top-L shared {r["jdca_vs_humphreys"]["topL_shared"]:.0%}', flush=True)
    for k in maps:
        p = r[f'precision_{k}']
        print(f'    {k:14s} ' + '  '.join(f'{m} {p[m]:.2f}' for m in ('L/5@6', 'L/2@6', 'L@6', 'L/5@24', 'L/2@24', 'L@24')), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--cpu-env', default=None, help='a python with stock TF 1.15 to also time their script on the CPU')
    ap.add_argument('--cores', default='0-7')
    ap.add_argument('--skip-humphreys', action='store_true', help="reuse the stored timings and outputs of their script")
    a = ap.parse_args()
    path = os.path.join(HERE, 'out', 'compare.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    from jdca import server as srv
    if srv._ask('status') is not None:
        sys.exit('compare.py: stop jdca-server first; every program is timed with the GPU otherwise free')
    if not a.skip_humphreys:
        res['_container_start'] = best(['apptainer', 'exec', '--nv', SIF, 'true'])    # part of every one-off command of their script
        for uid in a.ids:
            time_humphreys(uid, a, res)
            json.dump(res, open(path, 'w'), indent=1)
    for uid in a.ids:
        time_jdca(uid, res)
        json.dump(res, open(path, 'w'), indent=1)
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'      # this process's own maps (scoring only) release their GPU memory
    for uid in a.ids:
        score(uid, res)
        json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
