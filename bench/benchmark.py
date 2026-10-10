"""GaussDCA.jl vs jdca on AlphaFold DB targets: agreement, contact precision against the AFDB model, and wall times.

Data: bench/data/<UniProt>/ holds the AFDB model (AF-<id>-F1-model_v*.cif) and the MSA (*.a3m) whose first row is the UniProt
sequence. Every run reads the same aligned FASTA (lowercase insertions and '.' removed), because GaussDCA.jl does not accept a3m
insertions.

    python bench/benchmark.py P34884 P0A9S1 P57740 [--threads 1,8] [--cores 0-7] [--skip-jl]

Writes bench/out/<id>/ (rankings, timings) and bench/out/results.json; prints the tables used in docs/benchmark.md.
"""
import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import jdca                                                     # noqa: E402

PY = sys.executable
# NumPy asks for transparent huge pages for large arrays. Where memory is fragmented (a shared machine), those page faults can stall
# in compaction: a 28 MB array took 3 ms or 100+ ms on the benchmark machine. Every timed program runs without it, containers included.
ENV = dict(os.environ, PYTHONPATH=ROOT, XLA_PYTHON_CLIENT_PREALLOCATE='false', NUMPY_MADVISE_HUGEPAGE='0',
           APPTAINERENV_NUMPY_MADVISE_HUGEPAGE='0')


def strip_a3m(a3m, fasta):
    """A3M -> aligned FASTA: drop lowercase (insertions) and '.' from sequence lines, keep everything else."""
    drop = bytes(range(ord('a'), ord('z') + 1)) + b'.'
    opener = gzip.open if a3m.endswith('.gz') else open
    with opener(a3m, 'rb') as f, open(fasta, 'wb') as g:
        for line in f:
            g.write(line if line.startswith(b'>') else line.rstrip(b'\r\n').translate(None, drop) + b'\n')


def model(cif):
    """(sequence, CB coordinates (CA for Gly), pLDDT) of the AFDB model."""
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure as bs
    from biotite.sequence import ProteinSequence
    a = pdbx.get_structure(pdbx.CIFFile.read(cif), model=1, extra_fields=['b_factor'])
    a = a[a.hetero == False]                                          # noqa: E712
    seq, xyz, plddt = [], [], []
    for r in bs.residue_iter(a):
        pick = r[r.atom_name == ('CA' if r.res_name[0] == 'GLY' else 'CB')]
        if len(pick) == 0:
            pick = r[r.atom_name == 'CA']
        seq.append(ProteinSequence.convert_letter_3to1(r.res_name[0]))
        xyz.append(pick.coord[0])
        plddt.append(pick.b_factor[0])
    return ''.join(seq), np.array(xyz), np.array(plddt)


def precision(scores_or_rank, xyz, plddt, L, seps=(6, 12, 24), cut=8.0, conf=70.0):
    """Top-L/5, L/2, L precision for |i-j| >= sep over pairs whose two residues have pLDDT >= conf."""
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    ok = plddt >= conf
    i, j, _ = scores_or_rank
    out = {}
    for sep in seps:
        keep = (j - i >= sep) & ok[i] & ok[j]
        ii, jj = i[keep], j[keep]
        for name, k in (('L/5', L // 5), ('L/2', L // 2), ('L', L)):
            out[f'{name}@{sep}'] = float((d[ii[:k], jj[:k]] < cut).mean()) if k else float('nan')
    return out


def read_rank(path, L):
    i, j, s = np.loadtxt(path, unpack=True, ndmin=2)
    S = np.full((L, L), np.nan)
    S[i.astype(int) - 1, j.astype(int) - 1] = s
    return (i.astype(int) - 1, j.astype(int) - 1, s), S


def timed(cmd, **kw):
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV, **kw)
    if p.returncode:
        raise RuntimeError(f'{cmd}: {p.stderr[-2000:]}')
    return time.perf_counter() - t, p


def best(cmd, n=3, **kw):
    """Fastest of n runs of a command (wall seconds): robust to other jobs on a shared machine."""
    return min(timed(cmd, **kw)[0] for _ in range(n))


_STARTED = []


def ensure_server():
    """Start a jdca server unless one runs (it is stopped at exit if this script started it)."""
    from jdca import server as srv
    if srv._ask('status') is not None:
        return
    subprocess.run([PY, '-m', 'jdca.server', 'start'], env=ENV, capture_output=True)
    for _ in range(120):
        if srv._ask('status') is not None:
            break
        time.sleep(0.5)
    if not _STARTED:
        import atexit
        atexit.register(lambda: subprocess.run([PY, '-m', 'jdca.server', 'stop'], env=ENV, capture_output=True))
    _STARTED.append(True)


def jl_one_off(info):
    """GaussDCA.jl as a one-off command: Julia's start, package loading, the first (compiling) gDCA call and writing the ranking.
    run_gdca.jl's process also runs a GC, a second (warm) call and an untimed theta / Meff recomputation; those are subtracted."""
    return info['process_wall'] - info['t_gdca_warm_s'] - info['t_gc_s'] - info['t_stats_s']


def stop_server():
    """Stop the jdca server if this script started it (so it holds no GPU memory while the next target is timed)."""
    if _STARTED:
        subprocess.run([PY, '-m', 'jdca.server', 'stop'], env=ENV, capture_output=True)


def batch(fasta, call, prefix=(), n=3):
    """Per-map cost in one process: `call` (an expression of the alignment path f, returning a DCA) plus the ranking, after a first
    call; fastest of n. Runs in its own process, so this script holds no GPU memory."""
    code = (f'import sys, time, jdca\nf = sys.argv[1]\ng = lambda: jdca.ranking({call}.scores)\ng()\nts = []\n'
            f'for _ in range({n}):\n    t = time.perf_counter(); g(); ts.append(time.perf_counter() - t)\nprint(min(ts))')
    return float(timed(list(prefix) + [PY, '-c', code, fasta])[1].stdout.split()[-1])


def run_case(uid, a, res):
    d = os.path.join(HERE, 'data', uid)
    out = os.path.join(HERE, 'out', uid)
    os.makedirs(out, exist_ok=True)
    cif = sorted(glob.glob(os.path.join(d, 'AF-*-model_v*.cif')))[-1]
    a3m = sorted(glob.glob(os.path.join(d, '*.a3m')) + glob.glob(os.path.join(d, '*.a3m.gz')))[0]
    fasta = os.path.join(out, 'aln.fasta')
    strip_a3m(a3m, fasta)
    seq, xyz, plddt = model(cif)
    Z = jdca.read_alignment(fasta)
    M, L = Z.shape
    query = ''.join('ACDEFGHIKLMNPQRSTVWY-'[c] for c in Z[0])
    assert L == len(seq) and sum(x == y for x, y in zip(query, seq)) >= 0.95 * L, f'{uid}: MSA query does not match the model'
    r = res.setdefault(uid, {'M_file': sum(1 for line in open(fasta) if line.startswith('>')), 'M': M, 'L': L,
                             'plddt_ge70': float((plddt >= 70).mean())})
    print(f'== {uid}: L={L}, M={M} after the 90% gap filter, pLDDT>=70 on {r["plddt_ge70"]:.0%} of residues', flush=True)

    # ---- GaussDCA.jl (pinned image), warm call timed in-process; the cold call includes Julia's JIT
    jl = r.setdefault('gaussdca_jl', {})
    if not a.skip_jl:
        for t in a.threads:
            rk = os.path.join(out, f'jl_t{t}.txt')
            wall, p = timed(['taskset', '-c', a.cores, os.path.join(HERE, 'gaussdca_jl', 'run.sh'), fasta, rk, str(t)])
            info = json.loads([x for x in p.stderr.splitlines() if x.startswith('{')][-1])
            info.pop('hostname', None)
            for k in ('input', 'output'):
                if k in info:
                    info[k] = os.path.relpath(info[k], ROOT)
            jl[f't{t}'] = dict(info, process_wall=wall, one_off=jl_one_off(dict(info, process_wall=wall)))
            print(f'  GaussDCA.jl {t} thread(s): one-off command {jl[f"t{t}"]["one_off"]:.1f} s, batch {info["t_gdca_warm_s"]:.2f} s per map',
                  flush=True)
    rk_jl = os.path.join(out, f'jl_t{a.threads[-1]}.txt')

    # ---- jdca, timed with nothing else of ours on the GPU: one-off commands (fresh processes), then batch (a later call in one
    # process, from the file to the ranking, as GaussDCA.jl's timed gDCA(file) call), then commands with jdca-server running.
    # The CPU runs on the same cores as GaussDCA.jl.
    jd = r.setdefault('jdca', {})
    pin = ['taskset', '-c', a.cores]
    for dev in ('gpu', 'cpu'):
        cmd = [PY, '-m', 'jdca.cli', fasta, '-o', os.path.join(out, f'jdca_{dev}.txt'), '--device', dev]
        jd[f'{dev}_one_off'] = best((pin if dev == 'cpu' else []) + cmd)
    jd['gpu_batch'] = batch(fasta, 'jdca.gaussdca(f, device="gpu")')
    jd['cpu_batch'] = batch(fasta, 'jdca.gaussdca(f, device="cpu")', pin)
    jd['adaptive_gpu_batch'] = batch(fasta, 'jdca.gaussdca(jdca.read_alignment(f, 1.0), mode="adaptive", device="gpu")')
    ensure_server()
    cmd = [PY, '-m', 'jdca.cli', fasta, '-o', os.path.join(out, 'jdca_server.txt'), '--device', 'server']
    timed(cmd)                                                  # the server's first call of this size
    jd['gpu_server'] = best(cmd)
    stop_server()
    print(f'  jdca GPU: one-off command {jd["gpu_one_off"]:.2f} s, with jdca-server {jd["gpu_server"]:.2f} s, '
          f'batch {jd["gpu_batch"] * 1e3:.0f} ms per map; CPU ({a.cores}): one-off command {jd["cpu_one_off"]:.2f} s, '
          f'batch {jd["cpu_batch"]:.2f} s per map', flush=True)

    # ---- agreement with GaussDCA.jl and precision against the model
    g = jdca.gaussdca(Z, device='gpu')
    jr = jdca.ranking(g.scores, 5)
    r['jdca']['theta'], r['jdca']['meff'] = g.theta, g.meff
    if os.path.exists(rk_jl):
        rank_jl, S_jl = read_rank(rk_jl, L)
        m = ~np.isnan(S_jl)
        diff = np.abs(g.scores[m] - S_jl[m])
        top = lambda rk, k: set(zip(rk[0][:k], rk[1][:k]))
        from scipy.stats import spearmanr
        r['agreement'] = dict(max_abs=float(diff.max()), max_rel=float(diff.max() / np.abs(S_jl[m]).max()),
                              spearman=float(spearmanr(g.scores[m], S_jl[m]).statistic),
                              topL_shared=len(top(jr, L) & top(rank_jl, L)) / L)
        r['precision_jl'] = precision(rank_jl, xyz, plddt, L)
        print(f'  agreement: max |dS| {r["agreement"]["max_abs"]:.1e} (rel {r["agreement"]["max_rel"]:.1e}), '
              f'top-L shared {r["agreement"]["topL_shared"]:.1%}', flush=True)
    r['precision_jdca'] = precision(jr, xyz, plddt, L)
    Zc = jdca.read_alignment(fasta, 1.0)
    cl = jdca.gaussdca(Zc, mode='adaptive', device='gpu')
    r['precision_jdca_adaptive'] = precision(jdca.ranking(cl.scores, 5), xyz, plddt, L)
    jd['adaptive_meff'], jd['adaptive_pseudocount'] = cl.meff, jdca.gauss.adaptive_pseudocount(*Zc.shape)
    rd = jdca.ridge_dca(jdca.read_alignment(fasta, 1.0), device='gpu')
    r['precision_ridge'] = precision(jdca.ranking(rd.scores, 5), xyz, plddt, L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--threads', default='1,8', help='GaussDCA.jl thread counts (default 1,8)')
    ap.add_argument('--cores', default='0-7', help='cores for GaussDCA.jl and for jdca on the CPU (default 0-7)')
    ap.add_argument('--skip-jl', action='store_true')
    a = ap.parse_args()
    a.threads = [int(x) for x in a.threads.split(',')]
    path = os.path.join(HERE, 'out', 'results.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    from jdca import server as srv
    if srv._ask('status') is not None:
        sys.exit('benchmark.py: stop jdca-server first; every program is timed with the GPU otherwise free')
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'      # this process's own maps (agreement, precision) free their GPU memory
    for uid in a.ids:
        run_case(uid, a, res)
        json.dump(res, open(path, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
