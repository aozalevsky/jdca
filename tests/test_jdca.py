"""jdca against float64 transcriptions of the estimators (tests/reference.py) on small synthetic alignments, on the CPU and (when JAX
sees one) the GPU; plus the identity counts, the reader, the command line and the server."""
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

import jdca
from jdca import _jax as J
from reference import gaussdca_reference, humphreys_apc, humphreys_reference, ridge_reference, windowed_reference

DEVICES = ['cpu'] + (['gpu'] if J.device('auto').platform == 'gpu' else [])
TOL = 2e-5                        # measured float32 error: ~1e-7 to ~1e-6 relative


def synthetic(M, L, seed=0, gap=0.1, states=20):
    """Sequences drawn around a few ancestors, with planted co-varying column pairs and gaps (gap=0: none)."""
    rng = np.random.default_rng(seed)
    anc = rng.integers(0, states, size=(4, L))
    Z = anc[rng.integers(0, 4, M)].copy()
    mut = rng.random((M, L)) < 0.35
    Z[mut] = rng.integers(0, states, mut.sum())
    for i, j in rng.choice(L, size=(L // 6, 2), replace=False):
        Z[:, j] = (Z[:, i] * 7 + 3) % states
    if gap:
        Z[rng.random((M, L)) < gap] = 20
    return Z.astype(np.int8)


def rel(a, b):
    m = np.isfinite(b)
    return np.abs(a[m] - b[m]).max() / np.abs(b[m]).max()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('M,L', [(60, 40), (900, 30)])       # Woodbury (M + 1 < 20 L / sqrt 2) and direct routes
def test_gaussdca(device, M, L):
    Z = synthetic(M, L)
    S, meff, theta = gaussdca_reference(Z)
    r = jdca.gaussdca(Z, device=device)
    assert r.theta == pytest.approx(theta, rel=1e-12)
    assert r.meff == pytest.approx(meff, rel=1e-12)
    assert rel(r.scores, S) < TOL


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('M', [300, 2200])                  # L = 150: several GPU tiles; Woodbury and (chunked) direct routes
def test_long_chain(device, M):
    Z = synthetic(M, 150, seed=8)
    S, _, _ = gaussdca_reference(Z)
    assert rel(jdca.gaussdca(Z, device=device).scores, S) < TOL
    S, _ = ridge_reference(Z)
    assert rel(jdca.ridge_dca(Z, device=device).scores, S) < TOL


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('states', [20, 12])
def test_gaussdca_gap_free(device, states):
    """No gap and no other symbol: GaussDCA.jl takes q = the largest code present, so the last amino acid becomes the reference state."""
    Z = synthetic(150, 30, seed=6, gap=0, states=states)
    assert Z.max() == states - 1
    S, meff, _ = gaussdca_reference(Z)
    r = jdca.gaussdca(Z, device=device)
    assert r.meff == pytest.approx(meff, rel=1e-12)
    assert rel(r.scores, S) < TOL


@pytest.mark.parametrize('device', DEVICES)
def test_gaussdca_parameters(device):
    Z = synthetic(120, 35, seed=1)
    for pc, th in ((0.5, 0.2), (0.95, 0.0), (0.3, 0.3)):
        S, meff, _ = gaussdca_reference(Z, pseudocount=pc, theta=th)
        r = jdca.gaussdca(Z, pseudocount=pc, theta=th, device=device)
        assert r.meff == pytest.approx(meff, rel=1e-12)
        assert rel(r.scores, S) < TOL


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('M,L', [(60, 40), (900, 30)])
def test_gaussdca_adaptive(device, M, L):
    from jdca.gauss import adaptive_pseudocount
    Z = synthetic(M, L, seed=4)
    S, meff, _ = gaussdca_reference(Z, pseudocount=adaptive_pseudocount(M, L), theta='identity0.8', eps=1e-4, apc_offdiagonal=False,
                                    q=21)
    r = jdca.gaussdca(Z, mode='adaptive', device=device)
    assert r.meff == pytest.approx(meff, rel=1e-12)
    assert rel(r.scores, S) < TOL


@pytest.mark.parametrize('device', DEVICES)
def test_windowed(device):
    Z = synthetic(300, 61, seed=5)
    w = jdca.gaussdca(Z, window=10, device=device).scores
    sep = np.abs(np.subtract.outer(np.arange(61), np.arange(61)))
    assert np.isfinite(w[sep <= 10]).all() and np.isnan(w[sep > 10]).all()
    assert rel(w, windowed_reference(Z, 10)) < TOL
    i, j, _ = jdca.ranking(w, 5)
    assert (j - i <= 10).all() and len(i) == ((sep >= 5) & (sep <= 10)).sum() // 2
    whole = jdca.gaussdca(Z, device=device).scores                # one tile spans the chain: the whole map
    assert np.array_equal(jdca.gaussdca(Z, window=30, device=device).scores, whole)


@pytest.mark.parametrize('device', DEVICES)
def test_windowed_adaptive(device):
    Z = synthetic(300, 61, seed=9)
    assert rel(jdca.gaussdca(Z, mode='adaptive', window=10, device=device).scores, windowed_reference(Z, 10, adaptive=True)) < TOL


@pytest.mark.parametrize('device', DEVICES)
def test_ridge_direct_tiles(device):
    """Ridge on its Cholesky route (M + 1 >= 21 Lb / sqrt 2) with several tiles: the tile sums stop at the padded positions."""
    Z = synthetic(2600, 150, seed=10)
    S, meff = ridge_reference(Z)
    r = jdca.ridge_dca(Z, device=device)
    assert r.meff == pytest.approx(meff, rel=1e-12)
    assert rel(r.scores, S) < TOL


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('M,L', [(60, 40), (900, 30)])
@pytest.mark.parametrize('column_ridge,score', [(True, 'pc'), (False, 'fn'), (False, 'pc'), (True, 'fn')])
def test_ridge(device, M, L, column_ridge, score):
    Z = synthetic(M, L, seed=2)
    c = 4.07 if column_ridge else 4.5
    S, meff = ridge_reference(Z, c=c, column_ridge=column_ridge, score=score)
    r = jdca.ridge_dca(Z, c=c, column_ridge=column_ridge, score=score, device=device)
    assert r.meff == pytest.approx(meff, rel=1e-12)
    assert rel(r.scores, S) < TOL


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('M,L', [(60, 40), (900, 30)])
def test_humphreys(device, M, L):
    """humphreys_dca against a step-by-step transcription of their TensorFlow graph: raw scores (their output) and with their APC."""
    Z = synthetic(M, L, seed=4)
    raw, meff = humphreys_reference(Z)
    r = jdca.humphreys_dca(Z, device=device, use_apc=False)
    assert r.meff == pytest.approx(meff, rel=1e-12)
    assert rel(r.scores, raw) < TOL
    assert rel(jdca.humphreys_dca(Z, device=device).scores, humphreys_apc(raw)) < TOL


@pytest.mark.parametrize('device', DEVICES)
def test_identity_counts(device):
    rng = np.random.default_rng(3)
    Z = synthetic(2500, 25, seed=3, gap=0)                         # several row blocks on either device
    Z[rng.random(Z.shape) < rng.random((len(Z), 1)) * 0.8] = 20     # rows with very different gap counts: block pairs get skipped
    dev = J.device(device)
    S = (Z[:, None, :] == Z[None, :, :]).sum(-1)
    for k in (0, 1, 13, 20, 25):
        assert np.array_equal(jdca.identity_counts(Z, k, dev), (S >= k).sum(1))
    M = len(Z)
    assert jdca.mean_identity(Z, dev) == pytest.approx((S.sum() - S.trace()) / 2 / (M * (M - 1) / 2) / Z.shape[1], rel=1e-12)


@pytest.mark.parametrize('device', DEVICES)
def test_edge_cases(device):
    Z = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.int8)            # no column in common: mean identity 0 -> theta 0.5, as GaussDCA.jl
    assert jdca.sequence_weights(Z, 'auto', J.device(device))[2] == 0.5
    assert np.isfinite(jdca.gaussdca(synthetic(1, 20), device=device).scores).all()
    with pytest.raises(ValueError):
        jdca.ridge_dca(synthetic(1, 20), device=device)
    for kw in (dict(pseudocount=0), dict(pseudocount=1.5), dict(theta=2.0), dict(window=0)):
        with pytest.raises(ValueError):
            jdca.gaussdca(synthetic(50, 20), device=device, **kw)
    Z = synthetic(50, 20)
    for bad in (Z + 1, Z.astype(np.uint8) - np.uint8(1), Z[0], Z.astype(float)):     # 1-based, wrapped negative, 1-D, float
        for f in (jdca.gaussdca, jdca.ridge_dca, jdca.humphreys_dca):
            with pytest.raises(ValueError):
                f(bad, device=device)
    with pytest.raises(ValueError):                                 # pseudocount 1: every coupling is 0, APC undefined, not a NaN map
        jdca.gaussdca(Z, pseudocount=1.0, device=device)


@pytest.mark.parametrize('device', DEVICES)
def test_validation(device):
    Z = synthetic(50, 20)
    for kw in (dict(device='tpu'), dict(window=2.7), dict(window=True), dict(max_gap_fraction=0.5)):
        with pytest.raises(ValueError):
            jdca.gaussdca(Z, **dict(dict(device=device), **kw))
    for kw in (dict(identity=1.0), dict(identity=-0.1), dict(max_gap_fraction=0.5)):
        with pytest.raises(ValueError):
            jdca.ridge_dca(Z, device=device, **kw)
    assert jdca.ridge_dca(Z, device=device).theta == 0.2 and jdca.humphreys_dca(Z, device=device).theta == 0.2
    gaps = np.full((300, 50), 20, np.int8)                          # no amino acid: every coupling is exactly zero
    one = gaps.copy()
    one[:, 7] = synthetic(300, 1)[:, 0]                             # one informative column: still none
    for bad in (gaps, one):
        for kw in (dict(), dict(mode='adaptive')):
            with pytest.raises(ValueError):
                jdca.gaussdca(bad, device=device, **kw)
    two = one.copy()
    two[:, 30] = synthetic(300, 1, seed=1)[:, 0]
    assert np.isfinite(jdca.gaussdca(two, device=device).scores).all()
    long = np.full((200, 90), 20, np.int8)                          # windowed: a tile without signal gives zeros, the others scores
    long[:, :40] = synthetic(200, 40, seed=2)
    w = jdca.gaussdca(long, window=10, device=device).scores
    assert np.isfinite(w[:40, :40][np.abs(np.subtract.outer(np.arange(40), np.arange(40))) <= 10]).all()
    assert not w[60:, 60:][np.isfinite(w[60:, 60:])].any()


@pytest.mark.skipif('gpu' not in DEVICES, reason='needs a GPU')
def test_gpu_memory_fallback(monkeypatch, capsys):
    """A map that does not fit in GPU memory is computed on the CPU, with a message."""
    from jdca import gauss

    def oom(*a, **k):
        raise RuntimeError('RESOURCE_EXHAUSTED: Out of memory while trying to allocate 9.99GiB.')
    Z = synthetic(300, 40, seed=11)
    cpu = jdca.gaussdca(Z, device='cpu').scores
    monkeypatch.setattr(gauss, '_norms_jax', oom)
    assert rel(jdca.gaussdca(Z, device='gpu').scores, cpu) < 1e-12
    assert 'does not fit in GPU memory' in capsys.readouterr().err


def test_server_not_answering(monkeypatch, tmp_path, capsys):
    """A server that accepts no connection (e.g. suspended): the client gives up after TIMEOUT and computes locally."""
    import socket
    from jdca import server
    d = tmp_path / 'srv'
    d.mkdir(mode=0o700)
    path = str(d / 'server.sock')
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path)
    s.listen(1)                                                     # connections queue up, nothing ever answers
    monkeypatch.setattr(server, 'SOCKET', path)
    monkeypatch.setattr(server, 'TIMEOUT', 0.5)
    monkeypatch.setattr(server, '_CONN', {})
    try:
        t = time.perf_counter()
        r = jdca.gaussdca(synthetic(60, 20), device='auto')
        assert time.perf_counter() - t < 30 and np.isfinite(r.scores).all()
        assert 'does not answer' in capsys.readouterr().err
        assert server._ask('status') == ('timeout',)
        with pytest.raises(RuntimeError):
            jdca.gaussdca(synthetic(60, 20), device='server')
    finally:
        s.close()


def test_read_alignment(tmp_path):
    text = '>q\nACDE-FG\n>s1\nAC.dDEkk-FG\n>s2 gappy\n---x----\n>s3\nACXEBFZ\n'
    p = tmp_path / 'x.a3m'
    p.write_text(text)
    Z = jdca.read_alignment(p, max_gap_fraction=0.9)
    assert Z.tolist() == [[0, 1, 2, 3, 20, 4, 5], [0, 1, 2, 3, 20, 4, 5], [0, 1, 20, 3, 20, 4, 20]]
    assert len(jdca.read_alignment(p, max_gap_fraction=1.0)) == 4
    assert jdca.humphreys_dca(p, device='cpu').scores.shape == (7, 7)             # a path: every sequence kept, as in their script
    g = tmp_path / 'x.a3m.gz'
    g.write_bytes(gzip.compress(text.encode()))
    assert np.array_equal(jdca.read_alignment(g), Z)
    for bad, msg in (('>a\n\n>b\nAC\n', 'no match columns'), ('>a\nacd\n>b\nefg\n', 'no match columns'),
                     ('ACDE\nACDE\n', 'no FASTA'), ('>a\nAC\n>b\nACD\n', 'not aligned')):
        p.write_text(bad)
        with pytest.raises(ValueError, match=msg):
            jdca.read_alignment(p)


def test_cli(tmp_path):
    Z = synthetic(80, 30, seed=7)
    aln = tmp_path / 'aln.fasta'
    aln.write_text(''.join(f'>s{k}\n' + ''.join('ACDEFGHIKLMNPQRSTVWY-'[c] for c in row) + '\n' for k, row in enumerate(Z)))
    out = tmp_path / 'out.txt'
    from jdca.cli import main
    assert main([str(aln), '-o', str(out), '--device', 'cpu']) == 0
    rows = np.loadtxt(out)
    i, j, s = jdca.ranking(jdca.gaussdca(jdca.read_alignment(aln), device='cpu').scores)
    assert len(rows) == len(i) and np.array_equal(rows[:, 0].astype(int), i + 1) and np.array_equal(rows[:, 1].astype(int), j + 1)
    assert np.allclose(rows[:, 2], s, rtol=1e-6)
    for opts, call in ((['--method', 'humphreys'], lambda: jdca.humphreys_dca(aln, device='cpu')),
                       (['--method', 'ridge'], lambda: jdca.ridge_dca(aln, device='cpu')),
                       (['--mode', 'adaptive'], lambda: jdca.gaussdca(aln, mode='adaptive', device='cpu')),
                       (['--window', '10'], lambda: jdca.gaussdca(aln, window=10, device='cpu'))):
        assert main([str(aln), '-o', str(out), '--device', 'cpu'] + opts) == 0
        rows = np.loadtxt(out)
        i, j, s = jdca.ranking(call().scores)
        assert np.array_equal(rows[:, 0].astype(int), i + 1) and np.array_equal(rows[:, 1].astype(int), j + 1), opts
        assert np.allclose(rows[:, 2], s, rtol=1e-6), opts


@pytest.mark.skipif('gpu' not in DEVICES, reason='the server needs a GPU')
def test_server():
    """A server on a private socket: results equal the local GPU call, also with several threads in one client."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = tempfile.mkdtemp()                                        # mode 0700
    env = dict(os.environ, JDCA_SOCKET=os.path.join(d, 'server.sock'), PYTHONPATH=os.pathsep.join([root, os.path.join(root, 'tests')]))
    srv = subprocess.Popen([sys.executable, '-m', 'jdca.server', 'start', '--foreground'], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if os.path.exists(env['JDCA_SOCKET']):
                break
            time.sleep(0.5)
        code = ('import threading, numpy as np, jdca, jdca.server\n'
                'from test_jdca import synthetic\n'
                'Zs = [synthetic(200 + 40 * k, 30 + k, seed=k) for k in range(6)]\n'
                'local = [jdca.gaussdca(Z, device="gpu").scores for Z in Zs]\n'
                'got = [None] * 6\n'
                'def run(k): got[k] = jdca.gaussdca(Zs[k], device="server").scores\n'
                'ts = [threading.Thread(target=run, args=(k,)) for k in range(6)]\n'
                '[t.start() for t in ts]; [t.join() for t in ts]\n'
                'assert all(np.array_equal(a, b) for a, b in zip(got, local)), "server != local"\n'
                'Z = Zs[5]\n'
                'for f in (lambda d: jdca.ridge_dca(Z, device=d), lambda d: jdca.humphreys_dca(Z, device=d),\n'
                '          lambda d: jdca.gaussdca(Z, mode="adaptive", window=8, device=d)):\n'
                '    assert np.array_equal(f("server").scores, f("gpu").scores, equal_nan=True), "server != local"\n'
                'try:\n'
                '    jdca.server.request("gaussdca", Z, dict(mode="bogus"), required=True)\n'
                'except ValueError as e:\n'
                '    assert "bogus" in str(e)\n'
                'else:\n'
                '    raise AssertionError("no error reply")\n'
                'assert np.array_equal(jdca.gaussdca(Z, device="server").scores, local[5]), "connection unusable after an error"\n'
                'print("ok")\n')
        p = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, cwd=root)
        assert p.stdout.strip().endswith('ok'), p.stderr[-2000:]
    finally:
        stop = subprocess.run([sys.executable, '-m', 'jdca.server', 'stop'], env=env, capture_output=True, text=True)
        srv.wait(timeout=60)
        shutil.rmtree(d, ignore_errors=True)
    assert stop.stdout.strip() == 'stopped'                       # `stop` returns once the server has exited: a restart can follow
