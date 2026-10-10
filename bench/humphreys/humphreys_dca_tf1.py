"""The TensorFlow 1 DCA script of Jing Zhang, Ian R. Humphreys, Jimin Pei and co-authors (Science 2025, doi:10.1126/science.adt1630;
preprint bioRxiv 2024.10.01.615885, supplementary Methods M5.3), transcribed from the supplement for the comparison in
docs/comparison.md (the PDF text layer is garbled; only whitespace and line breaks were restored). The model code is the authors'
work, reproduced here only to run their method as published; it is not part of jdca and not covered by jdca's license.

Input: an npz of uint8 MSAs (gap = 20). Output: an npz of the raw L x L scores wi (before their APC).
Usage (TF 1.15): humphreys_dca_tf1.py IN.npz OUT.npz

The MODEL CODE (tf_cov and the graph from `x = tf.placeholder` to `x_wi = ...`) is theirs, unchanged. Only the I/O harness
around it is ours:
  * timing: graph build, tf.Session() creation (CUDA context / device init), and per key the FIRST sess.run vs the later runs on the
    SAME input (REPEAT, default 2; REPEAT=1 runs each key once). The npz stores the FIRST run's scores, as the original did.
    Per-key timings also go to OUT minus '.npz' plus '.timing.json'.
  * the Session gets gpu_options.allow_growth=True so it does not reserve the whole (shared) GPU. Memory policy only, no numerics.
  * device report (PLACEMENT_CHECK=1, the default; =0 skips all of it, for timing a one-off run): the session's devices, the device
    each key op (MatrixInverse, MatMul, ...) ran on (one traced run of the first key, after the timed runs), and
    tf.test.is_gpu_available() (called last so it cannot pre-initialise CUDA before the timed first call).
"""
import os, sys, time, json
_t_import0 = time.time()
import numpy as np
import tensorflow as tf
T_IMPORT = time.time() - _t_import0

def tf_cov(x, w=None):
    if w is None:
        num_points = tf.cast(tf.shape(x)[0], tf.float32) - 1
        x_mean = tf.reduce_mean(x, axis=0, keep_dims=True)
        x = (x - x_mean)
    else:
        num_points = tf.reduce_sum(w) - tf.sqrt(tf.reduce_mean(w))
        x_mean = tf.reduce_sum(x * w[:, None], axis=0, keepdims=True) / num_points
        x = (x - x_mean) * tf.sqrt(w[:, None])
    return tf.matmul(tf.transpose(x), x) / num_points

npz = np.load(sys.argv[1]); OUT = sys.argv[2]
REPEAT = max(1, int(os.environ.get('REPEAT', '2')))                      # harness
PLACEMENT_CHECK = os.environ.get('PLACEMENT_CHECK', '1') != '0'          # harness
print(f'[harness] tf {tf.__version__}  built_with_cuda={tf.test.is_built_with_cuda()}  import {T_IMPORT:.2f}s  '
      f'NVIDIA_TF32_OVERRIDE={os.environ.get("NVIDIA_TF32_OVERRIDE", "<unset>")}  REPEAT={REPEAT}', flush=True)
_t_graph0 = time.time()                                                  # harness
with tf.Graph().as_default():
    x = tf.placeholder(tf.uint8, shape=(None, None), name='x')
    x_shape = tf.shape(x)
    x_nr = x_shape[0]
    x_nc = x_shape[1]
    x_ns = 21
    x_msa = tf.one_hot(x, x_ns)
    x_cutoff = tf.cast(x_nc, tf.float32) * 0.8
    x_pw = tf.tensordot(x_msa, x_msa, [[1, 2], [1, 2]])
    x_cut = x_pw > x_cutoff
    x_weights = 1.0 / tf.reduce_sum(tf.cast(x_cut, dtype=tf.float32), -1)
    x_feat = tf.reshape(x_msa, (x_nr, x_nc * x_ns))
    x_c = tf_cov(x_feat, x_weights) + tf.eye(x_nc * x_ns) * 4.5 / tf.sqrt(tf.reduce_sum(x_weights))
    x_c_inv = tf.linalg.inv(x_c)
    x_w = tf.reshape(x_c_inv, (x_nc, x_ns, x_nc, x_ns))
    x_wi = tf.sqrt(tf.reduce_sum(tf.square(x_w[:, :-1, :, :-1]), (1, 3))) * (1 - tf.eye(x_nc))
    # ---- harness from here on ----
    t_graph = time.time() - _t_graph0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _t_sess0 = time.time()
    with tf.Session(config=config) as sess:
        t_session = time.time() - _t_sess0
        devices = [d.name for d in sess.list_devices()] if PLACEMENT_CHECK else None     # device queries: in checked runs only
        print(f'[harness] graph build {t_graph:.2f}s  tf.Session() (device/CUDA init) {t_session:.2f}s  devices: {devices}',
              flush=True)
        results = {}
        timing = {'tf_version': tf.__version__, 'import_s': T_IMPORT, 'graph_build_s': t_graph, 'session_init_s': t_session,
                  'devices': devices, 'NVIDIA_TF32_OVERRIDE': os.environ.get('NVIDIA_TF32_OVERRIDE'), 'keys': {}}
        for ik, pair in enumerate(npz.files):
            msa = np.copy(npz[pair])
            runs, first, maxdiff = [], None, 0.0
            for r in range(REPEAT):
                t0 = time.time()
                out = sess.run(x_wi, {x: msa})
                runs.append(time.time() - t0)
                if first is None:
                    first = out
                else:
                    maxdiff = max(maxdiff, float(np.max(np.abs(out - first))))
            results[pair] = first
            rec = {'shape': list(msa.shape), 'first_run_s': runs[0], 'later_runs_s': runs[1:]}
            msg = f'{pair} {msa.shape} first {runs[0]:.3f}s'
            if ik == 0:
                rec['cold_first_call_s'] = t_graph + t_session + runs[0]
                msg += f' (cold first call incl. graph+session {rec["cold_first_call_s"]:.3f}s)'
            if REPEAT > 1:
                rec['run_to_run_max_abs_diff'] = maxdiff
                msg += f'  second {runs[1]:.3f}s'
                if REPEAT > 2:
                    msg += f'  min(later) {min(runs[1:]):.3f}s'
                msg += f'  run-to-run max|d| {maxdiff:.3g}'
            print(msg, flush=True)
            timing['keys'][pair] = rec
        if PLACEMENT_CHECK and npz.files:
            ro = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            rm = tf.RunMetadata()
            sess.run(x_wi, {x: np.copy(npz[npz.files[0]])}, options=ro, run_metadata=rm)
            per_dev, key_ops = {}, {}
            for ds in rm.step_stats.dev_stats:
                if '/stream:' in ds.device or '/memcpy' in ds.device:
                    continue                                   # CUPTI stream views duplicate the device's nodes
                per_dev[ds.device] = len(ds.node_stats)
                if not ds.device.startswith('/job:'):
                    continue                                   # '/host:CPU' = profiler host view (launches), not placement
                for ns in ds.node_stats:
                    nm = ns.node_name.split(':')[0]
                    if nm.endswith('MatrixInverse') or nm.endswith('MatMul') or nm == 'one_hot':
                        key_ops.setdefault(nm, set()).add(ds.device.split('device:')[-1])
            timing['placement_nodes_per_device'] = per_dev
            timing['placement_key_ops'] = {k: sorted(v) for k, v in key_ops.items()}
            print(f'[harness] executed nodes per device: {per_dev}', flush=True)
            print(f'[harness] key ops -> device: {timing["placement_key_ops"]}', flush=True)
        if PLACEMENT_CHECK:
            gpu_avail = tf.test.is_gpu_available()
            timing['is_gpu_available'] = bool(gpu_avail)
            print(f'[harness] tf.test.is_gpu_available() = {gpu_avail}', flush=True)
np.savez(OUT, **results)
_tj = (OUT[:-4] if OUT.endswith('.npz') else OUT) + '.timing.json'
with open(_tj, 'w') as fh:
    json.dump(timing, fh, indent=1)
print(f'[harness] wrote {OUT} and {_tj}', flush=True)
