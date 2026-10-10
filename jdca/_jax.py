"""JAX plumbing shared by the estimators: lazy import, device choice, shape buckets, ahead-of-time compiled programs."""
import os
import threading

_AOT = {}
_COMPILE = threading.Lock()         # the compile below toggles process-wide JAX settings


def jax():
    """Import JAX once, with a persistent compilation cache (unless the caller configured one)."""
    import jax as j
    if not getattr(jax, '_done', False):
        if not j.config.jax_compilation_cache_dir:
            j.config.update('jax_compilation_cache_dir',
                            os.path.expanduser(os.environ.get('JDCA_JAX_CACHE', '~/.cache/jdca/jax')))
        jax._done = True
    return j


def device(kind='auto'):
    """'auto' = the first GPU JAX can see, else the CPU; 'gpu' = a GPU or an error; 'cpu' = the CPU. JAX initialises all its backends
    at the first device query, a GPU included, unless JAX_PLATFORMS restricts them (the `jdca` command sets JAX_PLATFORMS=cpu for
    --device cpu)."""
    if kind not in ('auto', 'gpu', 'cpu'):
        raise ValueError(f"device={kind!r}: expected 'auto', 'server', 'gpu' or 'cpu'")
    j = jax()
    if kind in ('auto', 'gpu'):
        try:
            gpus = j.devices('gpu')
        except RuntimeError:
            gpus = []
        if gpus:
            return gpus[0]
        if kind == 'gpu':
            raise RuntimeError('no GPU visible to JAX')
    return j.devices('cpu')[0]


def lbucket(L):
    """Columns rounded up to 8 steps per octave (at most 12.5% padding, minimum 32): one compiled program per bucket."""
    L = int(L)
    if L <= 32:
        return 32
    s = 1 << (L.bit_length() - 4)
    return -(-L // s) * s


def rbucket(r):
    """Rows rounded up to 8 steps per octave, then to a multiple of 256 (the accumulation chunk): at most 12.5% + 255 rows of padding."""
    return max(256, -(-lbucket(r) // 256) * 256)


def aot(key, fn, specs, dev, donate=()):
    """`fn` compiled ahead of time for the (shape, dtype) `specs` on `dev`, cached in-process and in the persistent cache."""
    j = jax()
    k = (key, tuple((tuple(s[0]), str(s[1])) for s in specs), str(dev), tuple(donate))
    ex = _AOT.get(k)
    if ex is not None:
        return ex
    with _COMPILE:
        ex = _AOT.get(k)
        if ex is not None:
            return ex
        from jax.sharding import SingleDeviceSharding
        sh = SingleDeviceSharding(dev)
        sp = [j.ShapeDtypeStruct(s[0], s[1], sharding=sh) for s in specs]
        # GPU programs go to the persistent cache whatever their compile time. CPU programs are not cached: they compile quickly,
        # and XLA warns loudly when it reloads a CPU program compiled for slightly different machine features.
        cpu = dev.platform == 'cpu'
        key, val = ('jax_enable_compilation_cache', False) if cpu else ('jax_persistent_cache_min_compile_time_secs', 0.0)
        old = getattr(j.config, key)
        j.config.update(key, val)
        try:
            ex = j.jit(fn, donate_argnums=donate).lower(*sp).compile()
        finally:
            j.config.update(key, old)
        _AOT[k] = ex
    return ex
