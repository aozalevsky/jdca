"""A resident GPU process: it owns the GPU and keeps the compiled programs loaded, so a short-lived client skips GPU start-up.

A fresh process pays about a second and a half before its first map (GPU initialisation, the first execution of each program, loading
the compiled programs); for a typical alignment that is most of the time. Clients send the int8 alignment and the parameters over a
Unix socket and get back exactly what the same call computes locally on the server's GPU. At connection the client and the server
compare a fingerprint of the jdca sources and the Python environment; on a mismatch (e.g. jdca was edited after the server started)
the client computes locally and says so.

The socket lives in a directory only its owner can enter ($XDG_RUNTIME_DIR/jdca, else /tmp/jdca-<uid>, mode 0700). On Linux both ends
also check that the peer runs as the same user before anything is read from the socket. A server that does not answer within
`TIMEOUT` seconds (e.g. a suspended process) is treated as absent: the client computes locally.

    jdca-server start [--foreground]   |   jdca-server status   |   jdca-server stop
"""
import argparse
import hashlib
import os
import pickle
import signal
import socket
import stat
import struct
import sys
import threading
import time


def _default_socket():
    run = os.environ.get('XDG_RUNTIME_DIR')
    base = os.path.join(run, 'jdca') if run and os.path.isdir(run) else f'/tmp/jdca-{os.getuid()}'
    return os.path.join(base, 'server.sock')


SOCKET = os.environ.get('JDCA_SOCKET') or _default_socket()
LOG = os.path.splitext(SOCKET)[0] + '.log'
TIMEOUT = 5.0                       # seconds for connecting and the server's first reply (a map itself may take as long as it needs)
_CONN = {}                          # pid -> (socket, lock), or ('unusable', socket file identity)


def fingerprint():
    h = hashlib.blake2b(digest_size=8)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(os.listdir(here)):
        if name.endswith('.py'):
            with open(os.path.join(here, name), 'rb') as f:
                h.update(name.encode() + f.read())
    h.update(sys.prefix.encode())
    return h.hexdigest()


def _private_dir(create=False):
    """The socket's directory, if it is ours and closed to everyone else (created with mode 0700 when `create`)."""
    d = os.path.dirname(SOCKET)
    if create:
        os.makedirs(d, mode=0o700, exist_ok=True)
    try:
        st = os.lstat(d)                                # a symlink is not accepted
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and not st.st_mode & (stat.S_IRWXG | stat.S_IRWXO)


def _same_user(s):
    """The peer of a connected Unix socket runs as this user (Linux SO_PEERCRED; elsewhere the private directory has to do)."""
    if not hasattr(socket, 'SO_PEERCRED'):
        return True
    _, uid, _ = struct.unpack('3i', s.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i')))
    return uid == os.getuid()


def _send(s, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    s.sendall(struct.pack('!Q', len(data)) + data)


def _recv(s):
    def exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = s.recv(min(n - len(buf), 1 << 24))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)
    head = exact(8)
    if head is None:
        return None
    data = exact(struct.unpack('!Q', head)[0])
    return None if data is None else pickle.loads(data)


def _warn(msg):
    print(f'jdca: {msg}', file=sys.stderr, flush=True)


def _connect():
    """A connected, verified socket to the server, or None."""
    if not os.path.exists(SOCKET):
        return None
    if not _private_dir():
        _warn(f'{os.path.dirname(SOCKET)} is not a private directory of this user; not using the server there')
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT)
    try:
        s.connect(SOCKET)
        if not _same_user(s):
            s.close()
            _warn(f'the server at {SOCKET} runs as another user; computing locally')
            return None
        _send(s, ('hello', fingerprint()))
        r = _recv(s)
    except socket.timeout:
        s.close()
        _warn(f'the server at {SOCKET} does not answer (is it suspended?); computing locally')
        return None
    except OSError as e:
        s.close()
        _warn(f'the server at {SOCKET} is unreachable ({type(e).__name__}); computing locally')
        return None
    if r is not None and r[0] == 'ok':
        s.settimeout(None)                              # a map takes as long as it takes
        return s
    s.close()
    _warn(f'the server at {SOCKET} runs different jdca sources or another Python environment; computing locally '
          f'(restart it: jdca-server stop; jdca-server start)')
    return None


def _socket_id():
    try:
        st = os.stat(SOCKET)
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns


def _connection():
    """This process's connection to the server, or None. A server found unusable (another user, other sources) is not retried
    until its socket file changes (a restart); a server started after this process's first call is found at the next call."""
    pid = os.getpid()                                   # a forked child never reuses its parent's socket
    c = _CONN.get(pid)
    if c is not None and c[0] != 'unusable':
        return c
    sid = _socket_id()
    if sid is None or c == ('unusable', sid):
        return None
    s = _connect()
    _CONN[pid] = (s, threading.Lock()) if s is not None else ('unusable', sid)
    return _CONN[pid] if s is not None else None


def _drop(c):
    if _CONN.get(os.getpid()) is c:
        del _CONN[os.getpid()]
    try:
        c[0].close()
    except OSError:
        pass


def request(method, Z, kwargs, required=False):
    """Run `method(Z, **kwargs)` on the server. Returns the result tuple, or None when no server is usable (computed locally then).
    Thread-safe: one request at a time per connection."""
    c = _connection()
    if c is None:
        if required:
            raise RuntimeError(f'no usable jdca server at {SOCKET}')
        return None
    s, lock = c
    with lock:
        try:
            _send(s, ('run', method, Z, kwargs))
            r = _recv(s)
        except OSError:
            r = None
        except BaseException:                           # interrupted mid-request: its reply must never reach the next request
            _drop(c)
            raise
    if r is None:
        _drop(c)                                        # reconnect on the next call (e.g. after a server restart)
        _warn('lost the server connection; computing locally')
        if required:
            raise RuntimeError('jdca server connection lost')
        return None
    if r[0] == 'ok':
        return r[1]
    if r[0] == 'linalg':
        import numpy as np
        raise np.linalg.LinAlgError(r[1])
    if r[0] == 'value':
        raise ValueError(r[1])
    raise RuntimeError(f'jdca server: {r[1]}')


class Server:
    def __init__(self):
        from . import _jax as J
        self.dev = J.device('gpu')
        self.fp = fingerprint()
        self.lock = threading.Lock()
        self.running = True
        self.stats = {'maps': 0, 'seconds': 0.0, 'errors': 0, 'started': time.time()}

    def warm(self):
        """Create the GPU context and the BLAS / solver handles once."""
        import numpy as np
        from .gauss import gaussdca
        t = time.perf_counter()
        gaussdca(np.random.default_rng(0).integers(0, 21, size=(300, 64)).astype(np.int8), device='gpu')
        print(f'warm-up {time.perf_counter() - t:.2f} s on {self.dev}', flush=True)

    def handle(self, conn):
        import numpy as np
        from .gauss import gaussdca
        from .ridge import ridge_dca
        methods = {'gaussdca': gaussdca, 'ridge_dca': ridge_dca}
        try:
            if not _same_user(conn):
                return
            while True:
                msg = _recv(conn)
                if msg is None:
                    break
                try:
                    if msg[0] == 'hello':
                        _send(conn, ('ok', None) if msg[1] == self.fp else ('mismatch', self.fp))
                    elif msg[0] == 'run':
                        _, method, Z, kw = msg
                        t = time.perf_counter()
                        with self.lock:
                            out = methods[method](Z, device='gpu', **kw)
                        self.stats['maps'] += 1
                        self.stats['seconds'] += time.perf_counter() - t
                        _send(conn, ('ok', tuple(out)))
                    elif msg[0] == 'status':
                        _send(conn, ('ok', dict(self.stats, pid=os.getpid(), device=str(self.dev), fingerprint=self.fp)))
                    elif msg[0] == 'stop':
                        _send(conn, ('ok', None))
                        self.running = False
                    else:
                        _send(conn, ('error', f'unknown request {msg[0]!r}'))
                except np.linalg.LinAlgError as e:
                    _send(conn, ('linalg', str(e)))
                except ValueError as e:
                    _send(conn, ('value', str(e)))
                except Exception as e:
                    self.stats['errors'] += 1
                    _send(conn, ('error', f'{type(e).__name__}: {e}'))
        except OSError:
            pass
        finally:
            conn.close()

    def serve(self):
        if os.path.exists(SOCKET):
            os.unlink(SOCKET)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            srv.bind(SOCKET)                            # created 0600, inside the 0700 directory
        finally:
            os.umask(old)
        srv.listen(64)
        srv.settimeout(1.0)
        print(f'listening on {SOCKET} (pid {os.getpid()})', flush=True)
        while self.running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()
        srv.close()
        if os.path.exists(SOCKET):
            os.unlink(SOCKET)


def _ask(op):
    """The server's reply to `op` ('status' / 'stop'); None when there is no server; ('timeout',) when it does not answer."""
    if not os.path.exists(SOCKET) or not _private_dir():
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT)
    try:
        s.connect(SOCKET)
        if not _same_user(s):
            return None
        _send(s, (op,))
        return _recv(s)
    except socket.timeout:
        return ('timeout',)
    except OSError:
        return None
    finally:
        s.close()


def _server_pid():
    """The pid the server wrote into its lock file, or None."""
    try:
        with open(os.path.splitext(SOCKET)[0] + '.lock') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _not_answering():
    pid = _server_pid()
    print(f'the jdca server at {SOCKET} does not answer' + (f' (pid {pid}: is it suspended? kill {pid} stops it)' if pid else ''),
          file=sys.stderr)


def _wait_exit(timeout=30.0):
    """Wait until the server process has exited: it holds the lock file's flock for its whole lifetime."""
    import fcntl
    t = time.time() + timeout
    with open(os.path.splitext(SOCKET)[0] + '.lock', 'a') as f:
        while time.time() < t:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                time.sleep(0.05)
                continue
            fcntl.flock(f, fcntl.LOCK_UN)
            return True
    return False


def _wait_started(pid, timeout=120.0):
    """In the parent of a background start: wait until the server answers, or say why it stopped."""
    t = time.time() + timeout
    while time.time() < t:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            if os.waitstatus_to_exitcode(status) == 0:
                print('another jdca server is starting or running')
                return 0
            try:
                with open(LOG) as f:
                    last = [x for x in f.read().splitlines() if x.strip()][-1:]
            except OSError:
                last = []
            print('the jdca server did not start' + (f': {last[0]}' if last else '') + f' (log {LOG})', file=sys.stderr)
            return 1
        r = _ask('status')
        if r is not None and r[0] == 'ok':
            print(f'jdca server started (pid {pid}, log {LOG})')
            return 0
        time.sleep(0.2)
    print(f'the jdca server (pid {pid}) is still starting (log {LOG})')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='jdca-server', description='resident jdca GPU server')
    ap.add_argument('command', choices=['start', 'stop', 'status'])
    ap.add_argument('--foreground', action='store_true', help='do not detach (start)')
    a = ap.parse_args(argv)
    if a.command == 'status':
        r = _ask('status')
        if r is None:
            print(f'no jdca server at {SOCKET}')
            return 1
        if r[0] == 'timeout':
            _not_answering()
            return 1
        st = r[1]
        same = 'matches' if st['fingerprint'] == fingerprint() else 'DIFFERS from these sources or this environment (restart it)'
        print(f"jdca server pid {st['pid']} on {st['device']}, up {(time.time() - st['started']) / 60:.1f} min, "
              f"{st['maps']} maps in {st['seconds']:.1f} s, {st['errors']} errors; fingerprint {same}")
        return 0
    if a.command == 'stop':
        r = _ask('stop')
        if r is None:
            print(f'no jdca server at {SOCKET}')
            return 0
        if r[0] == 'timeout':
            _not_answering()
            return 1
        print('stopped' if _wait_exit() else 'stop requested; the server is still shutting down')
        return 0
    if not _private_dir(create=True):
        print(f'{os.path.dirname(SOCKET)} exists but is not a private directory of this user; refusing to start', file=sys.stderr)
        return 1
    r = _ask('status')
    if r is not None:
        if r[0] == 'timeout':
            _not_answering()
            return 1
        print(f'a jdca server is already running at {SOCKET}')
        return 0
    if not a.foreground:
        pid = os.fork()
        if pid:
            return _wait_started(pid)
        os.setsid()
        fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        for k in (1, 2):
            os.dup2(fd, k)
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
        sys.stdout = os.fdopen(1, 'w', buffering=1)
        sys.stderr = os.fdopen(2, 'w', buffering=1)
    import fcntl
    lockf = open(os.path.splitext(SOCKET)[0] + '.lock', 'a')         # not truncated before the lock is ours (it holds the pid)
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)  # held for the server's lifetime: a second `start` gives way
    except OSError:
        print('another jdca server is starting or running; exiting', flush=True)
        return 0
    lockf.truncate(0)
    lockf.write(f'{os.getpid()}\n')
    lockf.flush()
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    try:
        server = Server()
    except RuntimeError as e:                           # no GPU visible to JAX
        print(f'jdca-server: {e}', file=sys.stderr, flush=True)
        return 1
    signal.signal(signal.SIGTERM, lambda *_: setattr(server, 'running', False))
    signal.signal(signal.SIGINT, lambda *_: setattr(server, 'running', False))
    server.warm()
    server.serve()
    return 0


if __name__ == '__main__':
    sys.exit(main())
