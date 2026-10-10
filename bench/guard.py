"""Run one timed benchmark command on a shared Linux machine, watch for other programs using what it times, and say whether the run
was clean.

    python bench/guard.py [--cores] [--ncores 8] [--quiet 60] [--json OUT.json --key ID [--field guard]] [--report R.json] -- cmd ...

'{cores}' in the command is replaced by the CPUs chosen for it (with --cores). Exit status: the command's own if it failed, else 0
when the run was clean and 3 when it was not (repeat it then).

The load average is a poor guide on a shared machine (it also counts processes waiting on disk). Instead, every second:
  * GPU: the processes holding a GPU context (`nvidia-smi --query-compute-apps`, which costs nothing; `nvidia-smi pmon` would slow
    every GPU job down). Another program's process counts as active in a second when it used more than 0.1 s of CPU (a GPU job needs
    its host thread) or when it appeared during the run; another program's `nvidia-smi pmon` / `dmon` counts too.
  * CPU (--cores): per-thread CPU time from /proc, attributed to the CPU each thread last ran on. Interference = other programs' CPU
    time on a chosen physical core (either hyper-thread) in a second when the command itself ran there. Every CPU carries a little
    background work (kernel threads, desktop), so the run counts as contaminated only when the interference exceeds 30% of one CPU
    over some 10-second window.
Before starting, the guard waits for `--quiet` seconds without other GPU activity and (with --cores) picks `--ncores` physical cores
on one socket whose hyper-threads were each below 5% use meanwhile. The command runs pinned away from them (its own
`taskset -c {cores}` puts the timed CPU work back on them); the guard itself runs on the last CPU and its sibling, never chosen.
A process is the command's when it descends from it or carries JDCA_GUARD=<the guard's pid> in its environment (set for the command
and, as APPTAINERENV_JDCA_GUARD, inside Apptainer containers; it also covers a detached jdca-server).

--json OUT.json --key ID stores the verdict as OUT.json[ID][field] once the command has finished (the benchmark scripts write their
records there); --report writes the full report.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time

TICK = os.sysconf('SC_CLK_TCK')
MARK = f'JDCA_GUARD={os.getpid()}'.encode()
_MARKED = {}


def marked(pid):
    """Does pid carry this guard's environment marker? None if it is gone (cached per pid). Another user's process never does
    (its environment cannot even be read)."""
    if pid not in _MARKED:
        try:
            _MARKED[pid] = (os.stat(f'/proc/{pid}').st_uid == os.getuid()
                            and MARK in open(f'/proc/{pid}/environ', 'rb').read().split(b'\0'))
        except OSError:
            return None
    return _MARKED[pid]


def siblings():
    out = {}
    for c in range(os.cpu_count()):
        s = open(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list').read().strip()
        cpus = set()
        for part in s.split(','):
            a, _, b = part.partition('-')
            cpus |= set(range(int(a), int(b or a) + 1))
        out[c] = sorted(cpus)
    return out


def nodes():
    res = {}
    base = '/sys/devices/system/node'
    for d in (os.listdir(base) if os.path.isdir(base) else []):
        if d.startswith('node') and d[4:].isdigit():
            cpus = set()
            for part in open(f'{base}/{d}/cpulist').read().strip().split(','):
                a, _, b = part.partition('-')
                cpus |= set(range(int(a), int(b or a) + 1))
            res[int(d[4:])] = cpus
    return res or {0: set(range(os.cpu_count()))}


def cpustat():
    """{cpu: busy ticks} from /proc/stat (everything, including processes that came and went between samples)."""
    out = {}
    for line in open('/proc/stat'):
        if line.startswith('cpu') and line[3].isdigit():
            f = line.split()
            v = list(map(int, f[1:]))
            out[int(f[0][3:])] = sum(v) - v[3] - v[4]
    return out


def threads():
    """{tid: (tgid, cpu_ticks, last_cpu, comm)} and {tgid: ppid}."""
    th, pp = {}, {}
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        try:
            st = open(f'/proc/{p}/stat').read()
            pp[int(p)] = int(st[st.rindex(')') + 2:].split()[1])
            for t in os.listdir(f'/proc/{p}/task'):
                s = open(f'/proc/{p}/task/{t}/stat').read()
                r = s.rindex(')')
                f = s[r + 2:].split()
                th[int(t)] = (int(p), int(f[11]) + int(f[12]), int(f[36]), s[s.index('(') + 1:r])
        except (OSError, ValueError, IndexError):
            pass
    return th, pp


def descendants(pp, roots):
    mine = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, par in pp.items():
            if par in mine and pid not in mine:
                mine.add(pid)
                changed = True
    return mine


def cmdline(pid):
    try:
        return open(f'/proc/{pid}/cmdline').read().replace('\0', ' ').strip()
    except OSError:
        return ''


class GPUApps(threading.Thread):
    """The pids holding a GPU context, refreshed every second; whether each is the command's is decided while it is alive."""
    def __init__(self):
        super().__init__(daemon=True)
        self.p = subprocess.Popen(['nvidia-smi', '--query-compute-apps=timestamp,pid', '--format=csv,noheader', '-lms', '1000'],
                                  stdout=subprocess.PIPE, text=True)
        self.lock = threading.Lock()
        self.first = {}                     # pid -> (time first seen, ours)
        self.roots = {os.getpid()}
        self.known = {}

    def ours(self, pid):
        """True / False, or None when the process is gone before it could be inspected."""
        if pid not in self.known:
            m = marked(pid)
            if m is None:
                return None
            p = pid
            try:
                while not m and p > 1 and p not in self.roots:
                    st = open(f'/proc/{p}/stat').read()
                    p = int(st[st.rindex(')') + 2:].split()[1])
            except (OSError, ValueError):
                return None
            self.known[pid] = m or p in self.roots
        return self.known[pid]

    def run(self):
        for line in self.p.stdout:
            f = [x.strip() for x in line.split(',')]
            if len(f) == 2 and f[1].isdigit():
                pid = int(f[1])
                mine = self.ours(pid)
                if mine is None:
                    continue
                with self.lock:
                    self.first.setdefault(pid, (time.time(), mine))

    def foreign(self):
        with self.lock:
            return {pid: t for pid, (t, mine) in self.first.items() if not mine}


def monitors(th, mine):
    """Other programs' nvidia-smi pmon / dmon processes (they slow every GPU job down)."""
    out = []
    for tid, (tg, _, _, comm) in th.items():
        if tid == tg and comm == 'nvidia-smi' and tg not in mine:
            c = cmdline(tg)
            if ' pmon' in c or ' dmon' in c:
                out.append(tg)
    return out


def store(path, key, field, verdict):
    """OUT.json[key][field] = verdict (the file is rewritten whole, as the benchmark scripts do)."""
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.setdefault(key, {})[field] = verdict
    json.dump(res, open(path, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cores', action='store_true', help='pick idle physical cores for the command ({cores}) and watch them')
    ap.add_argument('--ncores', type=int, default=8)
    ap.add_argument('--quiet', type=int, default=60, help='seconds without other GPU activity before starting')
    ap.add_argument('--json', help='a benchmark JSON file to store the verdict in')
    ap.add_argument('--key', help='its record (the UniProt id)')
    ap.add_argument('--field', default='guard')
    ap.add_argument('--report', help='write the full report here')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
    if not cmd or bool(a.json) != bool(a.key):
        ap.error('a command after --, and --json together with --key')
    sib, nod = siblings(), nodes()
    own_cpus = set(sib[os.cpu_count() - 1])
    os.sched_setaffinity(0, own_cpus)
    gpu = GPUApps()
    gpu.start()
    me = os.getpid()
    time.sleep(3)

    def gpu_activity(prev, cur, t0):
        """Other programs' GPU holders that used > 0.1 s CPU between two thread snapshots, or that first appeared after t0."""
        fpids = gpu.foreign()
        cpu = collections.Counter()
        for tid, (tg, ticks, _, _) in cur.items():
            if tg in fpids:
                cpu[tg] += (ticks - prev[tid][1] if tid in prev else ticks) / TICK
        return {pid: round(cpu[pid], 2) for pid in fpids if cpu[pid] > 0.1 or fpids[pid] > t0}

    # ---- wait for a quiet window
    while True:
        t0 = time.time()
        c0, th0 = cpustat(), threads()[0]
        busy_g = collections.Counter()
        for _ in range(a.quiet):
            time.sleep(1.0)
            th1, pp = threads()
            busy_g.update(gpu_activity(th0, th1, t0).keys())
            th0 = th1
        c1 = cpustat()
        mine = descendants(pp, [me])
        use = {c: (c1[c] - c0[c]) / TICK / a.quiet for c in c1}          # of ours, only the guard runs meanwhile (on its own CPUs)
        mon = monitors(th1, mine)
        chosen = []
        if a.cores:
            for n, cpus in sorted(nod.items()):
                cores = sorted({min(sib[c]) for c in cpus} - {min(sib[c]) for c in own_cpus})
                cores.sort(key=lambda c: max(use[x] for x in sib[c]))
                pick = cores[:a.ncores]
                if len(pick) == a.ncores and all(use[x] < 0.05 for c in pick for x in sib[c]):
                    chosen = pick
                    break
        if not busy_g and not mon and (chosen or not a.cores):
            break
        print(f'guard: not quiet ({time.strftime("%T")}: other GPU seconds {sum(busy_g.values())}, monitors {mon}, '
              f'cores {"ok" if chosen or not a.cores else "busy"}); waiting', flush=True)

    watch = {x for c in chosen for x in sib[c]}
    core_of = {x: min(sib[x]) for x in watch}
    away = sorted(set(range(os.cpu_count())) - watch - own_cpus) + sorted(own_cpus)
    cores = ','.join(map(str, chosen))
    cmd = [x.replace('{cores}', cores) for x in cmd]
    token = MARK.decode().split('=')[1]
    os.environ['JDCA_GUARD'] = os.environ['APPTAINERENV_JDCA_GUARD'] = token
    print(f'guard: start {time.strftime("%T")} cores {cores or "-"}', flush=True)
    t_start = time.time()
    prev, _ = threads()
    child = subprocess.Popen(['taskset', '-c', ','.join(map(str, away))] + cmd)
    gpu.roots.add(child.pid)
    inter, box, by_comm, on_watch = [], [], collections.Counter(), 0.0
    gpu_busy, gpu_by = 0, collections.Counter()
    mon_seen = set()
    while child.poll() is None:
        time.sleep(1.0)
        cur, pp = threads()
        mine = descendants(pp, [me, child.pid])
        g = gpu_activity(prev, cur, t_start)
        if g:
            gpu_busy += 1
            for pid in g:
                gpu_by[cmdline(pid)[:60] or str(pid)] += 1
        mon_seen.update(monitors(cur, mine))
        own, foreign, who, b = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter), 0.0
        for tid, (tg, ticks, cpu, comm) in cur.items():
            d = (ticks - prev[tid][1] if tid in prev else ticks) / TICK          # a new thread: all its time so far
            if d <= 0:
                continue
            if tg in mine or (cpu in watch and marked(tg)):
                if cpu in watch:
                    own[core_of[cpu]] += d
                continue
            b += d
            if cpu in watch:
                foreign[core_of[cpu]] += d
                who[core_of[cpu]][comm] += d
        busy = [c for c in own if own[c] > 0.05]
        inter.append(sum(foreign[c] for c in busy))
        for c in busy:
            by_comm.update(who[c])
        on_watch += sum(foreign.values())
        box.append(b)
        prev = cur
    rc = child.wait()
    t_end = time.time()
    max10 = max((sum(inter[i:i + 10]) / 10 for i in range(max(len(inter) - 9, 1))), default=0.0)
    clean = rc == 0 and not gpu_busy and not mon_seen and (not a.cores or max10 < 0.3)
    verdict = dict(clean=clean, start=time.strftime('%F %T', time.localtime(t_start)), wall_s=round(t_end - t_start, 1), cores=cores,
                   other_gpu_seconds=gpu_busy, other_cpu_on_cores_max_share_10s=round(max10, 3),
                   other_cpu_machine_mean=round(sum(box) / max(len(box), 1), 2))
    report = dict(verdict, cmd=cmd, rc=rc, other_gpu_by=dict(gpu_by), other_gpu_monitors=sorted(mon_seen),
                  other_cpu_on_cores_s=round(on_watch, 2), interference_s=round(sum(inter), 2),
                  interference_by_comm=dict(by_comm.most_common(5)), loadavg_end=os.getloadavg()[0])
    if a.report:
        json.dump(report, open(a.report, 'w'), indent=1)
    if a.json and rc == 0:
        store(a.json, a.key, a.field, verdict)
    print('guard: ' + json.dumps({k: v for k, v in report.items() if k != 'cmd'}), flush=True)
    gpu.p.terminate()
    sys.exit(rc if rc else (0 if clean else 3))


if __name__ == '__main__':
    main()
