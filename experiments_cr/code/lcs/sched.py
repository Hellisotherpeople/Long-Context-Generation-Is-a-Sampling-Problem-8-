"""Tiny GPU job scheduler.  Keeps all GPUs busy with queued jobs.

Queue layout (~/lcg-cr/queue):
  pending/<prio>_<name>.json   {"job": "/path/job.json", "gpus": 1, "model_dir": "...", "kind": "gen"}
  running/, done/, failed/
Lower prio number runs first.  A job waits until its model download is complete
(<model_dir>/.done).  Small jobs backfill idle GPUs while a larger job waits, unless the
larger job has waited longer than RESERVE_AFTER seconds (then GPUs are reserved for it).

Restart-safe: each launched job records pid/gpus in running/<name>.json and writes its exit
code to logs/jobs/<name>.rc; a restarted scheduler adopts still-running jobs.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time

Q = os.path.expanduser("~/lcg-cr/queue")
LOGS = os.path.expanduser("~/lcg-cr/logs/jobs")
NGPU = int(os.environ.get("NGPU", 8))
RESERVE_AFTER = 1800
for d in ("pending", "running", "done", "failed"):
    os.makedirs(os.path.join(Q, d), exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

CMDS = {
    "gen": [sys.executable, "-m", "lcs.gen"],
    "score": [sys.executable, "-m", "lcs.score"],
    "judge": [sys.executable, "-m", "lcs.judge"],
    "evalppl": [sys.executable, "-m", "lcs.evalppl"],
    "py": [sys.executable],
}


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def disabled_gpus():
    f = os.path.join(Q, "disabled_gpus")
    if os.path.exists(f):
        return {int(x) for x in open(f).read().split() if x.strip().isdigit()}
    return set()


def gpu_mem_used():
    """MiB used per GPU (a GPU whose memory is still held, e.g. by a dying job, is not free)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
    except Exception:
        return {}


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # zombie?
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def find_pid(job_path):
    try:
        out = subprocess.run(["pgrep", "-f", job_path], capture_output=True, text=True).stdout.split()
    except Exception:
        return None
    for p in out:
        p = int(p)
        try:
            cmd = open(f"/proc/{p}/cmdline").read()
        except OSError:
            continue
        if "lcs." in cmd and "bash" not in cmd.split("\0")[0]:
            return p
    return None


def gpus_of(pid):
    try:
        env = open(f"/proc/{pid}/environ").read().split("\0")
        for e in env:
            if e.startswith("CUDA_VISIBLE_DEVICES="):
                return [int(x) for x in e.split("=", 1)[1].split(",") if x]
    except OSError:
        pass
    return []


def main():
    running = {}  # name -> dict(pid, gpus, proc|None)
    first_seen = {}
    for name in os.listdir(os.path.join(Q, "running")):
        spec = json.load(open(os.path.join(Q, "running", name)))
        pid = spec.get("_pid") or find_pid(spec.get("job", "@@"))
        if pid and alive(pid):
            gpus = spec.get("_gpus") or gpus_of(pid)
            running[name] = {"pid": pid, "gpus": gpus, "proc": None}
            log(f"ADOPT {name} pid={pid} gpus={gpus}")
        else:
            shutil.move(os.path.join(Q, "running", name), os.path.join(Q, "pending", name))
            log(f"REQUEUE {name}")
    while True:
        for name, r in list(running.items()):
            if r["proc"] is not None:
                rc = r["proc"].poll()
                if rc is None:
                    continue
            else:
                if alive(r["pid"]):
                    continue
                rcf = os.path.join(LOGS, name.replace(".json", ".rc"))
                rc = int(open(rcf).read().strip() or 1) if os.path.exists(rcf) else 1
            rcf = os.path.join(LOGS, name.replace(".json", ".rc"))
            if os.path.exists(rcf):
                try:
                    rc = int(open(rcf).read().strip())
                except ValueError:
                    pass
            dst = "done" if rc == 0 else "failed"
            shutil.move(os.path.join(Q, "running", name), os.path.join(Q, dst, name))
            log(f"{dst.upper()} {name} rc={rc} gpus={r['gpus']}")
            del running[name]
        busy = {g for r in running.values() for g in r["gpus"]} | disabled_gpus()
        mem = gpu_mem_used()
        free = [g for g in range(NGPU) if g not in busy and mem.get(g, 0) < 2000]
        pend = sorted(os.listdir(os.path.join(Q, "pending")))
        ready = []
        for name in pend:
            try:
                spec = json.load(open(os.path.join(Q, "pending", name)))
            except Exception:
                continue
            md = spec.get("model_dir")
            if md and not os.path.exists(os.path.join(md, ".done")):
                continue
            first_seen.setdefault(name, time.time())
            ready.append((name, spec))
        reserved = 0
        for name, spec in ready:
            need = int(spec.get("gpus", 1))
            short = spec.get("kind") in ("judge", "evalppl", "score")  # minutes-long jobs may backfill reservations
            if need > len(free) - (0 if short else reserved):
                if time.time() - first_seen[name] > RESERVE_AFTER:
                    reserved += need
                continue
            gpus, free = free[:need], free[need:]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            env.update(spec.get("env", {}))
            cmd = CMDS[spec.get("kind", "gen")] + spec.get("args", [spec.get("job")])
            rcf = os.path.join(LOGS, name.replace(".json", ".rc"))
            if os.path.exists(rcf):
                os.remove(rcf)
            wrapped = ["bash", "-c", " ".join(shlex.quote(x) for x in cmd) + f"; echo $? > {shlex.quote(rcf)}"]
            lf = open(os.path.join(LOGS, name.replace(".json", ".log")), "a")
            proc = subprocess.Popen(wrapped, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=os.path.expanduser("~/lcg-cr"), start_new_session=True)
            spec["_pid"], spec["_gpus"] = proc.pid, gpus
            json.dump(spec, open(os.path.join(Q, "running", name), "w"))
            os.remove(os.path.join(Q, "pending", name))
            running[name] = {"pid": proc.pid, "gpus": gpus, "proc": proc}
            log(f"START {name} gpus={gpus}")
        with open(os.path.join(Q, "status.txt"), "w") as f:
            f.write(time.strftime("%m-%d %H:%M:%S") + "\n")
            for name, r in running.items():
                f.write(f"RUN  {r['gpus']} {name}\n")
            f.write(f"pending={len(pend)} free={free}\n")
        time.sleep(15)


if __name__ == "__main__":
    main()
