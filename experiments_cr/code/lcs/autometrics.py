"""Refresh metrics.jsonl for every run dir with new generations (loop forever)."""
from __future__ import annotations

import glob
import json
import os
import sys
import time

from .metrics import run_metrics

RUNS = os.path.expanduser("~/lcg-cr/runs")
SKIP = ("loop", "niah", "smoke", "_old", "reason")   # these have dedicated analyses


def once(force=False):
    for run in sorted(glob.glob(f"{RUNS}/*/*")):
        exp = run.split("/")[-2]
        if any(exp.startswith(s) for s in SKIP) or not os.path.isdir(f"{run}/gen"):
            continue
        if not os.path.exists(f"{run}/meta.jsonl"):
            continue
        ngen = len(os.listdir(f"{run}/gen"))
        nmet = sum(1 for _ in open(f"{run}/metrics.jsonl")) if os.path.exists(f"{run}/metrics.jsonl") else 0
        if ngen <= nmet and not force:
            continue
        try:
            with open(f"{run}/meta.jsonl") as f:
                model_path = json.loads(f.readline())["model"]
        except Exception:
            continue
        try:
            n = run_metrics(run, model_path, nproc=48, force=force)
            print(time.strftime("%H:%M:%S"), run, n, flush=True)
        except Exception as e:
            print("ERR", run, repr(e)[:300], flush=True)


if __name__ == "__main__":
    if "--force" in sys.argv:
        once(force=True)
        sys.exit(0)
    while True:
        once()
        if "--once" in sys.argv:
            break
        time.sleep(600)
