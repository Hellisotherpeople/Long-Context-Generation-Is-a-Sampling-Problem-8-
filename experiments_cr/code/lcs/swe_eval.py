"""Daemon: evaluate finished SWE-bench agent runs with the official harness (agent venv, CPU + docker).

For every runs/swe/<tag>/<config>/ with .agent_done and no .eval_done, runs swebench.harness.run_evaluation on its
preds.json against the locally committed (pruned) instance images and writes report.json + .eval_done.
python -m lcs.swe_eval [--once] [--workers 24] [--parallel 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

H = os.path.expanduser("~/lcg-cr")


def todo():
    out = []
    for d in sorted(glob.glob(f"{H}/runs/swe/*/*/")):
        d = d.rstrip("/")
        if (os.path.exists(f"{d}/.agent_done") and os.path.exists(f"{d}/preds.json")
                and not os.path.exists(f"{d}/.eval_done") and not os.path.exists(f"{d}/.eval_fail")):
            out.append(d)
    return out


def partial_todo(min_new=60):
    """Configs still running whose preds.json has grown by >= min_new since the last partial evaluation."""
    out = []
    for d in sorted(glob.glob(f"{H}/runs/swe/*/*/")):
        d = d.rstrip("/")
        if d.split("/")[-2].startswith("_") or os.path.exists(f"{d}/.agent_done") or not os.path.exists(f"{d}/preds.json"):
            continue
        try:
            n = len(json.load(open(f"{d}/preds.json")))
        except Exception:
            continue
        last = int(open(f"{d}/.partial_n").read()) if os.path.exists(f"{d}/.partial_n") else 0
        if n - last >= min_new:
            out.append((d, n))
    return out


def start(d, workers, partial=False):
    """Same run_id for partial and final evaluations: the harness skips instances that already have a report, so
    partial runs simply evaluate finished instances early."""
    tag, cfg = d.split("/")[-2:]
    run_id = f"{tag}__{cfg}".replace("@", "_")
    preds = f"{d}/preds.json"
    if partial:
        snap = json.load(open(preds))
        json.dump(snap, open(f"{d}/preds_partial.json", "w"))
        open(f"{d}/.partial_n", "w").write(str(len(snap)))
        preds = f"{d}/preds_partial.json"
    cmd = [sys.executable, "-m", "lcs.swe_harness", "--dataset_name", f"{H}/data/swe/verified_v5.jsonl",
           "--predictions_path", preds, "--max_workers", str(workers), "--run_id", run_id,
           "--timeout", "1800", "--report_dir", d]
    lf = open(f"{d}/eval.log", "a")
    return subprocess.Popen(cmd, cwd=d, stdout=lf, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONPATH=f"{H}/code")), run_id


def finish(d, run_id, attempt):
    """Returns True when done; False -> rerun (the harness skips instances that already have a report, so a rerun only
    retries instances that errored, e.g. on transient docker failures under load)."""
    reps = glob.glob(f"{d}/*.{run_id}.json")
    errs = json.load(open(reps[0])).get("error_ids", []) if reps else None
    # no report (harness crashed): retry up to 5 times; instance errors (mostly patches that do not apply or test
    # timeouts, i.e. genuine failures) get a single retry in case some were transient docker failures
    if (not reps and attempt < 5) or (errs and attempt < 1):
        print("retry", d, "attempt", attempt + 1, "errors", None if errs is None else len(errs), flush=True)
        if reps:
            os.remove(reps[0])
        return False
    if not reps:
        print("no report for", d, flush=True)
        open(f"{d}/.eval_fail", "w").close()
        return True
    os.replace(reps[0], f"{d}/report.json")
    open(f"{d}/.eval_done", "w").close()
    r = json.load(open(f"{d}/report.json"))
    print("evaluated", d, "resolved", r.get("resolved_instances"), "/", r.get("total_instances"),
          "errors", len(r.get("error_ids", [])), flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    running, attempts, partial = {}, {}, set()
    while True:
        for d, (p, rid) in list(running.items()):
            if p.poll() is not None:
                del running[d]
                if d in partial:
                    partial.discard(d)
                    reps = glob.glob(f"{d}/*.{rid}.json")
                    if reps:
                        os.replace(reps[0], f"{d}/report_partial.json")
                        r = json.load(open(f"{d}/report_partial.json"))
                        print("partial", d, "resolved", r.get("resolved_instances"), "/", r.get("submitted_instances"),
                              flush=True)
                    continue
                if not finish(d, rid, attempts.get(d, 0)):
                    attempts[d] = attempts.get(d, 0) + 1
                    running[d] = start(d, a.workers)
        for d in todo():
            if d not in running and len(running) < a.parallel:
                running[d] = start(d, a.workers)
                print("start eval", d, flush=True)
        if not todo():
            for d, n in partial_todo():
                if d not in running and len(running) < a.parallel:
                    running[d] = start(d, a.workers, partial=True)
                    partial.add(d)
                    print("start partial eval", d, n, flush=True)
        if a.once and not running and not todo():
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
