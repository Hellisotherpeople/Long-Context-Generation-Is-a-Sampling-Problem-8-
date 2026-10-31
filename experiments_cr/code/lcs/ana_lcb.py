"""Score LiveCodeBench runs with the official lcb_runner tests.

Untrusted generated code runs in a docker sandbox (image lcs-lcb = python:3.12-slim + numpy + tqdm; no network,
read-only code/data mounts, memory and pid limits, host uid).  The answer is the part after the end of reasoning (</think> or the gpt-oss final
channel); a generation that never leaves its reasoning has no answer (fail).  Code = last fenced block (lcb_runner
extract_code).
python -m lcs.ana_lcb   -> <run>/lcb_eval.jsonl (cached) + ~/lcg-cr/results/lcb_{gen,cells}.parquet
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

H = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ~/lcg-cr (also inside the sandbox)
LCB = f"{H}/data/bench/LiveCodeBench"
ENDS = ("</think>", "<|channel|>final<|message|>")
SPECIAL = re.compile(r"<\|[^|>]{1,40}\|>|<｜[^｜]{1,40}｜>")


def answer_part(text):
    cut = max(text.rfind(e) + len(e) if text.rfind(e) >= 0 else -1 for e in ENDS)
    if cut < 0:
        return None
    return SPECIAL.sub("", text[cut:])


def extract_code(out):
    lines = out.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    return "\n".join(lines[idx[-2] + 1: idx[-1]]) if len(idx) >= 2 else ""


def _tests(q):
    import base64
    import pickle
    import zlib
    pub = json.loads(q["public_test_cases"])
    try:
        priv = json.loads(q["private_test_cases"])
    except Exception:
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(q["private_test_cases"].encode("utf-8")))))
    t = pub + priv
    return {"input_output": json.dumps({"inputs": [x["input"] for x in t], "outputs": [x["output"] for x in t],
                                        "fn_name": json.loads(q["metadata"]).get("func_name", None)})}


def _eval(run, release="test6"):
    """Inside the sandbox container (stdlib + numpy + tqdm): run every generation against the official tests."""
    import types
    for name, path in (("lcb_runner", f"{LCB}/lcb_runner"), ("lcb_runner.evaluation", f"{LCB}/lcb_runner/evaluation")):
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod  # skip the package __init__ files (they import datasets etc.)
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations
    probs = {q["question_id"]: q for q in (json.loads(l) for l in open(f"{H}/data/bench/livecodebench__code_generation_lite/{release}.jsonl"))}
    codes = json.load(open(f"{run}/lcb_codes.json"))  # uid -> (pid, code)
    by_pid = {}
    for uid, (pid, code) in codes.items():
        by_pid.setdefault(pid, []).append((uid, code))
    flat = [(uid, pid, code) for pid in sorted(by_pid) for uid, code in by_pid[pid]]
    tests = {pid: _tests(probs[pid]) for pid in by_pid}
    res, _ = evaluate_generations([tests[pid] for _, pid, _ in flat], [[code] for _, _, code in flat], debug=False,
                                  num_process_evaluate=int(os.environ.get("NPROC", 96)), timeout=6)
    with open(f"{run}/lcb_eval.jsonl", "w") as f:
        for k, (uid, pid, code) in enumerate(flat):
            ok = bool(code) and bool(np.all(np.asarray(res[k][0]) > 0))
            f.write(json.dumps({"uid": uid, "pass": ok}) + "\n")


SANDBOX_IMAGE = "lcs-lcb:latest"


def sandbox_cmd(run):
    """docker: no network, read-only code/data mounts, 128 GB memory, pid limit, host uid."""
    ro = [f"{H}/code", LCB, f"{H}/data/bench/livecodebench__code_generation_lite"]
    cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "128g", "--pids-limit", "8192",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "PYTHONPATH=" + f"{H}/code", "-e", "OMP_NUM_THREADS=1",
           "-e", "HOME=/tmp", "--tmpfs", "/tmp:exec", "-w", "/tmp"]
    for d in ro:
        cmd += ["-v", f"{d}:{d}:ro"]
    cmd += ["-v", f"{run}:{run}:rw", SANDBOX_IMAGE, "python", "-m", "lcs.ana_lcb", "_eval", run]
    return cmd


def evaluate(run):
    from .tok import load_tokenizer
    metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
    tok = load_tokenizer(metas[0]["model"])
    codes = {}
    for m in metas:
        text = tok.decode(np.load(f"{run}/gen/{m['uid']}.npz")["tokens"], skip_special_tokens=False)
        ans = answer_part(text)
        codes[m["uid"]] = (m["meta"]["pid"], extract_code(ans) if ans is not None else "")
    json.dump(codes, open(f"{run}/lcb_codes.json", "w"))
    subprocess.run(sandbox_cmd(run), check=True)


def main():
    import pandas as pd
    rows = []
    for run in sorted(glob.glob(f"{H}/runs/lcb/*")):
        if not os.path.exists(f"{run}/meta.jsonl"):
            continue
        metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
        ev = f"{run}/lcb_eval.jsonl"
        if not os.path.exists(ev) or sum(1 for _ in open(ev)) < len(metas):
            evaluate(run)
        ok = {json.loads(l)["uid"]: json.loads(l)["pass"] for l in open(ev)}
        mt = {}
        if os.path.exists(f"{run}/metrics.jsonl"):
            for l in open(f"{run}/metrics.jsonl"):
                r = json.loads(l)
                mt[r["uid"]] = r
        codes = json.load(open(f"{run}/lcb_codes.json"))
        for m in metas:
            mm, r = m["meta"], mt.get(m["uid"], {})
            lo = r.get("loop_onset_win")
            rows.append({"model": run.split("/")[-1], "pid": mm["pid"], "difficulty": mm["difficulty"],
                         "sampler": mm["sampler"], "T": mm["T"], "i": mm["i"], "n_tokens": m["n_tokens"],
                         "exhausted": m.get("finish") == "length", "answered": bool(codes[m["uid"]][1]),
                         "looped": lo is not None and not (isinstance(lo, float) and np.isnan(lo)),
                         "pass": ok.get(m["uid"], False)})
    df = pd.DataFrame(rows)
    os.makedirs(f"{H}/results", exist_ok=True)
    df.to_parquet(f"{H}/results/lcb_gen.parquet")
    cells = df.groupby(["model", "sampler", "T"]).agg(n=("pid", "size"), acc=("pass", "mean"),
                                                      answered=("answered", "mean"), exhausted=("exhausted", "mean"),
                                                      looped=("looped", "mean"), len=("n_tokens", "mean")).reset_index()
    cells.to_parquet(f"{H}/results/lcb_cells.parquet")
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(cells.round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "_eval":
        _eval(sys.argv[2])
    else:
        main()
