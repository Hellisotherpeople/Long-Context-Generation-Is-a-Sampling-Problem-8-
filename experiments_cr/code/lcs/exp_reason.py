"""Long chain-of-thought reasoning under different samplers.

Problems: AIME 2025 + AIME 2026 (60, integer answers) and MATH-500 level 5 (134).
Thinking-mode models, budget up to 32K generated tokens (natural EOS; no forced length).
Metrics (analysis): accuracy, budget-exhaustion rate, terminal-loop rate, length.

~/lcg-cr/ana/bin/python -m lcs.exp_reason --models qwen3.5-4b --prio 35
"""
from __future__ import annotations

import argparse
import os

from . import models as MR
from .make_jobs import RUNS, enqueue

DATA = os.path.expanduser("~/lcg-cr/data/math")
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

THINK = {  # chat kwargs that turn reasoning ON
    "qwen3.5": {"enable_thinking": True}, "qwen3.6": {"enable_thinking": True}, "qwen3.8": {"enable_thinking": True},
    "nemotron3": {"enable_thinking": True}, "gemma4": {"enable_thinking": True},
    "gpt-oss": {"reasoning_effort": "high"},
}

R_SAMPLERS = [
    ("vendor", [["top_k", 20], ["top_p", 0.95]], [0.6, 1.0]),
    ("temp", [], [0.6, 1.0]),
    ("topk50", [["top_k", 50]], [1.0]),
    ("topp0.95", [["top_p", 0.95]], [0.6, 1.0, 1.3]),
    ("minp0.1", [["min_p", 0.1]], [0.6, 1.0, 1.3, 1.6]),
    ("nsigma1", [["top_nsigma", 1.0]], [0.6, 1.0, 1.3, 1.6]),
    ("pless", [["p_less", 0.0]], [0.6, 1.0, 1.3, 1.6]),
    ("toph0.4", [["top_h", 0.4]], [0.6, 1.0, 1.3, 1.6]),
]


def problems(sets=("aime25", "aime26", "math5")):
    import pandas as pd
    out = []
    if "aime25" in sets:
        d = pd.read_parquet(f"{DATA}/MathArena_aime_2025.parquet")
        out += [{"pid": f"aime25_{r.problem_idx}", "set": "aime", "problem": r.problem, "answer": str(r.answer)} for r in d.itertuples()]
    if "aime26" in sets:
        d = pd.read_parquet(f"{DATA}/MathArena_aime_2026.parquet")
        out += [{"pid": f"aime26_{r.problem_idx}", "set": "aime", "problem": r.problem, "answer": str(r.answer)} for r in d.itertuples()]
    if "math5" in sets:
        d = pd.read_json(f"{DATA}/math500.jsonl", lines=True)
        d = d[d.level == 5]
        out += [{"pid": f"math5_{i}", "set": "math5", "problem": r.problem, "answer": str(r.answer)} for i, r in enumerate(d.itertuples())]
    return out


def reason_job(model, n_samples=4, budget=32768, sets=("aime25", "aime26", "math5")):
    m = MR.get(model)
    ck = dict(m["chat_kwargs"])
    ck.update(THINK.get(m["family"], {}))
    reqs = []
    for pr in problems(sets):
        for sname, ops, temps in R_SAMPLERS:
            for T in temps:
                for i in range(n_samples):
                    uid = f"{pr['pid']}__{sname}@T{T}__n{i}"
                    reqs.append({"uid": uid, "prompt": {"type": "chat", "messages": [
                        {"role": "user", "content": pr["problem"] + SUFFIX}], "chat_kwargs": ck},
                        "max_tokens": budget, "force_len": False, "lcs": {"T": T, "ops": ops},
                        "meta": {"pid": pr["pid"], "set": pr["set"], "answer": pr["answer"],
                                 "sampler": sname, "T": T, "i": i}})
        reqs.append({"uid": f"{pr['pid']}__greedy__n0", "prompt": {"type": "chat", "messages": [
            {"role": "user", "content": pr["problem"] + SUFFIX}], "chat_kwargs": ck},
            "max_tokens": budget, "force_len": False, "lcs": {"T": 1.0, "ops": [["greedy", 0]]},
            "meta": {"pid": pr["pid"], "set": pr["set"], "answer": pr["answer"], "sampler": "greedy", "T": 0.0, "i": 0}})
    return {"job_id": f"reason/{model}", "model": m["path"], "tp": m["tp"],
            "engine": {"max_model_len": budget + 2048, "max_num_seqs": 512},
            "out_dir": f"{RUNS}/reason/{model}", "requests": reqs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prio", type=int, default=35)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    for model in a.models.split(","):
        m = MR.get(model)
        job = reason_job(model, a.n)
        enqueue(f"reason/{model}", job, m["tp"], a.prio, m["path"])


if __name__ == "__main__":
    main()
