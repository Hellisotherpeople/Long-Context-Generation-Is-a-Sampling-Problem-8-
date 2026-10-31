"""Long chain-of-thought reasoning, second wave: AIME 2025/2026 + HMMT Feb 2025/2026 (MathArena), stronger reasoning models,
and the absorption-free keep_top=2 variants.  Thinking mode, 32K budget, natural EOS.

python -m lcs.exp_reason2 --models qwen3.5-9b,nemotron3-30b-a3b --samples 2 --prio 050
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from .exp_reason import SUFFIX, THINK

H = os.path.expanduser("~/lcg-cr")
B = f"{H}/data/bench"
SAMPLERS = [
    ("temp", [], [0.6, 1.0]),
    ("topp0.95", [["top_p", 0.95]], [0.6, 1.0, 1.3]),
    ("minp0.1", [["min_p", 0.1]], [0.6, 1.0, 1.3]),
    ("nsigma1", [["top_nsigma", 1.0]], [0.6, 1.0, 1.3]),
    ("pless", [["p_less", 0.0]], [0.6, 1.0, 1.3]),
    ("toph0.4", [["top_h", 0.4]], [0.6, 1.0, 1.3]),
    ("pless+k2", [["p_less", 0.0], ["keep_top", 2]], [0.6, 1.0, 1.3]),
    ("nsigma1+k2", [["top_nsigma", 1.0], ["keep_top", 2]], [0.6, 1.0, 1.3]),
]


def problems():
    out = []
    for name, set_ in (("aime_2025", "aime"), ("aime_2026", "aime"), ("hmmt_feb_2025", "hmmt"), ("hmmt_feb_2026", "hmmt")):
        fs = glob.glob(f"{B}/MathArena__{name}/**/*.parquet", recursive=True)
        d = pd.concat([pd.read_parquet(f) for f in fs])
        for r in d.itertuples():
            out.append({"pid": f"{name}_{r.problem_idx}", "set": set_, "src": name, "problem": r.problem, "answer": str(r.answer)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--prio", default="050")
    a = ap.parse_args()
    probs = problems()
    for model in a.models.split(","):
        srcf = next((f for f in (f"{H}/jobs/reason/{model}.json", f"{H}/jobs/core32k/{model}.json") if os.path.exists(f)), None)
        src = json.load(open(srcf))
        fam = next((k for k in THINK if model.startswith(k)), None)
        ck = dict(THINK.get(fam, {}))
        reqs = []
        for pr in probs:
            msgs = [{"role": "user", "content": pr["problem"] + SUFFIX}]
            base = {"prompt": {"type": "chat", "messages": msgs, "chat_kwargs": ck}, "max_tokens": a.budget, "force_len": False}
            meta = {"pid": pr["pid"], "set": pr["set"], "src": pr["src"], "answer": pr["answer"]}
            reqs.append(dict(base, uid=f"{pr['pid']}__greedy__s0", lcs={"T": 1.0, "ops": [["greedy", 0.0]]},
                             meta=dict(meta, sampler="greedy", T=0.0, i=0)))
            for name, ops, temps in SAMPLERS:
                for T in temps:
                    for i in range(a.samples):
                        reqs.append(dict(base, uid=f"{pr['pid']}__{name}@T{T}__s{i}", lcs={"T": T, "ops": ops},
                                         meta=dict(meta, sampler=name, T=T, i=i)))
        eng = {"max_model_len": a.budget + 2048, "max_num_seqs": 512, "enable_prefix_caching": True}
        for k in ("language_model_only", "tokenizer_mode", "config_format", "load_format"):  # model-specific engine flags
            if k in src.get("engine", {}):
                eng[k] = src["engine"][k]
        job = {"job_id": f"reason2/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/reason2/{model}", "requests": reqs}
        os.makedirs(f"{H}/jobs/reason2", exist_ok=True)
        jf = f"{H}/jobs/reason2/{model}.json"
        json.dump(job, open(jf, "w"))
        env = {}
        for fn in os.listdir(f"{H}/queue/done"):
            if fn.endswith(f"_core32k_{model}.json") or fn.endswith(f"_reason_{model}.json"):
                env.update(json.load(open(f"{H}/queue/done/{fn}")).get("env", {}) or {})
        q = {"job": jf, "gpus": int(src.get("tp", 1)), "model_dir": src["model"], "kind": "gen"}
        if env:
            q["env"] = env
        json.dump(q, open(f"{H}/queue/pending/{a.prio}_reason2_{model}.json", "w"))
        print(model, "problems", len(probs), "requests", len(reqs), "chat_kwargs", ck)


if __name__ == "__main__":
    main()
