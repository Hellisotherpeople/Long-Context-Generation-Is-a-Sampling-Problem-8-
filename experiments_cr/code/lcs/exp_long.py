"""Very long generation (128K .. 1M tokens) in a single context, saved in segments.

~/lcg-cr/ana/bin/python -m lcs.exp_long --models nemotron3-30b-a3b --total 1000000 --prio 40
"""
from __future__ import annotations

import argparse

from . import models as MR
from .exp_niah import YARN_QWEN35
from .make_jobs import RUNS, enqueue
from .prompts import chat_prompt, fiction_prompts

DRY = {"mult": 0.8, "base": 1.75, "allowed": 2}
LONG_CFGS = [
    ("topp0.9@T1.0", {"T": 1.0, "ops": [["top_p", 0.9]]}),
    ("minp0.1@T1.0", {"T": 1.0, "ops": [["min_p", 0.1]]}),
    ("minp0.1@T1.6", {"T": 1.6, "ops": [["min_p", 0.1]]}),
    ("nsigma1@T1.0", {"T": 1.0, "ops": [["top_nsigma", 1.0]]}),
    ("pless@T1.0", {"T": 1.0, "ops": [["p_less", 0.0]]}),
    ("toph0.4@T1.6", {"T": 1.6, "ops": [["top_h", 0.4]]}),
    ("nsigma1+k2@T1.3", {"T": 1.3, "ops": [["top_nsigma", 1.0], ["keep_top", 2]]}),
    ("dry+topp0.9@T1.0", {"T": 1.0, "ops": [["top_p", 0.9]], "dry": DRY}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--total", type=int, default=1000000)
    ap.add_argument("--seg", type=int, default=131072)
    ap.add_argument("--n_prompts", type=int, default=2)
    ap.add_argument("--cfgs", default="all")
    ap.add_argument("--prio", type=int, default=40)
    ap.add_argument("--yarn", action="store_true")
    ap.add_argument("--tp", type=int, default=None)
    a = ap.parse_args()
    prompts = fiction_prompts(1)[: a.n_prompts]
    cfgs = LONG_CFGS if a.cfgs == "all" else [c for c in LONG_CFGS if c[0] in a.cfgs.split(",")]
    n_seg = (a.total + a.seg - 1) // a.seg
    for model in a.models.split(","):
        m = MR.get(model)
        reqs = []
        for pr in prompts:
            for cname, lcs in cfgs:
                if "dry" in lcs and a.total > 65536:
                    continue  # DRY buffer is capped at 70K tokens of context
                reqs.append({"uid": f"{pr['pid']}__{cname}", "prompt": chat_prompt(pr["text"], m["chat_kwargs"]),
                             "max_tokens": a.seg, "force_len": True, "lcs": lcs,
                             "chain": {"seg_tokens": a.seg, "n_seg": n_seg},
                             "meta": {"pid": pr["pid"], "sampler": cname.split("@")[0],
                                      "T": lcs["T"], "total": a.total}})
        eng = {"max_model_len": a.total + 2048, "max_num_seqs": 64, "enable_prefix_caching": True,
               "max_num_batched_tokens": 16384}
        if a.yarn and m["family"] in ("qwen3.5", "qwen3.6", "qwen3.8"):
            eng["hf_overrides"] = YARN_QWEN35
        tag = f"long{a.total // 1000}k"
        tp = a.tp or m["tp"]
        job = {"job_id": f"{tag}/{model}", "model": m["path"], "tp": tp, "engine": eng,
               "out_dir": f"{RUNS}/{tag}/{model}", "requests": reqs, "max_len": a.total + 1024}
        enqueue(f"{tag}/{model}", job, tp, a.prio, m["path"])


if __name__ == "__main__":
    main()
