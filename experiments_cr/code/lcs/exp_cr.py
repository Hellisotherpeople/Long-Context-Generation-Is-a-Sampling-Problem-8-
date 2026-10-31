"""Camera-ready wave definitions (pre-registered in ~/lcg-cr/PREREG.md before any output is read).

python -m lcs.exp_cr r1|r2 [--prio N]
"""
from __future__ import annotations

import argparse

from . import models as MR
from .make_jobs import SAMPLERS_SLIM, core_job, enqueue, sampler_cfgs
from .prompts import fiction_prompts, nonfiction_prompts

PAPER_MODELS = ["llama3.2-1b", "qwen2.5-1.5b", "qwen2.5-3b", "mistral-7b-v0.3", "llama3.1-8b",
                "mistral-nemo-12b", "qwen2.5-14b", "qwen2.5-32b", "llama3.3-70b", "gpt-oss-120b"]
NONFIC_MODELS = ["llama3.2-1b", "qwen2.5-3b", "llama3.1-8b", "qwen3.5-9b", "gemma4-31b", "llama3.3-70b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wave")
    ap.add_argument("--prio", type=int, default=20)
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    if a.wave == "r1":
        cfgs = sampler_cfgs(SAMPLERS_SLIM, [1.0, 1.5, 2.0])
        for m in PAPER_MODELS:
            mt = 16000 if MR.get(m)["ctx"] >= 17024 else MR.get(m)["ctx"] - 1024
            job = core_job(m, "r1_16k", fiction_prompts(1), cfgs, mt, seeds=(1,))
            enqueue(f"r1_16k/{m}", job, MR.get(m)["tp"], a.prio, MR.get(m)["path"])
        for m in ["gpt-oss-120b"]:
            job = core_job(m, "r1_32k", fiction_prompts(1), cfgs, 32000, seeds=(1,))
            enqueue(f"r1_32k/{m}", job, MR.get(m)["tp"], a.prio + 5, MR.get(m)["path"])
    elif a.wave == "r2":
        cfgs = sampler_cfgs(SAMPLERS_SLIM, [1.0, 1.5])
        for m in NONFIC_MODELS:
            job = core_job(m, "r2_nonfic16k", nonfiction_prompts(), cfgs, 16000, seeds=(1,))
            enqueue(f"r2_nonfic16k/{m}", job, MR.get(m)["tp"], a.prio + 2, MR.get(m)["path"])
    elif a.wave == "r4":
        # natural-length multi-turn continuation ("Continue."), EOS allowed, 16K total
        cfgs = sampler_cfgs(SAMPLERS_SLIM, [1.0, 1.5, 2.0])
        for m in PAPER_MODELS:
            if m == "gpt-oss-120b":
                continue
            job = core_job(m, "r4_mt", fiction_prompts(1), cfgs, 16000, seeds=(1,))
            for r in job["requests"]:
                r["force_len"] = False
                r["multiturn"] = {"turn_tokens": 4096, "total": 16000, "continue_msg": "Continue.", "max_turns": 24}
            ml = min(MR.get(m)["ctx"], 16000 + 4096 + 2048)
            job["max_len"] = ml - 1024
            job["engine"]["max_model_len"] = ml
            enqueue(f"r4_mt/{m}", job, MR.get(m)["tp"], a.prio + 15, MR.get(m)["path"])
    elif a.wave == "r5":
        # hyperparameter x temperature map for the healthy region (fiction, forced 8K, 3 small models)
        HP = [("minp", "min_p", [0.02, 0.05, 0.1, 0.2, 0.3]), ("topp", "top_p", [0.8, 0.9, 0.95]),
              ("nsigma", "top_nsigma", [0.5, 1.0, 1.5, 2.0, 3.0]), ("toph", "top_h", [0.2, 0.4, 0.6, 0.8]),
              ("topk", "top_k", [10, 50, 200])]
        samplers = [(f"{n}{v}", [[op, v]]) for n, op, vs in HP for v in vs] + [("pless", [["p_less", 0.0]]), ("temp", [])]
        cfgs = sampler_cfgs(samplers, [0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0], greedy=False)
        for m in (a.models.split(",") if a.models else ["llama3.2-1b", "qwen2.5-3b", "llama3.1-8b", "qwen3.5-9b"]):
            job = core_job(m, "r5_hp", fiction_prompts(1), cfgs, 8000, seeds=(1,))
            enqueue(f"r5_hp/{m}", job, 1, a.prio + 12, MR.get(m)["path"])
    elif a.wave == "r8":
        # mechanism ablation: candidate set decided at T=1, sampling at t ("temperature last")
        base = [("topp0.9", [["top_p", 0.9]]), ("toph0.4", [["top_h", 0.4]]), ("minp0.1", [["min_p", 0.1]]),
                ("pless", [["p_less", 0.0]]), ("topk50", [["top_k", 50]])]
        cfgs = []
        for n, ops in base:
            for t in (1.5, 2.0, 3.0):
                cfgs.append((f"{n}_tl@T{t}", {"T": 1.0, "ops": ops + [["temp_last", t]]}, n + "_tl", t))
        for m in ["llama3.2-1b", "qwen2.5-3b", "llama3.1-8b", "qwen3.5-9b"]:
            job = core_job(m, "r8_tl", fiction_prompts(1), cfgs, 8000, seeds=(1,))
            enqueue(f"r8_tl/{m}", job, 1, a.prio, MR.get(m)["path"])
    elif a.wave == "r9":
        cfgs = sampler_cfgs(SAMPLERS_SLIM, [1.0, 1.5, 2.0])
        for m in ["llama3.2-1b", "qwen2.5-3b", "mistral-7b-v0.3", "llama3.1-8b"]:
            job = core_job(m, "r9_16k_s2", fiction_prompts(1), cfgs, 16000, seeds=(2,))
            enqueue(f"r9_16k_s2/{m}", job, 1, a.prio + 3, MR.get(m)["path"])
    elif a.wave == "r9b":
        cfgs = sampler_cfgs(SAMPLERS_SLIM, [1.0, 1.5, 2.0])
        for m in ["qwen2.5-1.5b", "mistral-nemo-12b", "qwen2.5-14b", "qwen2.5-32b", "llama3.3-70b", "gpt-oss-120b"]:
            job = core_job(m, "r9_16k_s2", fiction_prompts(1), cfgs, 16000, seeds=(2,))
            enqueue(f"r9_16k_s2/{m}", job, MR.get(m)["tp"], a.prio, MR.get(m)["path"])
    elif a.wave == "r10":
        # large / modern models on the R1 grid (fiction, forced 16K, seed 1)
        cfgs = sampler_cfgs(SAMPLERS_SLIM, [1.0, 1.5, 2.0])
        for m in ["qwen3.5-122b-a10b", "glm4.5-air", "qwen3.5-27b", "gemma4-31b", "olmo3.1-32b", "nemotron3-30b-a3b"]:
            job = core_job(m, "r1_16k", fiction_prompts(1), cfgs, 16000, seeds=(1,))
            enqueue(f"r1_16k/{m}", job, MR.get(m)["tp"], a.prio, MR.get(m)["path"])
    elif a.wave == "r11":
        # sampling stacks: truncation rule + DRY, and + top-2 floor (fiction, forced 16K)
        DRY = {"mult": 0.8, "base": 1.75, "allowed": 2}
        rules = [("topp0.9", [["top_p", 0.9]]), ("minp0.1", [["min_p", 0.1]]), ("nsigma1", [["top_nsigma", 1.0]]),
                 ("pless", [["p_less", 0.0]]), ("toph0.4", [["top_h", 0.4]])]
        cfgs = []
        for n, ops in rules:
            for T in (1.0, 1.5, 2.0):
                cfgs.append((f"dry+{n}@T{T}", {"T": T, "ops": ops, "dry": DRY}, f"dry+{n}", T))
                cfgs.append((f"k2+{n}@T{T}", {"T": T, "ops": ops + [["keep_top", 2]]}, f"k2+{n}", T))
        for T in (1.0,):
            cfgs.append((f"dry+temp@T{T}", {"T": T, "ops": [], "dry": DRY}, "dry+temp", T))
        for m in ["llama3.2-1b", "qwen2.5-3b", "mistral-7b-v0.3", "llama3.1-8b", "mistral-nemo-12b", "qwen2.5-14b", "llama3.3-70b", "gpt-oss-120b"]:
            job = core_job(m, "r11_stack", fiction_prompts(1), cfgs, 16000, seeds=(1,))
            enqueue(f"r11_stack/{m}", job, MR.get(m)["tp"], a.prio, MR.get(m)["path"])
    else:
        raise SystemExit(a.wave)


if __name__ == "__main__":
    main()
