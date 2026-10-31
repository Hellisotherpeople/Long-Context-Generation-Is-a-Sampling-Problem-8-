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
    else:
        raise SystemExit(a.wave)


if __name__ == "__main__":
    main()
