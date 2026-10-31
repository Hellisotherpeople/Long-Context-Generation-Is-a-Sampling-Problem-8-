"""LongProc (Ye et al., 2025): long procedural OUTPUTS (up to ~8K tokens) with deterministic, rule-based scoring.

Tasks: html_to_tsv_8k, pseudo_to_code_2k, path_traversal_8k, tom_tracking_8k, countdown_8k, travel_planning_8k.  Prompts and
evaluation functions come from the official repo (princeton-pli/LongProc, longproc.longproc_data).  Every sampler config
gets the same items; scoring is done by lcs.ana_longproc with the official evaluators.

python -m lcs.exp_longproc --models qwen3.5-4b,qwen3.5-9b --per-task 60 --prio 047
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

H = os.path.expanduser("~/lcg-cr")
REPO = f"{H}/data/bench/LongProc"
TASKS = ["html_to_tsv_8k", "pseudo_to_code_2k", "path_traversal_8k", "tom_tracking_8k", "countdown_8k", "travel_planning_8k"]
MAXTOK = {"0.5k": 1024, "2k": 3072, "8k": 10240}


def load_task(name):
    sys.path.insert(0, REPO)
    from longproc.longproc_data import load_longproc_data
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        data, _ = load_longproc_data(name, "./data")
    finally:
        os.chdir(cwd)
    return data


def main():
    from .exp_lb2 import CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--per-task", type=int, default=60)
    ap.add_argument("--prio", default="047")
    a = ap.parse_args()
    rng = random.Random(0)
    items = []
    for t in TASKS:
        data = load_task(t)
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for i in sorted(idx[:a.per_task]):
            items.append((t, i, data[i]["input_prompt"]))
    for model in a.models.split(","):
        src = json.load(open(f"{H}/jobs/core32k/{model}.json"))
        ck = src["requests"][0]["prompt"].get("chat_kwargs") or {}
        env = {}
        for fn in os.listdir(f"{H}/queue/done"):
            if fn.endswith(f"_core32k_{model}.json"):
                env.update(json.load(open(f"{H}/queue/done/{fn}")).get("env", {}) or {})
        # html_to_tsv has inputs up to 83K tokens; the other tasks fit in 24K -> two jobs (same run dir) so the short
        # tasks are admitted at full concurrency
        for part, tasks, max_len in (("main", [t for t in TASKS if t != "html_to_tsv_8k"], 24576),
                                     ("html", ["html_to_tsv_8k"], 98304)):
            reqs = []
            for t, i, prompt in items:
                if t not in tasks:
                    continue
                p = {"type": "chat", "messages": [{"role": "user", "content": prompt}], "chat_kwargs": ck}
                for name, T, ops in CONFIGS:
                    reqs.append({"uid": f"{t}-{i}__{name}__s0", "prompt": p, "max_tokens": MAXTOK[t.rsplit('_', 1)[1]],
                                 "force_len": False, "lcs": {"T": T, "ops": ops},
                                 "meta": {"pid": f"{t}-{i}", "task": t, "item": i, "sampler": name.split("@")[0],
                                          "T": 0.0 if name == "greedy" else T}})
            eng = {"max_model_len": max_len + 1024, "max_num_seqs": 512, "enable_prefix_caching": True}
            job = {"job_id": f"longproc/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
                   "out_dir": f"{H}/runs/longproc/{model}", "requests": reqs, "max_len": max_len}
            os.makedirs(f"{H}/jobs/longproc", exist_ok=True)
            jf = f"{H}/jobs/longproc/{model}__{part}.json"
            json.dump(job, open(jf, "w"))
            q = {"job": jf, "gpus": int(src.get("tp", 1)), "model_dir": src["model"], "kind": "gen"}
            e2 = dict(env)
            if max_len > 32768:
                e2["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            if e2:
                q["env"] = e2
            json.dump(q, open(f"{H}/queue/pending/{a.prio}_longproc_{model}__{part}.json", "w"))
            print(model, part, "requests", len(reqs))


if __name__ == "__main__":
    main()
