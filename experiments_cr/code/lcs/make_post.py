"""Queue post-processing jobs (evaluator perplexity, LLM judge) for experiment runs.

~/lcg-cr/ana/bin/python -m lcs.make_post evalppl --exps core32k,book32k --prio 7
~/lcg-cr/ana/bin/python -m lcs.make_post judge --exps core32k --tasks window,global --prio 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HOME = os.path.expanduser("~/lcg-cr")
EVALUATOR = f"{HOME}/models/Qwen__Qwen3.5-35B-A3B-Base"
JUDGE = f"{HOME}/models/openai__gpt-oss-120b"


def runs_for(exps):
    out = {}
    for exp in exps:
        for run in sorted(glob.glob(f"{HOME}/runs/{exp}/*")):
            mf = f"{run}/meta.jsonl"
            if os.path.exists(mf):
                with open(mf) as f:
                    line = f.readline().strip()
                if line:  # runs that have just started have an empty meta file
                    out[run] = json.loads(line)["model"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("--exps", required=True)
    ap.add_argument("--prio", type=int, default=30)
    ap.add_argument("--tasks", default="window,global")
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", type=int, default=1, help="split runs over N jobs (GPUs)")
    ap.add_argument("--models", default=None, help="restrict to these model dirs")
    ap.add_argument("--judge-model", default=JUDGE, help="judge model dir (default gpt-oss-120b)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chat-kwargs", default='{"reasoning_effort": "low"}')
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()
    runs = runs_for(a.exps.split(","))
    if a.models:
        keep = set(a.models.split(","))
        runs = {r: t for r, t in runs.items() if r.split("/")[-1] in keep}
    items = sorted(runs.items())
    chunks = [dict(items[i::a.split]) for i in range(a.split)]
    os.makedirs(f"{HOME}/jobs/post", exist_ok=True)
    for ci, ch in enumerate(chunks):
        if not ch:
            continue
        tag = f"{a.kind}{('-' + a.name) if a.name else ''}_{a.exps.replace(',', '+')}_{a.models.replace(',', '+') if a.models else 'all'}_{ci}"
        if a.kind == "evalppl":
            spec = {"evaluator": EVALUATOR, "name": a.name or "q35a3b", "tp": 1, "runs": ch, "max_model_len": 40960}
            md = EVALUATOR
        elif a.kind == "judge":
            spec = {"judge": a.judge_model, "judge_name": a.name or "gptoss", "tp": a.tp, "runs": list(ch), "tok": ch,
                    "tasks": a.tasks.split(","), "chat_kwargs": json.loads(a.chat_kwargs), "max_model_len": a.max_model_len}
            md = a.judge_model
        else:
            raise SystemExit(a.kind)
        sp = f"{HOME}/jobs/post/{tag}.json"
        json.dump(spec, open(sp, "w"), indent=1)
        q = {"job": sp, "gpus": a.tp if a.kind == "judge" else 1, "model_dir": md, "kind": a.kind, "args": [sp]}
        json.dump(q, open(f"{HOME}/queue/pending/{a.prio:03d}_{tag}.json", "w"))
        print(f"queued {tag}: {len(ch)} runs")


if __name__ == "__main__":
    main()
