"""LongBench v2 stage 2 (answer extraction) and scoring.

python -m lcs.ana_lb2 stage2 --models qwen3.5-9b --prio 016   # queue greedy extraction jobs (official 0shot_cot_ans)
python -m lcs.ana_lb2 score                                    # -> ~/lcg-cr/results/lb2_{gen,cells}.parquet

Accuracy uses the official regex on the stage-2 answer; `acc_direct` also accepts an answer already stated in the CoT.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

H = os.path.expanduser("~/lcg-cr")
B = f"{H}/data/bench"


def extract(resp):
    resp = (resp or "").replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", resp) or re.search(r"The correct answer is ([A-D])", resp)
    return m.group(1) if m else None


def stage2(a):
    from .tok import load_tokenizer
    data = {d["_id"]: d for d in json.load(open(f"{B}/zai-org__LongBench-v2/data.json"))}
    tmpl = open(f"{B}/LongBench/prompts/0shot_cot_ans.txt").read()
    for model in a.models.split(","):
        run = f"{H}/runs/lb2/{model}"
        metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
        tok = load_tokenizer(metas[0]["model"])
        src = json.load(open(f"{H}/jobs/lb2/{model}.json"))
        ck = json.load(open(f"{H}/jobs/core32k/{model}.json"))["requests"][0]["prompt"].get("chat_kwargs") or {}
        reqs = []
        for m in metas:
            d = data[m["meta"]["pid"]]
            cot = tok.decode(np.load(f"{run}/gen/{m['uid']}.npz")["tokens"], skip_special_tokens=True)
            text = tmpl.replace("$Q$", d["question"].strip()).replace("$C_A$", d["choice_A"].strip()) \
                .replace("$C_B$", d["choice_B"].strip()).replace("$C_C$", d["choice_C"].strip()) \
                .replace("$C_D$", d["choice_D"].strip()).replace("$COT$", cot)
            reqs.append({"uid": m["uid"], "prompt": {"type": "chat", "messages": [{"role": "user", "content": text}],
                                                    "chat_kwargs": ck},
                         "max_tokens": 128, "force_len": False, "lcs": {"T": 1.0, "ops": [["greedy", 0.0]]},
                         "meta": m["meta"]})
        eng = {"max_model_len": 8192, "max_num_seqs": 512}
        for k in ("language_model_only", "tokenizer_mode", "config_format", "load_format"):  # model-specific flags
            if k in src.get("engine", {}):
                eng[k] = src["engine"][k]
        job = {"job_id": f"lb2_ans/{model}", "model": src["model"], "tp": src.get("tp", 1),
               "engine": eng, "out_dir": f"{H}/runs/lb2_ans/{model}",
               "requests": reqs}
        os.makedirs(f"{H}/jobs/lb2_ans", exist_ok=True)
        jf = f"{H}/jobs/lb2_ans/{model}.json"
        json.dump(job, open(jf, "w"))
        q = json.load(open(next(f for f in glob.glob(f"{H}/queue/*/*_lb2_{model}.json"))))
        q2 = {"job": jf, "gpus": q["gpus"], "model_dir": q["model_dir"], "kind": "gen"}
        if q.get("env"):
            q2["env"] = q["env"]
        json.dump(q2, open(f"{H}/queue/pending/{a.prio}_lb2ans_{model}.json", "w"))
        print(model, len(reqs))


def score(a):
    import pandas as pd
    from .tok import load_tokenizer
    rows = []
    for run in sorted(glob.glob(f"{H}/runs/lb2/*")):
        model = run.split("/")[-1]
        ans_run = f"{H}/runs/lb2_ans/{model}"
        if not os.path.exists(f"{run}/meta.jsonl"):
            continue
        metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
        n_req = len(json.load(open(f"{H}/jobs/lb2/{model}.json"))["requests"])
        n_ans = sum(1 for _ in open(f"{ans_run}/meta.jsonl")) if os.path.exists(f"{ans_run}/meta.jsonl") else 0
        if len(metas) < n_req or n_ans < len(metas):  # only models whose stage 1 and stage 2 are complete
            print("skip", model, len(metas), n_req, n_ans)
            continue
        tok = load_tokenizer(metas[0]["model"])
        ans = {}
        if os.path.exists(f"{ans_run}/meta.jsonl"):
            for l in open(f"{ans_run}/meta.jsonl"):
                m = json.loads(l)
                ans[m["uid"]] = tok.decode(np.load(f"{ans_run}/gen/{m['uid']}.npz")["tokens"], skip_special_tokens=True)
        for m in metas:
            mm = m["meta"]
            cot = tok.decode(np.load(f"{run}/gen/{m['uid']}.npz")["tokens"], skip_special_tokens=True)
            p2 = extract(ans.get(m["uid"]))
            p1 = extract(cot)
            rows.append({"model": model, "pid": mm["pid"], "difficulty": mm["difficulty"], "length": mm["length"],
                         "sampler": mm["sampler"], "T": mm["T"], "n_ctx": mm.get("n_ctx"), "cot_tokens": m["n_tokens"],
                         "cot_truncated": m["n_tokens"] >= 1024, "pred": p2, "pred_direct": p1 or p2,
                         "correct": p2 == mm["answer"], "acc_direct": (p1 or p2) == mm["answer"]})
    df = pd.DataFrame(rows)
    os.makedirs(f"{H}/results", exist_ok=True)
    df.to_parquet(f"{H}/results/lb2_gen.parquet")
    cells = df.groupby(["model", "sampler", "T"]).agg(n=("pid", "size"), acc=("correct", "mean"),
                                                      acc_direct=("acc_direct", "mean"),
                                                      truncated=("cot_truncated", "mean")).reset_index()
    cells.to_parquet(f"{H}/results/lb2_cells.parquet")
    pd.set_option("display.width", 220, "display.max_rows", 500)
    for m, g in cells.groupby("model"):
        print("==", m)
        print(g.pivot_table(index="sampler", columns="T", values="acc").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--models", default="")
    ap.add_argument("--prio", default="016")
    a = ap.parse_args()
    stage2(a) if a.cmd == "stage2" else score(a)


if __name__ == "__main__":
    main()
