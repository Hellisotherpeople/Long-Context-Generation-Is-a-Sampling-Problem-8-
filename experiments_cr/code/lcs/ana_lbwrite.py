"""Score LongBench-Write runs: official length score S_l, judged quality S_q (6 dims, (x-1)*25), final = (S_l+S_q)/2.

python -m lcs.ana_lbwrite [--judge gptoss] -> ~/lcg-cr/results/lbwrite_{gen,cells}.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

H = os.path.expanduser("~/lcg-cr")
DIMS = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]


def count_words(text):  # official (LongWriter/evaluation/pred.py)
    return len(re.findall(r"[一-鿿]", text)) + len(re.findall(r"\b[a-zA-Z]+\b", text))


def s_len(x, y):  # official (eval_length.py): x required, y produced
    if y == 0:
        return 0.0
    return 100 * max(0, 1. - (y / x - 1) / 3) if y > x else 100 * max(0, 1. - (x / y - 1) / 2)


def main():
    from .tok import load_tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gptoss")
    a = ap.parse_args()
    rows = []
    for run in sorted(glob.glob(f"{H}/runs/lbwrite/*")):
        if not os.path.exists(f"{run}/meta.jsonl"):
            continue
        model = run.split("/")[-1]
        metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
        tok = load_tokenizer(metas[0]["model"])
        J = {}
        jf = f"{run}/judge_{a.judge}/lbw.jsonl"
        if os.path.exists(jf):
            for l in open(jf):
                r = json.loads(l)
                sc = r.get("scores") or {}
                vals = [float(sc[d]) for d in DIMS if isinstance(sc.get(d), (int, float))]
                J[r["uid"]] = (np.mean(vals) - 1) * 25 if len(vals) == len(DIMS) else np.nan
        for m in metas:
            mm = m["meta"]
            text = tok.decode(np.load(f"{run}/gen/{m['uid']}.npz")["tokens"], skip_special_tokens=True)
            n = count_words(text)
            sl = s_len(mm["length"], n)
            sq = J.get(m["uid"], np.nan)
            rows.append({"model": model, "pid": mm["pid"], "required": mm["length"], "words": n, "sampler": mm["sampler"],
                         "T": mm["T"], "S_l": sl, "S_q": sq, "S": (sl + sq) / 2 if not np.isnan(sq) else np.nan,
                         "finish": m.get("finish"), "n_tokens": m["n_tokens"]})
    df = pd.DataFrame(rows)
    df.to_parquet(f"{H}/results/lbwrite_gen.parquet")
    cells = df.groupby(["model", "sampler", "T"])[["S_l", "S_q", "S"]].mean().reset_index()
    cells.to_parquet(f"{H}/results/lbwrite_cells.parquet")
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(df.groupby(["sampler", "T"])[["S_l", "S_q", "S"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
