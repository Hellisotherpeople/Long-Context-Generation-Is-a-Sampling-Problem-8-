"""Analyse long-input retrieval (NIAH) under stochastic decoding.

python -m lcs.ana_niah   -> ~/lcg-cr/results/niah_{gen,cells}.parquet + printed accuracy table
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

RUNS = os.path.expanduser("~/lcg-cr/runs")
OUT = os.path.expanduser("~/lcg-cr/results")


def first_stats(run, uid):
    """Entropy / pmax / kept at the first answer token."""
    parts = sorted(glob.glob(f"{run}/stats/{uid}.p*.npz"))
    for p in parts:
        d = np.load(p)
        i = np.where(d["pos"] == 0)[0]
        if len(i):
            s = d["stats"][i[-1]]
            return float(s[0]), float(s[2]), float(s[3])
    return np.nan, np.nan, np.nan


def run_rows(run):
    from .tok import load_tokenizer
    metas = [json.loads(l) for l in open(f"{run}/meta.jsonl")]
    tok = load_tokenizer(metas[0]["model"])
    statf = {}
    for f in os.listdir(f"{run}/stats"):
        statf.setdefault(f.split(".p")[0], []).append(f)
    rows = []
    for m in metas:
        mm = m["meta"]
        toks = np.load(f"{run}/gen/{m['uid']}.npz")["tokens"]
        text = tok.decode(toks, skip_special_tokens=True)
        nums = re.findall(r"\d{7}", text.replace(",", ""))
        H1 = pmax = kept = np.nan
        for p in sorted(statf.get(m["uid"], [])):
            d = np.load(f"{run}/stats/{p}")
            i = np.where(d["pos"] == 0)[0]
            if len(i):
                s_ = d["stats"][i[-1]]
                H1, pmax, kept = float(s_[0]), float(s_[2]), float(s_[3])
                break
        rows.append({"model": run.split("/")[-1], "rope": run.split("/")[-2].split("-")[1], "L": mm["L"],
                     "inst": mm["inst"], "key": mm["key"], "depth": mm["depth"], "sampler": mm["sampler"],
                     "T": mm["T"], "i": mm["i"], "correct": bool(nums and nums[0] == mm["val"]),
                     "any_number": bool(nums), "H1_first": H1, "pmax_first": pmax, "kept_first": kept})
    return rows


def main():
    os.makedirs(f"{OUT}/niah_cache", exist_ok=True)
    frames = []
    for run in sorted(glob.glob(f"{RUNS}/niah-*/*")):
        mf = f"{run}/meta.jsonl"
        if not os.path.exists(mf):
            continue
        n = sum(1 for _ in open(mf))
        cache = f"{OUT}/niah_cache/{run.split('/')[-2]}__{run.split('/')[-1]}.parquet"
        if os.path.exists(cache):
            c = pd.read_parquet(cache)
            if len(c) == n:
                frames.append(c)
                continue
        c = pd.DataFrame(run_rows(run))
        c.to_parquet(cache)
        frames.append(c)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(f"{OUT}/niah_gen.parquet")
    cells = df.groupby(["model", "rope", "L", "sampler", "T"]).agg(
        n=("correct", "size"), acc=("correct", "mean"), H1=("H1_first", "mean"), pmax=("pmax_first", "mean")).reset_index()
    cells.to_parquet(f"{OUT}/niah_cells.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 3000)
    for model, g in cells.groupby("model"):
        print(f"== {model}")
        print(g.pivot_table(index=["sampler", "T"], columns="L", values="acc").round(3).to_string())
        print(g.groupby("L")[["H1", "pmax"]].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
