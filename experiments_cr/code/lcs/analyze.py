"""Aggregate metrics.jsonl files into tidy tables (run with the analysis venv).

python -m lcs.analyze core32k            -> ~/lcg-cr/results/core32k_{gen,cells}.parquet + summary print
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RUNS = os.path.expanduser("~/lcg-cr/runs")
OUT = os.path.expanduser("~/lcg-cr/results")

FAMILY = {
    "temp": "none", "topk50": "fixed", "topp0.9": "partial", "topp0.95": "partial",
    "minp0.05": "partial", "minp0.1": "partial", "minp0.2": "partial", "typical0.9": "partial",
    "eta0.0009": "partial", "eps0.0003": "fixed", "nsigma1": "full", "nsigma1.5": "full", "nsigma0.5": "full",
    "pless": "full", "pless_norm": "full", "toph0.4": "full", "toph0.6": "full", "greedy": "greedy",
}


def load_exp(exp):
    rows = []
    for f in sorted(glob.glob(f"{RUNS}/{exp}/*/metrics.jsonl")):
        model = f.split("/")[-2]
        for line in open(f):
            r = json.loads(line)
            r["model"] = model
            win = r.pop("win")
            r["win"] = win
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["family"] = df["sampler"].map(FAMILY).fillna("other")
        df["looped"] = df["loop_onset_win"].notna()
    return df


def cell_table(df):
    keys = ["model", "sampler", "T"]
    agg = {
        "n": ("uid", "count"),
        "loop_rate": ("looped", "mean"),
        "Q4_rep32": ("Q4_rep32", "mean"), "Q1_rep32": ("Q1_rep32", "mean"),
        "Q4_rep10": ("Q4_rep10", "mean"),
        "Q4_d2": ("Q4_d2", "mean"), "Q1_d2": ("Q1_d2", "mean"),
        "Q4_zlib": ("Q4_zlib", "mean"),
        "Q4_argmax": ("Q4_argmax_frac", "mean"), "Q1_argmax": ("Q1_argmax_frac", "mean"),
        "Q4_logrank": ("Q4_logrank_mean", "mean"), "Q1_logrank": ("Q1_logrank_mean", "mean"),
        "Q4_H1": ("Q4_H1", "mean"), "Q1_H1": ("Q1_H1", "mean"),
        "Q4_Hq": ("Q4_Hq", "mean"), "Q1_Hq": ("Q1_Hq", "mean"),
        "Q4_surpr": ("Q4_surpr", "mean"), "Q1_surpr": ("Q1_surpr", "mean"),
        "d2_all": ("d2_all", "mean"),
    }
    return df.groupby(keys).agg(**agg).reset_index()


def main(exp):
    os.makedirs(OUT, exist_ok=True)
    df = load_exp(exp)
    print(f"{exp}: {len(df)} generations, models: {sorted(df.model.unique()) if len(df) else []}")
    if not len(df):
        return
    bad = df[df.lq_outside > 0]
    if len(bad):
        print(f"WARNING {len(bad)} generations with tokens outside the candidate set")
    cells = cell_table(df)
    flat = df.drop(columns=["win"]).copy()
    for c in flat.columns:
        if flat[c].dtype == object:
            flat[c] = flat[c].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    flat.to_parquet(f"{OUT}/{exp}_gen.parquet")
    df[["uid", "model", "sampler", "T", "win"]].to_pickle(f"{OUT}/{exp}_win.pkl")
    cells.to_parquet(f"{OUT}/{exp}_cells.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    show = ["model", "sampler", "T", "n", "loop_rate", "Q1_rep32", "Q4_rep32", "Q4_d2", "Q4_zlib", "Q1_argmax",
            "Q4_argmax", "Q1_H1", "Q4_H1", "Q4_Hq", "Q4_surpr"]
    print(cells[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
