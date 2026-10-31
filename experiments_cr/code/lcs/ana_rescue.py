"""Analyse loop-rescue interventions (runs/rescue/*, runs/rescue_mt/*).

escaped   : the final 2K tokens of the continuation are not repetitive (mean 32-gram repetition < 0.5)
recovered : escaped AND the judge rates the final 1.5K tokens at least as coherent as the 10th percentile of human text (>= 4)
ended     : (multi-turn only) the model ended its turn within the 4K continuation

python -m lcs.ana_rescue  -> ~/lcg-cr/results/rescue_{gen,cells}.parquet + printed tables
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

RUNS = os.path.expanduser("~/lcg-cr/runs")
OUT = os.path.expanduser("~/lcg-cr/results")


def load_exp(exp):
    rows = []
    for run in sorted(glob.glob(f"{RUNS}/{exp}/*")):
        mf = f"{run}/metrics.jsonl"
        if not os.path.exists(mf):
            continue
        judge = {}
        jf = f"{run}/judge_gptoss/tail.jsonl"
        if os.path.exists(jf):
            for l in open(jf):
                r = json.loads(l)
                judge[r["uid"]] = (r.get("scores") or {}).get("coherence")
        fin = {}
        for l in open(f"{run}/meta.jsonl"):
            m = json.loads(l)
            fin[m["uid"]] = m.get("finish")
        for l in open(mf):
            r = json.loads(l)
            w = r["win"]
            rep = np.array([np.nan if x is None else x for x in w["rep32"]], float)
            H = np.array([np.nan if x is None else x for x in w["H1"]], float)
            tail = rep[-2:] if len(rep) >= 2 else rep
            rows.append({"exp": exp, "model": run.split("/")[-1], "uid": r["uid"], "src_sampler": r.get("src_sampler"),
                         "sampler": r["sampler"], "T": r["T"], "n_tokens": r["n_tokens"],
                         "rep_tail": float(np.nanmean(tail)), "H_tail": float(np.nanmean(H[-2:])) if len(H) else np.nan,
                         "coh_tail": judge.get(r["uid"]), "finish": fin.get(r["uid"])})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["escaped"] = df.rep_tail < 0.5
    df["recovered"] = df.escaped & (df.coh_tail.astype(float) >= 4)
    df["ended"] = df.finish == "stop"
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = [load_exp(e) for e in ("rescue", "rescue_mt")]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df.to_parquet(f"{OUT}/rescue_gen.parquet")
    cells = df.groupby(["exp", "sampler"]).agg(n=("uid", "size"), escaped=("escaped", "mean"),
                                               recovered=("recovered", "mean"), ended=("ended", "mean"),
                                               rep_tail=("rep_tail", "mean"), H_tail=("H_tail", "mean"),
                                               coh=("coh_tail", "mean")).reset_index()
    cells.to_parquet(f"{OUT}/rescue_cells.parquet")
    pd.set_option("display.width", 220, "display.max_rows", 400)
    for exp, g in cells.groupby("exp"):
        print("==", exp)
        print(g.sort_values("recovered", ascending=False).round(3).to_string(index=False))
    print(df.groupby(["exp", "model", "sampler"]).escaped.mean().unstack("sampler").round(2).to_string())


if __name__ == "__main__":
    main()
