"""WritingBench aggregation: critic scores (lcs.wb_critic, 1-10 per query-specific criterion) -> per response mean ->
per (model, sampler, T).  Also response length and loop flag from metrics.jsonl.

python -m lcs.ana_wbench -> ~/lcg-cr/results/wbench_{gen,cells}.parquet
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

H = os.path.expanduser("~/lcg-cr")


def main():
    rows = []
    for run in sorted(glob.glob(f"{H}/runs/wbench/*")):
        cf = f"{run}/critic.jsonl"
        if not os.path.exists(cf):
            continue
        sc = {}
        for l in open(cf):
            r = json.loads(l)
            if r.get("score") is not None:
                sc.setdefault(r["uid"], []).append(min(10, max(1, r["score"])))
        mt = {}
        if os.path.exists(f"{run}/metrics.jsonl"):
            for l in open(f"{run}/metrics.jsonl"):
                r = json.loads(l)
                mt[r["uid"]] = r
        for l in open(f"{run}/meta.jsonl"):
            m = json.loads(l)
            mm, r = m["meta"], mt.get(m["uid"], {})
            lo = r.get("loop_onset_win")
            rows.append({"model": run.split("/")[-1], "pid": mm["pid"], "sampler": mm["sampler"], "T": mm["T"],
                         "score": float(np.mean(sc[m["uid"]])) if m["uid"] in sc else np.nan,
                         "n_crit": len(sc.get(m["uid"], [])), "n_tokens": m["n_tokens"],
                         "truncated": m.get("finish") == "length",
                         "looped": lo is not None and not (isinstance(lo, float) and np.isnan(lo))})
    df = pd.DataFrame(rows)
    os.makedirs(f"{H}/results", exist_ok=True)
    df.to_parquet(f"{H}/results/wbench_gen.parquet")
    cells = df.groupby(["model", "sampler", "T"]).agg(n=("pid", "size"), score=("score", "mean"),
                                                      looped=("looped", "mean"), truncated=("truncated", "mean"),
                                                      length=("n_tokens", "mean")).reset_index()
    cells.to_parquet(f"{H}/results/wbench_cells.parquet")
    pd.set_option("display.width", 220, "display.max_rows", 400)
    print(cells.pivot_table(index="sampler", columns="T", values="score").round(2).to_string())
    print(cells.round(3).to_string())


if __name__ == "__main__":
    main()
