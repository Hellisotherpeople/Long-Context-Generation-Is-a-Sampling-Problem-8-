"""Paired bootstrap tests for the benchmark claims (analysis venv).

For each benchmark and pair of configurations (A, B) evaluated on the same items, the statistic is the mean over models of
mean_item(score_A - score_B); items are resampled within each model (B=2000).  Also: the drop from T=1 to high T per rule.
python -m lcs.stats2 -> ~/lcg-cr/results/stats2.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .bench_summary import load

R = os.path.expanduser("~/lcg-cr/results")
rng = np.random.default_rng(0)


def item_key(d):
    return d.groupby(["model", "rule", "T"]).cumcount()  # fallback when no item id


def paired(d, a, b, B=2000):
    """a, b = (rule, T).  Returns mean difference (points) and 95% CI."""
    diffs = []
    for m, g in d.groupby("model"):
        x = g[(g.rule == a[0]) & np.isclose(g["T"], a[1])]
        y = g[(g.rule == b[0]) & np.isclose(g["T"], b[1])]
        if not len(x) or not len(y):
            continue
        xs = x.groupby("item").score.mean()
        ys = y.groupby("item").score.mean()
        idx = xs.index.intersection(ys.index)
        diffs.append((xs[idx] - ys[idx]).to_numpy())
    if not diffs:
        return None
    est = float(np.mean([v.mean() for v in diffs]))
    boots = np.array([np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in diffs]) for _ in range(B)])
    return {"diff": round(est, 2), "lo": round(float(np.percentile(boots, 2.5)), 2),
            "hi": round(float(np.percentile(boots, 97.5)), 2), "models": len(diffs)}


def main():
    B = load()
    # item ids: reload with ids where available
    ids = {"LongBench v2": ("lb2_gen.parquet", "pid"), "LongProc": ("longproc_gen.parquet", ["task", "item"]),
           "AIME/HMMT 25--26": ("reason2_gen.parquet", ["pid", "i"]), "LiveCodeBench v6": ("lcb_gen.parquet", ["pid", "i"]),
           "LongBench-Write": ("lbwrite_gen.parquet", "pid"), "WritingBench": ("wbench_gen.parquet", "pid")}
    out = {}
    for name, d in B.items():
        if name in ids and os.path.exists(f"{R}/{ids[name][0]}"):
            raw = pd.read_parquet(f"{R}/{ids[name][0]}")
            cols = ids[name][1] if isinstance(ids[name][1], list) else [ids[name][1]]
            d = d.copy()
            d["item"] = raw[cols].astype(str).agg("/".join, axis=1).to_numpy()
        elif name == "SWE-bench Verified":
            raw = pd.read_parquet(f"{R}/swe_inst.parquet")
            raw = raw[~raw.think]
            d = d.copy()
            d["item"] = raw.loc[d.index, "instance"].to_numpy()
        else:
            continue
        hi = max(t for t in d["T"].unique() if t > 0)
        tp = "topp0.95" if "topp0.95" in set(d.rule) else "topp0.9"
        res = {"high_T": hi}
        for lab, a, b in (("nsigma_vs_topp_highT", ("nsigma1", hi), (tp, hi)),
                          ("minp_vs_topp_highT", ("minp0.1", hi), (tp, hi)),
                          ("pless_vs_topp_highT", ("pless", hi), (tp, hi)),
                          ("topp_highT_vs_T1", (tp, hi), (tp, 1.0)),
                          ("nsigma_highT_vs_T1", ("nsigma1", hi), ("nsigma1", 1.0)),
                          ("plessk2_vs_pless_T1", ("pless+k2", 1.0), ("pless", 1.0)),
                          ("nsigma_highT_vs_greedy", ("nsigma1", hi), ("greedy", 0.0))):
            r = paired(d, a, b)
            if r:
                res[lab] = r
        out[name] = res
        print(name, json.dumps(res))
    json.dump(out, open(f"{R}/stats2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
