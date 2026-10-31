"""Agreement between the primary judge (GPT-OSS-120B) and the second judge (GLM-4.5-Air).

python -m lcs.ana_judges [--second glm45air] -> printed summary + ~/lcg-cr/results/judge_agreement.json
Item level: Spearman correlation of global quality and of last-window coherence.  Configuration level: Spearman
correlation of mean global quality per (experiment, model, sampler, T) cell, the unit our conclusions are drawn on.
Healthy agreement: each judge's healthy threshold is calibrated on human PG-19 windows (10th percentile, as in the main
analysis) and we report how often the two judges agree on healthy vs. degenerate for non-looping generations.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .figs import load, load_judge

RES = os.path.expanduser("~/lcg-cr/results")


def human_threshold(judge):
    J = load_judge("book32k_human", judge=judge)
    c = [s.get("coherence") for j in J.values() for s in j["win"].values() if s and s.get("coherence") is not None]
    return float(np.quantile(c, 0.10)) if c else np.nan, len(c)


def table(exps, judge):
    rows = []
    for exp in exps:
        J = load_judge(exp, judge=judge)
        df = load(exp)
        for _, r in df.iterrows():
            j = J.get((r["model"], r["uid"]))
            if not j:
                continue
            last = j["win"][max(j["win"])] if j["win"] else None
            lo = r["loop_onset_win"]
            rows.append({"exp": exp, "model": r["model"], "uid": r["uid"], "sampler": r["sampler"], "T": r["T"],
                         "loop": lo is not None and not (isinstance(lo, float) and np.isnan(lo)),
                         "gov": (j["global"] or {}).get("overall", np.nan),
                         "coh_last": (last or {}).get("coherence", np.nan)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="gptoss")
    ap.add_argument("--second", default="glm45air")
    ap.add_argument("--exps", default="mt32k,mtk2_32k")
    a = ap.parse_args()
    exps = a.exps.split(",")
    A, B = table(exps, a.first), table(exps, a.second)
    m = A.merge(B, on=["exp", "model", "uid", "sampler", "T", "loop"], suffixes=("_1", "_2"))
    out = {"n_items": int(len(m))}
    for col in ("gov", "coh_last"):
        x = m[[f"{col}_1", f"{col}_2"]].dropna()
        out[f"item_spearman_{col}"] = float(spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]) if len(x) > 2 else np.nan
    cells = m.groupby(["exp", "model", "sampler", "T"])[["gov_1", "gov_2"]].mean().dropna()
    out["cell_spearman_gov"] = float(spearmanr(cells.gov_1, cells.gov_2)[0]) if len(cells) > 2 else np.nan
    out["n_cells"] = int(len(cells))
    t1, n1 = human_threshold(a.first)
    t2, n2 = human_threshold(a.second)
    out.update(thr_first=t1, thr_second=t2, n_human_windows=[n1, n2])
    nl = m[~m.loop].dropna(subset=["coh_last_1", "coh_last_2"])
    h1, h2 = nl.coh_last_1 >= t1, nl.coh_last_2 >= t2
    out["healthy_agreement"] = float((h1 == h2).mean()) if len(nl) else np.nan
    out["healthy_rate_first"], out["healthy_rate_second"] = float(h1.mean()), float(h2.mean())
    # does the second judge reproduce the per-model ranking of rules at their best temperature?
    best = cells.reset_index().groupby(["exp", "model", "sampler"])[["gov_1", "gov_2"]].max().reset_index()
    rk = [spearmanr(g.gov_1, g.gov_2)[0] for _, g in best.groupby(["exp", "model"]) if len(g) > 3]
    out["rule_ranking_spearman_median"] = float(np.nanmedian(rk)) if rk else np.nan
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{RES}/judge_agreement.json", "w"), indent=1)


if __name__ == "__main__":
    main()
