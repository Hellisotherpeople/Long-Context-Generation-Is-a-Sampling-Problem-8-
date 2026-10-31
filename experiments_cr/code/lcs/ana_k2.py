"""Absorption-free truncation (keep_top=2) vs. the base rule in natural multi-turn generation.

python -m lcs.ana_k2 [--base mt32k --k2 mtk2_32k]  -> ~/lcg-cr/results/k2_cells.parquet, ~/lcg-cr/tables/k2.tex
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .figs import LABEL, load, load_judge, outcome_judge
from .tables import SHORT

RES = os.path.expanduser("~/lcg-cr/results")
TAB = os.path.expanduser("~/lcg-cr/tables")
BASES = ["pless", "toph0.4", "nsigma1", "minp0.1", "topp0.9"]


def cells(exp):
    J = load_judge(exp)
    df = load(exp)
    df = df[df["T"] > 0].copy()
    df["base"] = df.sampler.str.replace("+k2", "", regex=False)
    df["k2"] = df.sampler.str.endswith("+k2")
    df["out"] = df.apply(lambda r: outcome_judge(r, J), axis=1)
    df["gov"] = df.apply(lambda r: ((J.get((r["model"], r["uid"])) or {}).get("global") or {}).get("overall", np.nan), axis=1)
    df["loop"] = df.out == "loop"
    df["healthy"] = df.out == "healthy"
    df = df[df.out.notna()]
    return df.groupby(["model", "base", "k2", "T"]).agg(n=("uid", "size"), loop=("loop", "mean"),
                                                       healthy=("healthy", "mean"), gov=("gov", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mt32k")
    ap.add_argument("--k2", default="mtk2_32k")
    a = ap.parse_args()
    c = pd.concat([cells(a.base), cells(a.k2)], ignore_index=True)
    ORDER_M = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b",
               "nemotron3-30b-a3b", "gemma4-e4b", "llama3.1-8b", "ministral3-8b"]
    both = set(c[c.k2].model) & set(c[~c.k2].model)
    models = [m for m in ORDER_M if m in both] + sorted(m for m in both if m not in ORDER_M)
    c = c[c.model.isin(models) & c.base.isin(BASES)]
    c.to_parquet(f"{RES}/k2_cells.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 500)
    for m in models:
        print("==", m)
        print(c[c.model == m].pivot_table(index=["base", "T"], columns="k2", values=["loop", "healthy", "gov"]).round(2).to_string())
    # table: per model, T=1.0 loop/healthy without -> with keep_top, and best judged quality over T
    lines = ["\\begin{tabular}{ll" + "c" * len(models) + "}", "\\toprule",
             " & & " + " & ".join(SHORT.get(m, m) for m in models) + " \\\\", "\\midrule"]
    for b in BASES:
        for lab, f in (("loops $T{=}1$", "loop"), ("healthy $T{=}1$", "healthy"), ("best quality", "gov")):
            cellv = []
            for m in models:
                s = c[(c.model == m) & (c.base == b)]
                if f == "gov":
                    v0 = s[~s.k2].gov.max()
                    v1 = s[s.k2].gov.max()
                    cellv.append("--" if np.isnan(v0) or np.isnan(v1) else f"{v0:.1f}$\\to${v1:.1f}")
                else:
                    x0 = s[(~s.k2) & (s["T"] == 1.0)][f]
                    x1 = s[(s.k2) & (s["T"] == 1.0)][f]
                    cellv.append("--" if not len(x0) or not len(x1) else f"{100 * x0.iloc[0]:.0f}$\\to${100 * x1.iloc[0]:.0f}")
            name = LABEL.get(b, b) if lab.startswith("loops") else ""
            lines.append(f"{name} & {lab} & " + " & ".join(cellv) + " \\\\")
        lines.append("\\midrule" if b != BASES[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    os.makedirs(TAB, exist_ok=True)
    open(f"{TAB}/k2.tex", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
