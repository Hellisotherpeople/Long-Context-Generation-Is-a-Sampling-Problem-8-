"""Cross-benchmark summary: mean score (over models) of a few reference configurations on every benchmark.

python -m lcs.bench_summary -> ~/lcg-cr/results/bench_summary.{parquet,tex}
High temperature = the highest T of each benchmark's grid (1.5 for LongBench v2 / LongProc / SWE-bench, 1.3 otherwise).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

R = os.path.expanduser("~/lcg-cr/results")


def load():
    """benchmark -> DataFrame(model, rule, T, score in [0, 100])."""
    B = {}

    def add(name, f, rule_col, score_col, scale=100.0, split=False):
        if not os.path.exists(f"{R}/{f}"):
            return
        d = pd.read_parquet(f"{R}/{f}")
        if split:
            d["rule"] = d[rule_col].str.split("@T").str[0]
            d["T"] = d[rule_col].str.split("@T").str[1].astype(float)
            d.loc[d[rule_col] == "greedy", ["rule", "T"]] = ["greedy", 0.0]
        else:
            d = d.rename(columns={rule_col: "rule"})
        d["score"] = d[score_col].astype(float) * scale
        B[name] = d[["model", "rule", "T", "score"]]

    add("LongBench v2", "lb2_gen.parquet", "sampler", "correct")
    add("LongProc", "longproc_gen.parquet", "sampler", "score")
    add("AIME/HMMT 25--26", "reason2_gen.parquet", "sampler", "correct")
    add("LiveCodeBench v6", "lcb_gen.parquet", "sampler", "pass")
    add("LongBench-Write", "lbwrite_gen.parquet", "sampler", "S", scale=1.0)
    add("WritingBench", "wbench_gen.parquet", "sampler", "score", scale=10.0)
    if os.path.exists(f"{R}/swe_inst.parquet"):
        d = pd.read_parquet(f"{R}/swe_inst.parquet")
        d = d[~d.think]
        d["rule"] = d.config.str.split("@T").str[0]
        d["T"] = pd.to_numeric(d.config.str.split("@T").str[1], errors="coerce")
        d.loc[d.config == "greedy", ["rule", "T"]] = ["greedy", 0.0]
        d["score"] = 100.0 * d.resolved.astype(float)
        B["SWE-bench Verified"] = d.dropna(subset=["T"])[["model", "rule", "T", "score"]]
    return B


COLS = [("greedy", "greedy", None), ("top-$p$ 0.95, $T{=}1$", "topp0.95", 1.0), ("top-$p$ 0.95, high $T$", "topp0.95", "hi"),
        ("min-$p$ 0.1, high $T$", "minp0.1", "hi"), ("top-$n\\sigma$, high $T$", "nsigma1", "hi"),
        ("\\pless{}, $T{=}1$", "pless", 1.0), ("\\pless{}+keep-2, $T{=}1$", "pless+k2", 1.0)]


def main():
    B = load()
    rows = []
    for name, d in B.items():
        Ts = sorted(t for t in d["T"].unique() if t > 0)
        hi = max(Ts)
        r = {"benchmark": name, "models": d.model.nunique(), "high_T": hi}
        for lab, rule, T in COLS:
            if rule == "greedy":
                s = d[d.rule == "greedy"]
            else:
                tt = hi if T == "hi" else T
                rr = "topp0.9" if rule == "topp0.95" and "topp0.95" not in set(d.rule) else rule  # WritingBench grid
                s = d[(d.rule == rr) & np.isclose(d["T"], tt)]
            r[lab] = s.groupby("model").score.mean().mean() if len(s) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df[[c[0] for c in COLS]].notna().any(axis=1)]
    rows = df.to_dict("records")
    df.to_parquet(f"{R}/bench_summary.parquet")
    pd.set_option("display.width", 250)
    print(df.round(1).to_string())
    with open(f"{R}/bench_summary.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "c" * (len(COLS) + 1) + "}\n\\toprule\n")
        f.write("Benchmark & high $T$ & " + " & ".join(c[0] for c in COLS) + "\\\\\n\\midrule\n")
        for r in rows:
            vals = [r[c[0]] for c in COLS]
            best = np.nanmax(vals)
            cells = ["--" if np.isnan(v) else (f"\\textbf{{{v:.1f}}}" if v == best else f"{v:.1f}") for v in vals]
            f.write(f"{r['benchmark']} ({r['models']}) & {r['high_T']:g} & " + " & ".join(cells) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
