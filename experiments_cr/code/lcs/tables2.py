"""Appendix tables for the wave-2/3 benchmarks (analysis venv).  python -m lcs.tables2 -> ~/lcg-cr/results/tables/*.tex"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .figs2 import SHORT

R = os.path.expanduser("~/lcg-cr/results")
OUT = f"{R}/tables"
RULES = [("greedy", "greedy"), ("temp", "pure"), ("topk50", "top-$k$"), ("topp0.9", "top-$p$ .9"), ("topp0.95", "top-$p$ .95"),
         ("minp0.05", "min-$p$ .05"), ("minp0.1", "min-$p$ .1"), ("toph0.4", "Top-H"), ("pless", "\\pless{}"),
         ("nsigma1", "top-$n\\sigma$"), ("toph0.4+k2", "Top-H+k2"), ("pless+k2", "\\pless{}+k2"), ("nsigma1+k2", "top-$n\\sigma$+k2")]


TASKL = {"html_to_tsv_8k": "HTML$\\to$TSV 8K", "pseudo_to_code_2k": "pseudocode$\\to$code 2K",
         "path_traversal_8k": "path traversal 8K", "tom_tracking_8k": "ToM tracking 8K", "countdown_8k": "Countdown 8K",
         "travel_planning_8k": "travel planning 8K"}


def wide(d, row, val, fmt="{:.1f}", scale=1.0, best="max"):
    """rows = `row` values, columns = (rule, T) present in the data; bold = best per row."""
    d = d.copy()
    cols = []
    for r, lab in RULES:
        Ts = sorted(d[d.rule == r]["T"].unique())
        cols += [(r, t, lab) for t in Ts]
    lines = []
    head1 = " & ".join(f"\\multicolumn{{{sum(1 for c in cols if c[0] == r)}}}{{c}}{{{lab}}}"
                       for r, lab in RULES if any(c[0] == r for c in cols))
    head2 = " & ".join("--" if t == 0 else f"{t:g}" for r, t, _ in cols)
    lines.append("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule")
    lines.append(f" & {head1}\\\\")
    lines.append(f"$T$ & {head2}\\\\\n\\midrule")
    for key, g in d.groupby(row, sort=False):
        vals = [g[(g.rule == r) & np.isclose(g["T"], t)][val].mean() * scale for r, t, _ in cols]
        b = np.nanmax(vals) if best == "max" else (np.nanmin(vals) if best == "min" else np.nan)
        cells = ["--" if np.isnan(v) else (f"\\textbf{{{fmt.format(v)}}}" if np.isclose(v, b) else fmt.format(v)) for v in vals]
        name = TASKL.get(key, SHORT.get(key, str(key).replace("_", "\\_")))
        lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    lines.append("\\bottomrule\n\\end{tabular}")
    return "\n".join(lines) + "\n"


COMPACT = {"html_to_tsv_8k": "HTML$\\to$TSV", "pseudo_to_code_2k": "pseudo$\\to$code", "path_traversal_8k": "path",
           "tom_tracking_8k": "ToM", "countdown_8k": "Countdown", "travel_planning_8k": "travel"}


def compact(k):
    if k in COMPACT:
        return COMPACT[k]
    n = SHORT.get(k, str(k))
    for a, b in (("Qwen3.5-", "Qwen "), ("Qwen3.6-", "Qwen3.6 "), ("Qwen3.8-", "Qwen3.8 "), ("Llama-3.1-", "Llama "),
                 ("Llama-3.3-", "Llama "), ("Nemotron-3-Nano-", "Nemotron "), ("Gemma-4-", "Gemma "),
                 ("Ministral-3-", "Ministral "), ("OLMo-3-", "OLMo ")):
        n = n.replace(a, b)
    return n.replace("_", "\\_")


def tall(d, col, val, fmt="{:.1f}", scale=1.0, best="max"):
    """rows = (rule, T) present in the data, columns = `col` values (models or tasks) + mean; bold = best per column."""
    keys = list(dict.fromkeys(d[col]))
    keys = [k for k in SHORT if k in keys] + [k for k in keys if k not in SHORT]
    rows = []
    for r, lab in RULES:
        for t in sorted(d[d.rule == r]["T"].unique()):
            g = d[(d.rule == r) & np.isclose(d["T"], t)]
            vals = [g[g[col] == k][val].mean() * scale for k in keys]
            rows.append((lab, t, vals + [np.nanmean(vals)]))
    M = np.array([v for _, _, v in rows], float)
    with np.errstate(all="ignore"):
        bcol = np.nanmax(M, 0) if best == "max" else (np.nanmin(M, 0) if best == "min" else np.full(M.shape[1], np.nan))
    names = [compact(k) for k in keys] + ["mean"]
    out = ["\\begin{tabular}{ll" + "c" * len(names) + "}\n\\toprule",
           "rule & $T$ & " + " & ".join(names) + "\\\\\n\\midrule"]
    prev = None
    for (lab, t, _), v in zip(rows, M):
        cells = ["--" if np.isnan(x) else (f"\\textbf{{{fmt.format(x)}}}" if np.isclose(x, b) else fmt.format(x))
                 for x, b in zip(v, bcol)]
        out.append(f"{lab if lab != prev else ''} & {'--' if t == 0 else f'{t:g}'} & " + " & ".join(cells) + "\\\\")
        prev = lab
    out.append("\\bottomrule\n\\end{tabular}")
    return "\n".join(out) + "\n"


SWE_ROWS = [("greedy", "greedy"), ("temp@T0.7", "pure sampling, $T{=}0.7$"), ("temp@T1.0", "pure sampling, $T{=}1$"),
            ("topp0.95@T1.0", "top-$p$ 0.95, $T{=}1$"), ("vendor", "vendor default"),
            ("minp0.1@T1.0", "min-$p$ 0.1, $T{=}1$"), ("toph0.4@T1.0", "Top-H, $T{=}1$"), ("pless@T1.0", "\\pless{}, $T{=}1$"),
            ("nsigma1@T1.0", "top-$n\\sigma$, $T{=}1$"), ("topp0.95@T1.5", "top-$p$ 0.95, $T{=}1.5$"),
            ("minp0.1@T1.5", "min-$p$ 0.1, $T{=}1.5$"), ("nsigma1@T1.5", "top-$n\\sigma$, $T{=}1.5$")]


def swe_table(d):
    """SWE-bench Verified: resolved %, repeated-action loops %, mean turns, per model (columns) and configuration (rows)."""
    d = d[~d.think]
    models = [m for m in SHORT if m in set(d.model)]
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{compact(m)}}}" for m in models)
    sub = " & ".join("resolved & loops & turns" for _ in models)
    out = ["\\begin{tabular}{l" + "rrr" * len(models) + "}\n\\toprule", f" & {head}\\\\", f"sampler & {sub}\\\\\n\\midrule"]
    best = {m: d[d.model == m].groupby("config").resolved.mean().max() for m in models}
    for cfg, lab in SWE_ROWS:
        cells = []
        for m in models:
            g = d[(d.model == m) & (d.config == cfg)]
            if g.empty or g.resolved.isna().all():
                cells += ["--", "--", "--"]
                continue
            r = 100 * g.resolved.mean()
            rs = f"{r:.1f}"
            if np.isclose(g.resolved.mean(), best[m]):
                rs = f"\\textbf{{{rs}}}"
            cells += [rs, f"{100 * g.loop.mean():.1f}", f"{g.steps.mean():.0f}"]
        out.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    out.append("\\bottomrule\n\\end{tabular}")
    return "\n".join(out) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{R}/longproc_gen.parquet"):
        d = pd.read_parquet(f"{R}/longproc_gen.parquet").rename(columns={"sampler": "rule"})
        open(f"{OUT}/longproc_task.tex", "w").write(tall(d, "task", "score", scale=100))
        open(f"{OUT}/longproc_model.tex", "w").write(tall(d, "model", "score", scale=100))
    if os.path.exists(f"{R}/lbwrite_gen.parquet"):
        d = pd.read_parquet(f"{R}/lbwrite_gen.parquet").rename(columns={"sampler": "rule"})
        open(f"{OUT}/lbwrite_model.tex", "w").write(tall(d, "model", "S"))
    if os.path.exists(f"{R}/wbench_gen.parquet"):
        d = pd.read_parquet(f"{R}/wbench_gen.parquet").rename(columns={"sampler": "rule"})
        open(f"{OUT}/wbench_model.tex", "w").write(tall(d, "model", "score", fmt="{:.2f}"))
    if os.path.exists(f"{R}/lb2_gen.parquet"):
        d = pd.read_parquet(f"{R}/lb2_gen.parquet").rename(columns={"sampler": "rule"})
        open(f"{OUT}/lb2_model.tex", "w").write(tall(d, "model", "correct", scale=100))
    if os.path.exists(f"{R}/reason2_gen.parquet"):
        d = pd.read_parquet(f"{R}/reason2_gen.parquet").rename(columns={"sampler": "rule"})
        open(f"{OUT}/reason2_model.tex", "w").write(tall(d, "model", "correct", scale=100))
        open(f"{OUT}/reason2_loops.tex", "w").write(tall(d, "model", "looped", scale=100, fmt="{:.0f}", best=None))
    if os.path.exists(f"{R}/swe_inst.parquet"):
        open(f"{OUT}/swe.tex", "w").write(swe_table(pd.read_parquet(f"{R}/swe_inst.parquet")))
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
