"""LaTeX tables for the paper (run with the analysis venv after the analyses; writes ~/lcg-cr/tables/*.tex)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .figs import COL, LABEL, PARAMS, escape_T, load, load_judge, outcome_judge

OUT = os.path.expanduser("~/lcg-cr/tables")
os.makedirs(OUT, exist_ok=True)
RES = os.path.expanduser("~/lcg-cr/results")
ORDER = ["temp", "topk50", "topp0.9", "topp0.95", "minp0.05", "minp0.1", "toph0.4", "pless", "nsigma1"]


def tex_label(s):
    return LABEL.get(s, s).replace("$", "$")


SHORT = {"qwen3.5-0.8b": "Q-0.8B", "qwen3.5-2b": "Q-2B", "qwen3.5-4b": "Q-4B", "qwen3.5-9b": "Q-9B",
         "qwen3.5-27b": "Q-27B", "qwen3.5-35b-a3b": "Q-35B-A3B", "qwen3.5-122b-a10b": "Q-122B-A10B",
         "nemotron3-30b-a3b": "Nem-30B-A3B", "gemma4-e4b": "G-E4B", "gemma4-31b": "G-31B", "olmo3-7b": "OLMo-7B",
         "llama3.2-1b": "L-1B", "llama3.1-8b": "L-8B", "ministral3-14b": "Min-14B", "ministral3-8b": "Min-8B", "llama3.3-70b": "L-70B",
         "glm4.5-air": "GLM-Air", "mistral-7b-v0.3": "Mis-7B", "mistral-nemo-12b": "Nemo-12B",
         "qwen2.5-1.5b": "Q2.5-1.5B", "qwen2.5-3b": "Q2.5-3B", "qwen2.5-14b": "Q2.5-14B",
         "qwen3.6-27b": "Q3.6-27B", "qwen3.8-27b": "Q3.8-27B"}


def t_escape():
    df = load("core32k")
    df = df[(df["T"] > 0) & df.model.isin(SHORT)]
    df["looped"] = df.loop_onset_win.notna()
    rows = {}
    for (m, s), sub in df.groupby(["model", "sampler"]):
        if sub["T"].nunique() >= 4:
            rows.setdefault(s, {})[m] = escape_T(sub)
    models = [m for m in SHORT if m in set(df.model)]
    lines = ["\\begin{tabular}{l" + "c" * len(models) + "}", "\\toprule",
             "Rule & " + " & ".join(f"\\rotatebox{{70}}{{{SHORT[m]}}}" for m in models) + " \\\\", "\\midrule"]
    for s in ORDER:
        if s not in rows:
            continue
        vals = [rows[s].get(m, np.nan) for m in models]
        tmin = df["T"].min()
        cells = ["never" if np.isinf(v) else ("--" if np.isnan(v) else (f"$\\le${tmin:g}" if v <= tmin + 1e-9 else f"{v:.2f}"))
                 for v in vals]
        lines.append(f"{tex_label(s)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/escape_T.tex", "w").write("\n".join(lines))


def best_T_table(exp, models, name, metric="gov"):
    J = load_judge(exp)
    out = {}
    have = set(load(exp).model)
    models = [m for m in models if m in have]
    for m in models:
        df = load(exp, [m])
        if not len(df):
            continue
        df["gov"] = df.apply(lambda r: ((J.get((r["model"], r["uid"])) or {}).get("global") or {}).get("overall", np.nan), axis=1)
        df["healthy"] = df.apply(lambda r: outcome_judge(r, J), axis=1) == "healthy"
        df = df[df["T"] > 0]
        g = df.groupby(["sampler", "T"]).agg(gov=("gov", "mean"), healthy=("healthy", "mean"), n=("uid", "size")).reset_index()
        for s, sub in g.groupby("sampler"):
            if sub[metric].isna().all():
                continue
            b = sub.loc[sub[metric].idxmax()]
            out.setdefault(s, {})[m] = (b[metric], b["T"], b["healthy"])
    lines = ["\\begin{tabular}{l" + "c" * len(models) + "}", "\\toprule",
             "Rule & " + " & ".join(f"\\rotatebox{{70}}{{{SHORT.get(m, m)}}}" for m in models) + " \\\\", "\\midrule"]
    for s in ORDER + sorted(k for k in out if k not in ORDER):
        if s not in out:
            continue
        cells = []
        for m in models:
            v = out[s].get(m)
            best = max((out[r][m][0] for r in out if m in out[r]), default=None)
            txt = "--" if v is None else f"{v[0]:.2f} ({v[1]:g})"
            cells.append(f"\\textbf{{{txt}}}" if v is not None and best is not None and abs(v[0] - best) < 1e-9 else txt)
        lines.append(f"{tex_label(s)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/{name}.tex", "w").write("\n".join(lines))
    return out


def main():
    t_escape()
    best_T_table("mt32k", ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b",
                           "qwen3.5-122b-a10b", "nemotron3-30b-a3b", "gemma4-e4b", "gemma4-31b", "llama3.1-8b",
                           "ministral3-8b"], "natural_bestT")
    print(open(f"{OUT}/escape_T.tex").read())
    print(open(f"{OUT}/natural_bestT.tex").read())


if __name__ == "__main__":
    main()
