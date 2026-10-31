"""Wave-2/3 paper figures: benchmarks (LongBench v2, LongProc, LongBench-Write, WritingBench), diversity frontier,
agentic coding, loop rescue.  python -m lcs.figs2 <name> [...]   (analysis venv; reads ~/lcg-cr/results)"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figs import COL, LABEL, OUT

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "lines.linewidth": 1.4, "lines.markersize": 3.5})

R = os.path.expanduser("~/lcg-cr/results")
CLASSIC = ["temp", "topk50", "topp0.9", "topp0.95"]
MODERN = ["minp0.1", "toph0.4", "pless", "nsigma1"]
K2 = ["pless+k2", "nsigma1+k2"]
COL2 = dict(COL, **{"pless+k2": "#1abc9c", "nsigma1+k2": "#5499c7", "minp0.05+k2": "#85c1e9", "topp0.95+k2": "#f1948a"})
LABEL2 = dict(LABEL, **{"pless+k2": "P-less + keep-2", "nsigma1+k2": "top-$n\\sigma$ + keep-2"})
SHORT = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B",
         "qwen3.5-27b": "Qwen3.5-27B", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B", "qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
         "qwen3.6-27b": "Qwen3.6-27B", "qwen3.8-27b": "Qwen3.8-27B", "llama3.1-8b": "Llama-3.1-8B",
         "llama3.3-70b": "Llama-3.3-70B", "nemotron3-30b-a3b": "Nemotron-3-Nano-30B", "gemma4-e4b": "Gemma-4-E4B",
         "gemma4-31b": "Gemma-4-31B", "ministral3-8b": "Ministral-3-8B", "ministral3-14b": "Ministral-3-14B",
         "olmo3-7b": "OLMo-3-7B", "glm4.5-air": "GLM-4.5-Air", "gpt-oss-20b": "gpt-oss-20b"}


def split_cfg(df, col="sampler"):
    """Rows with sampler names like 'minp0.1@T1.0' or separate sampler/T columns -> (rule, T)."""
    if "T" not in df.columns:
        df = df.copy()
        df["rule"] = df[col].str.split("@T").str[0]
        df["T"] = df[col].str.split("@T").str[1].astype(float)
    else:
        df = df.assign(rule=df[col])
    return df


def _family_lines(ax, g, x="T", y="acc", rules=CLASSIC + MODERN, ref=None, lw=1.8):
    for rule in rules:
        s = g[g.rule == rule].sort_values(x)
        if s.empty:
            continue
        ls = "-" if rule in MODERN or rule in K2 else (0, (4, 2))
        ax.plot(s[x], s[y], ls=ls, marker="o", ms=3.5, lw=lw, color=COL2.get(rule, "#888"), label=LABEL2.get(rule, rule))
    if ref is not None:
        ax.axhline(ref, color="k", lw=1, ls=":", label="greedy")


def fig_lb2(name="fig_lb2"):
    """LongBench v2 (503 questions, contexts middle-truncated to 120K tokens): (a) accuracy relative to greedy, averaged
    over models, vs T; (b) per-model accuracy at T=1.5 (classic vs modern)."""
    df = pd.read_parquet(f"{R}/lb2_gen.parquet")
    df = split_cfg(df)
    greedy = df[df.rule == "greedy"].groupby("model").correct.mean()
    cells = df[df.rule != "greedy"].groupby(["model", "rule", "T"]).correct.mean().rename("acc").reset_index()
    cells["rel"] = cells.acc - cells.model.map(greedy)
    models = [m for m in SHORT if m in set(cells.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    agg = cells.groupby(["rule", "T"]).rel.agg(["mean", "sem"]).reset_index()
    for rule in CLASSIC + MODERN:
        s = agg[agg.rule == rule].sort_values("T")
        ls = "-" if rule in MODERN else (0, (4, 2))
        ax.errorbar(s["T"], 100 * s["mean"], yerr=100 * s["sem"], ls=ls, marker="o", ms=2.5, lw=1.4, capsize=1.5,
                    color=COL2[rule], label=LABEL2[rule])
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel("accuracy $-$ greedy accuracy (points)")
    ax.set_title(f"(a) mean over {len(models)} models", fontsize=7.5)
    ax.set_xticks(sorted(cells["T"].unique()))
    ax.legend(fontsize=5.8, ncol=2, loc="lower left")
    ax = axes[1]
    t15 = cells[cells["T"] == 1.5]
    w = 0.8 / len(CLASSIC + MODERN)
    for i, rule in enumerate(CLASSIC + MODERN):
        v = [100 * t15[(t15.model == m) & (t15.rule == rule)].acc.mean() for m in models]
        ax.bar(np.arange(len(models)) + (i - 3.5) * w, v, w, color=COL2[rule], label=LABEL2[rule],
               hatch="//" if rule in CLASSIC else None, edgecolor="white", lw=0.3)
    ax.scatter(np.arange(len(models)), [100 * greedy[m] for m in models], marker="_", s=150, color="k", zorder=5,
               label="greedy")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([SHORT[m] for m in models], rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("accuracy at $T=1.5$ (%)")
    ax.set_title("(b) every model, $T=1.5$ (hatched: classic samplers)", fontsize=7.5)
    ax.axhline(25, color="#999", lw=0.8, ls="--")
    ax.text(len(models) - 0.5, 25.5, "chance", fontsize=5.5, color="#777", ha="right", va="bottom")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    print(cells.pivot_table(index=["rule"], columns="T", values="rel").mul(100).round(1).to_string())



def _cfg_label(c):
    if c == "same":
        return "same sampler (control)"
    if c == "greedy":
        return "greedy"
    rule, _, T = c.partition("@T")
    names = {"temp": "pure sampling", "topk50": "top-$k$ 50", "topp0.9": "top-$p$ 0.9", "minp0.1": "min-$p$ 0.1",
             "nsigma1": "top-$n\\sigma$", "nsigma1+k2": "top-$n\\sigma$ + keep-2", "pless+k2": "P-less + keep-2",
             "toph0.4": "Top-H 0.4", "dry+topp0.9": "DRY + top-$p$ 0.9"}
    return f"{names.get(rule, rule)}, $T{{=}}{T}$"


def fig_rescue(name="fig_rescue"):
    """Loop rescue: resume generation from inside an existing loop with a different sampler (same model, same context).
    Stacked bars: recovered (escaped + judged coherent), escaped but incoherent, still looping."""
    df = pd.read_parquet(f"{R}/rescue_gen.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    titles = {"rescue": "(a) base models continuing a book (4 models, 24 loops each)",
              "rescue_mt": "(b) chat models, multi-turn writing (5 models)"}
    for ax, exp in zip(axes, ("rescue", "rescue_mt")):
        g = df[df.exp == exp].groupby("sampler").agg(esc=("escaped", "mean"), rec=("recovered", "mean"),
                                                      n=("uid", "size")).reset_index()
        g = g.sort_values(["esc", "rec"])
        y = np.arange(len(g))
        ax.barh(y, 100 * g.rec, color="#27ae60", label="escaped, coherent")
        ax.barh(y, 100 * (g.esc - g.rec), left=100 * g.rec, color="#8e44ad", alpha=0.75, label="escaped, incoherent")
        ax.barh(y, 100 * (1 - g.esc), left=100 * g.esc, color="#c0392b", alpha=0.8, label="still looping")
        ax.set_yticks(y)
        ax.set_yticklabels([_cfg_label(c) for c in g.sampler], fontsize=6)
        for t in ax.get_yticklabels():
            if t.get_text().startswith("same"):
                t.set_fontweight("bold")
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of continuations (4K tokens after the cut)")
        ax.set_title(titles[exp], fontsize=7.2)
        ax.grid(axis="y", visible=False)
    axes[0].legend(fontsize=5.8, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")



def load_divx():
    """All divx runs: per-(model, prompt, rule, T) group metrics + each model's greedy reward."""
    import glob
    import json
    frames, greedy = [], {}
    for f in sorted(glob.glob(os.path.expanduser("~/lcg-cr/runs/divx/*/divscore_groups.parquet"))):
        run = os.path.dirname(f)
        model = os.path.basename(run)
        g = pd.read_parquet(f).assign(model=model)
        frames.append(g)
        gu = {json.loads(l)["uid"] for l in open(f"{run}/meta.jsonl") if '"greedy"' in l}
        rs = [json.loads(l)["reward"] for l in open(f"{run}/divscore_samples.jsonl")]
        us = [json.loads(l)["uid"] for l in open(f"{run}/divscore_samples.jsonl")]
        greedy[model] = float(np.nanmean([r for u, r in zip(us, rs) if u in gu]))
    df = pd.concat(frames, ignore_index=True).rename(columns={"sampler": "rule"})
    df["rel_reward"] = df.reward_mean - df.model.map(greedy)
    return df, greedy


DIVX_RULES = ["temp", "topk50", "topp0.95", "minp0.1", "toph0.4", "pless", "nsigma1", "pless+k2", "nsigma1+k2"]


def fig_divx(name="fig_divx"):
    """Quality-diversity frontier on NoveltyBench (curated + WildChat) and Infinity-Chat prompts, 8 samples per prompt.
    (a) distinct answers (NoveltyBench classes, of 8) vs reward relative to greedy, one curve per rule through
    T = 0.7 ... 2.0; (b) NoveltyBench utility (quality-weighted novelty) vs T; (c) semantic spread (Vendi) vs T."""
    df, greedy = load_divx()
    models = sorted(df.model.unique())
    cell = df.groupby(["model", "rule", "T"])[["rel_reward", "nb_distinct", "utility", "vendi"]].mean().reset_index()
    # utility relative to the common default (top-p 0.95, T=1) of the same model, so models are comparable
    ref = cell[(cell.rule == "topp0.95") & (cell["T"] == 1.0)].set_index("model").utility
    cell["rel_utility"] = cell.utility - cell.model.map(ref)
    agg = cell.groupby(["rule", "T"])[["rel_reward", "nb_distinct", "rel_utility", "vendi"]].agg(["mean", "sem"])
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    ax = fig.add_subplot(gs[0])
    for rule in DIVX_RULES:
        if rule not in agg.index.get_level_values(0):
            continue
        a = agg.loc[rule].sort_index()
        ls = (0, (4, 2)) if rule in CLASSIC else ("-" if rule in MODERN else ":")
        ax.plot(a[("nb_distinct", "mean")], a[("rel_reward", "mean")], ls=ls, lw=1.3, color=COL2.get(rule, "#888"),
                marker="o", ms=2.5, label=LABEL2.get(rule, rule))
        for T, r in a.iterrows():
            if T == 2.0:
                ax.annotate(f"T={T:g}", (r[("nb_distinct", "mean")], r[("rel_reward", "mean")]), fontsize=5,
                            xytext=(3, 2), textcoords="offset points", color=COL2.get(rule, "#888"))
    ax.axhline(0, color="k", lw=1, ls=":")
    ins = ax.inset_axes([0.52, 0.5, 0.45, 0.45])  # zoom on the useful region: more diversity at small quality cost
    for rule in DIVX_RULES:
        if rule not in agg.index.get_level_values(0):
            continue
        a = agg.loc[rule].sort_index()
        ls = (0, (4, 2)) if rule in CLASSIC else ("-" if rule in MODERN else ":")
        ins.plot(a[("nb_distinct", "mean")], a[("rel_reward", "mean")], ls=ls, lw=1.0, color=COL2.get(rule, "#888"),
                 marker="o", ms=1.8)
    ins.set_xlim(2.6, 5.6)
    ins.set_ylim(-3.2, 0.2)
    ins.axhline(0, color="k", lw=0.8, ls=":")
    ins.tick_params(labelsize=5)
    ins.set_title("zoom", fontsize=5.5)
    ax.indicate_inset_zoom(ins, edgecolor="#888")
    ax.set_xlabel("distinct answers of 8 (NoveltyBench)")
    ax.set_ylabel("reward $-$ greedy reward")
    ax.set_title(f"(a) quality vs. diversity ({len(models)} models)", fontsize=7)
    ax.legend(fontsize=5.3, ncol=2, loc="lower left")
    for k, (col, lab) in enumerate((("rel_utility", "(b) utility vs. top-$p$ 0.95, $T{=}1$"), ("vendi", "(c) embedding diversity (Vendi)"))):
        ax = fig.add_subplot(gs[k + 1])
        for rule in DIVX_RULES:
            if rule not in agg.index.get_level_values(0):
                continue
            a = agg.loc[rule].sort_index()
            ls = (0, (4, 2)) if rule in CLASSIC else ("-" if rule in MODERN else ":")
            ax.errorbar(a.index, a[(col, "mean")], yerr=a[(col, "sem")], ls=ls, lw=1.2, color=COL2.get(rule, "#888"),
                        marker="o", ms=2.2, capsize=1)
        ax.set_xlabel("temperature $T$")
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([0.7, 1.0, 1.3, 1.6, 2.0])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    print(agg.round(3).to_string())


def bench_cells():
    """Per-benchmark cells (model, rule, T, acc) and greedy accuracy, from the analysis parquet files that exist."""
    out = {}
    f = f"{R}/lb2_gen.parquet"
    if os.path.exists(f):
        d = split_cfg(pd.read_parquet(f))
        out["LongBench v2\n(long input, $\\leq$120K)"] = d.rename(columns={"correct": "acc"})[["model", "rule", "T", "acc"]]
    f = f"{R}/longproc_gen.parquet"
    if os.path.exists(f):
        d = pd.read_parquet(f).rename(columns={"sampler": "rule", "score": "acc"})
        out["LongProc\n(long structured output)"] = d[["model", "rule", "T", "acc"]]
    for exp, lab in (("reason2", "AIME/HMMT 2025-26\n(32K reasoning)"), ("lcb", "LiveCodeBench v6\n(32K reasoning)")):
        f = f"{R}/{exp}_gen.parquet"
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d = d.rename(columns={"sampler": "rule", "correct": "acc", "pass": "acc"})
            out[lab] = d[["model", "rule", "T", "acc"]]
    f = f"{R}/lbwrite_gen.parquet"
    if os.path.exists(f):
        d = pd.read_parquet(f).rename(columns={"sampler": "rule"})
        d["acc"] = d.S / 100.0
        out["LongBench-Write\n(long-form writing)"] = d[["model", "rule", "T", "acc"]]
    f = f"{R}/wbench_gen.parquet"
    if os.path.exists(f):
        d = pd.read_parquet(f).rename(columns={"sampler": "rule"})
        d["acc"] = d.score / 10.0
        out["WritingBench\n(writing, critic model)"] = d[["model", "rule", "T", "acc"]]
    f = f"{R}/swe_inst.parquet"
    if os.path.exists(f):
        d = pd.read_parquet(f)
        d = d[(d.config.str.contains("@T") | (d.config == "greedy")) & d.resolved.notna()]
        d["rule"] = d.config.str.split("@T").str[0]
        d["T"] = d.config.str.split("@T").str[1].astype(float).fillna(0.0) if d.config.str.contains("@T").any() else 0.0
        d.loc[d.config == "greedy", ["rule", "T"]] = ["greedy", 0.0]
        out["SWE-bench Verified\n(agentic coding)"] = d.rename(columns={"resolved": "acc"})[["model", "rule", "T", "acc"]]
    return out


def fig_bench(name="fig_bench", rules=("temp", "topp0.95", "minp0.1", "pless", "nsigma1", "pless+k2", "nsigma1+k2")):
    """One row of panels, one per benchmark: accuracy minus greedy accuracy (mean over models) vs temperature."""
    B = bench_cells()
    ncol = 3 if len(B) > 4 else len(B)
    nrow = int(np.ceil(len(B) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.33 * ncol, 1.62 * nrow), sharey=False, squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(B):]:
        ax.axis("off")
    for ax, (lab, d) in zip(flat, B.items()):
        g = d[d.rule == "greedy"].groupby("model").acc.mean()
        c = d[d.rule != "greedy"].groupby(["model", "rule", "T"]).acc.mean().rename("acc").reset_index()
        c = c[c.model.isin(g.index)]
        c["rel"] = 100 * (c.acc - c.model.map(g))
        a = c.groupby(["rule", "T"]).rel.agg(["mean", "sem"]).reset_index()
        rb = [("topp0.9" if r == "topp0.95" and "topp0.95" not in set(a.rule) else r) for r in rules]
        for rule in rb:
            s = a[a.rule == rule].sort_values("T")
            if s.empty:
                continue
            ls = (0, (4, 2)) if rule in CLASSIC else ("-" if rule in MODERN else ":")
            ax.errorbar(s["T"], s["mean"], yerr=s["sem"], ls=ls, marker="o", ms=2.2, lw=1.2, capsize=1,
                        color=COL2.get(rule, "#888"), label=LABEL2.get(rule, rule))
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.set_title(f"{lab.replace(chr(10), ' ')} ({c.model.nunique()})", fontsize=6.6)
        ax.set_xlabel("temperature $T$")
        ax.set_xticks(sorted(c["T"].unique()))
    for r in range(nrow):
        axes[r][0].set_ylabel("score $-$ greedy (points)")
    handles, labels = flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")


SWE_ORDER = ["greedy", "temp@T0.7", "vendor", "temp@T1.0", "topp0.95@T1.0", "minp0.1@T1.0", "toph0.4@T1.0", "pless@T1.0",
             "nsigma1@T1.0", "topp0.95@T1.5", "minp0.1@T1.5", "nsigma1@T1.5"]
SWE_LABEL = {"greedy": "greedy", "temp@T0.7": "pure, $T{=}0.7$", "vendor": "vendor default", "temp@T1.0": "pure, $T{=}1$",
             "topp0.95@T1.0": "top-$p$ .95, $T{=}1$", "minp0.1@T1.0": "min-$p$ .1, $T{=}1$", "toph0.4@T1.0": "Top-H, $T{=}1$",
             "pless@T1.0": "P-less, $T{=}1$", "nsigma1@T1.0": "top-$n\\sigma$, $T{=}1$",
             "topp0.95@T1.5": "top-$p$ .95, $T{=}1.5$", "minp0.1@T1.5": "min-$p$ .1, $T{=}1.5$",
             "nsigma1@T1.5": "top-$n\\sigma$, $T{=}1.5$"}


def fig_swe(name="fig_swe"):
    """SWE-bench Verified with mini-swe-agent: per sampler configuration, stacked outcome shares
    (resolved / submitted but wrong / stopped by the step limit / format failures / other errors) per model."""
    d = pd.read_parquet(f"{R}/swe_inst.parquet")
    d = d[~d.think]
    models = [m for m in SHORT if m in set(d.model)]
    d["outcome"] = np.select(
        [d.resolved == True, (d.exit == "Submitted") & (d.patch_len > 0), d.exit == "LimitsExceeded",
         d.exit.fillna("").str.contains("Format")],
        ["resolved", "submitted, wrong", "step limit", "format failures"], "context / error")
    cats = ["resolved", "submitted, wrong", "step limit", "format failures", "context / error"]
    colors = ["#27ae60", "#95a5a6", "#c0392b", "#8e44ad", "#34495e"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.0), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        g = d[d.model == m]
        cfgs = [c for c in SWE_ORDER if c in set(g.config)]
        y = np.arange(len(cfgs))
        left = np.zeros(len(cfgs))
        for cat, col in zip(cats, colors):
            v = np.array([100 * (g[g.config == c].outcome == cat).mean() for c in cfgs])
            ax.barh(y, v, left=left, color=col, label=cat, height=0.75)
            left += v
        for k, c in enumerate(cfgs):
            r = 100 * (g[g.config == c].outcome == "resolved").mean()
            ax.text(r + 0.8, k, f"{r:.0f}", va="center", fontsize=5, color="white" if r > 8 else "k")
        ax.set_yticks(y)
        ax.set_yticklabels([SWE_LABEL.get(c, c) for c in cfgs], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of 500 issues")
        ax.set_title(SHORT[m], fontsize=7.5)
        ax.grid(axis="y", visible=False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")


def fig_hero(name="fig_hero", run=os.path.expanduser("~/lcg-cr/runs/core32k/qwen3.5-9b"),
             uid="fic_fantasy_0__topp0.9@T1.0__s0", smooth=256):
    """Figure 1: (a) a loop forms (one forced generation); (b) same model and context, different sampler (loop rescue,
    chat models); (c) across benchmarks, classic vs full-distribution-aware samplers as temperature rises."""
    import glob
    from .metrics import ngram_rep
    fig = plt.figure(figsize=(7.0, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.1], wspace=0.62)
    # (a)
    sub = gs[0].subgridspec(2, 1, hspace=0.12)
    toks = np.load(f"{run}/gen/{uid}.npz")["tokens"].astype(np.int64)
    L = len(toks)
    st = np.full((L, 7), np.nan)
    for f in sorted(glob.glob(f"{run}/stats/{uid}.p*.npz")):
        dd = np.load(f)
        pos, S = dd["pos"], dd["stats"]
        st[pos[pos < L], :S.shape[1]] = S[pos < L]
    k = np.ones(smooth) / smooth
    sm = lambda x: np.convolve(np.nan_to_num(x), k, mode="same")
    x = np.arange(L) / 1000
    a1 = fig.add_subplot(sub[0])
    a1.plot(x, sm(st[:, 0]), color="#7d3c98", lw=1.2, label="model entropy")
    a1.set_ylabel("entropy", fontsize=6.3)
    a1.tick_params(labelbottom=False, labelsize=5.5)
    a2 = fig.add_subplot(sub[1], sharex=a1)
    a2.plot(x, sm(ngram_rep(toks, 32).astype(float)), color="#c0392b", lw=1.2)
    a2.set_ylabel("repetition", fontsize=6.3)
    a2.set_xlabel("generated tokens (K)", fontsize=6.3)
    a2.tick_params(labelsize=5.5)
    hi = np.where(st[:, 6] > 0.5)[0]
    for ax in (a1, a2):
        if len(hi):
            ax.axvline(hi[0] / 1000, color="k", ls=":", lw=0.8)
    a1.set_title("(a) the model finishes, then a loop locks in", fontsize=7)
    if len(hi):
        a1.annotate("wants to end", (hi[0] / 1000, a1.get_ylim()[1] * 0.85), fontsize=5.5, xytext=(4, 0),
                    textcoords="offset points")
    # (b)
    ax = fig.add_subplot(gs[1])
    r = pd.read_parquet(f"{R}/rescue_gen.parquet")
    g = r.groupby(["exp", "sampler"]).agg(esc=("escaped", "mean"), rec=("recovered", "mean")).reset_index()
    pick = [("same", "original sampler"), ("pless+k2@T1.3", "P-less + keep-2"), ("nsigma1+k2@T1.3", "top-$n\\sigma$ + keep-2"),
            ("dry+topp0.9@T1.0", "DRY")]
    labs, v_book, v_chat = [], [], []
    for c, lab in pick:
        labs.append(lab)
        v_book.append(100 * g[(g.exp == "rescue") & (g.sampler == c)].esc.mean())
        v_chat.append(100 * g[(g.exp == "rescue_mt") & (g.sampler == c)].esc.mean())
    yy = np.arange(len(pick))
    ax.barh(yy - 0.2, v_book, 0.38, color="#7f8c8d", label="book continuation")
    ax.barh(yy + 0.2, v_chat, 0.38, color="#2874a6", label="multi-turn chat")
    ax.set_yticks(yy)
    ax.set_yticklabels(labs, fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of real loops escaped", fontsize=6.3)
    ax.set_title("(b) same model, same context,\nonly the sampler changes", fontsize=7)
    ax.legend(fontsize=5.5, loc="upper right")
    ax.grid(axis="y", visible=False)
    # (c)
    ax = fig.add_subplot(gs[2])
    B = bench_cells()
    fam = {"classic": CLASSIC, "full-distribution": MODERN}
    rows = []
    for lab, d in B.items():
        gr = d[d.rule == "greedy"].groupby("model").acc.mean()
        c = d[d.rule != "greedy"].groupby(["model", "rule", "T"]).acc.mean().rename("acc").reset_index()
        c = c[c.model.isin(gr.index)]
        if c.empty:
            continue
        Ts = sorted(c["T"].unique())
        lev = {Ts[0]: "low", Ts[len(Ts) // 2]: "T=1", Ts[-1]: "high"}
        c["lev"] = c["T"].map(lev)
        c["rel"] = 100 * (c.acc - c.model.map(gr))
        for f, rs in fam.items():
            q = c[c.rule.isin(rs)].groupby("lev").rel.mean()
            for lv, v in q.items():
                rows.append({"bench": lab, "fam": f, "lev": lv, "rel": v})
    q = pd.DataFrame(rows)
    order = ["low", "T=1", "high"]
    for f, col, ls in (("classic", "#c0392b", (0, (4, 2))), ("full-distribution", "#1f618d", "-")):
        z = q[q.fam == f].groupby("lev").rel.agg(["mean", "sem"]).reindex(order)
        ax.errorbar(range(3), z["mean"], yerr=z["sem"], color=col, ls=ls, lw=1.5, marker="o", ms=3, capsize=2,
                    label=("classic (pure, top-$k$, top-$p$)" if f == "classic" else "full-distribution (min-$p$, Top-H,\nP-less, top-$n\\sigma$)"))
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(3), ["low $T$\n(0.6-0.7)", "$T{=}1$", "high $T$\n(1.3-1.5)"], fontsize=6.3)
    ax.set_ylabel("score $-$ greedy (points)", fontsize=6.3)
    ax.set_title(f"(c) {q.bench.nunique()} long-context benchmarks", fontsize=7)
    ax.legend(fontsize=5.5, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")


def fig_divx_models(name="fig_divx_models"):
    """Appendix: the quality-diversity frontier of fig_divx (a) for every model separately."""
    df, greedy = load_divx()
    cell = df.groupby(["model", "rule", "T"])[["rel_reward", "nb_distinct"]].mean().reset_index()
    models = [m for m in SHORT if m in set(cell.model)]
    nc = 4
    nr = int(np.ceil(len(models) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(1.75 * nc, 1.65 * nr), squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        g = cell[cell.model == m]
        for rule in DIVX_RULES:
            a = g[g.rule == rule].sort_values("T")
            if a.empty:
                continue
            ls = (0, (4, 2)) if rule in CLASSIC else ("-" if rule in MODERN else ":")
            ax.plot(a.nb_distinct, a.rel_reward, ls=ls, lw=1.0, color=COL2.get(rule, "#888"), marker="o", ms=1.6,
                    label=LABEL2.get(rule, rule))
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_ylim(max(-30, g.rel_reward.min() - 1), 1.5)
        ax.set_title(f"{SHORT[m]} (greedy reward {greedy[m]:.1f})", fontsize=5.8)
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("reward $-$ greedy", fontsize=6)
    for ax in axes[-1, :]:
        ax.set_xlabel("distinct answers (of 8)", fontsize=6)
    axes.ravel()[0].legend(fontsize=4.5, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")

if __name__ == "__main__":
    for n in sys.argv[1:]:
        globals()[f"fig_{n}"]()
