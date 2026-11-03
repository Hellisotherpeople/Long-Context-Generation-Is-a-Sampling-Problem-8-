"""Second figure set: the cross-benchmark scoreboard, NIAH-by-length, LongProc, and appendix figures.
python -m lcs.figs_cr2 [names] -> ~/lcg-cr/figs/fig_*.pdf (+ .png). Same palette/markers as cr_figs.
Reads ~/lcg-cr/results (camera-ready waves) and ~/lcgen/results (October campaign, same sampler code).
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .cr_figs import style, save, S, ORDER, INK, INK2, GRID, line, H
from .cr_tables import load

LG = os.path.expanduser("~/lcgen/results")
R = f"{H}/results"
K = ["model", "uid"]
ALIAS = {"topp0.95": "topp0.9"}          # reasoning / LiveCodeBench use top-p 0.95; drawn as Top-p
LEG7 = ["nsigma1", "pless", "minp0.1", "toph0.4", "topk50", "topp0.9", "temp"]


def handles(keys, lw=1.6):
    return [mpl.lines.Line2D([], [], color=S[k][1], marker=S[k][2], mec="white", lw=lw,
                             ls=":" if k == "temp" else "-", label=S[k][0]) for k in keys]


def qcells(exp):
    met, win, _ = load(exp)
    return met.join(win.groupby(K).j_overall.mean().rename("q"), on=K), win


# ------------------------------------------------------------------ scoreboard
def scoreboard_rows():
    rows = []
    t = pd.DataFrame(json.load(open(f"{R}/cr_swe.json")))
    swe = [("r3a_swe", "qwen3.5-9b", "Qwen3.5-9B, 100 issues"), ("r7_swe", "qwen3.5-27b", "Qwen3.5-27B, 50 issues"),
           ("r7_swe", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B, 50 issues"), ("r18_swe_new100", "qwen3.5-9b", "Qwen3.5-9B, 100 fresh"),
           ("r20_swe_new100", "qwen3.5-27b", "Qwen3.5-27B, 100 fresh"), ("r20_swe_new100", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B, 100 fresh")]
    for e, m, lab in swe:
        g = t[(t.exp == e) & (t.model == m)].pivot_table(index="sampler", columns="T", values="resolved")
        rows.append(("Coding agents\n(SWE-bench Verified)", lab, 2.0, {s: (g.loc[s, 1.0], g.loc[s, 2.0]) for s in g.index if s in ORDER}))
    lcb = pd.read_parquet(f"{R}/lcb_cells.parquet")
    for m, lab in [("qwen3.8-27b", "LiveCodeBench v6, Qwen3.8-27B"), ("qwen3.6-27b", "LiveCodeBench v6, Qwen3.6-27B")]:
        g = lcb[lcb.model == m].pivot_table(index="sampler", columns="T", values="acc")
        rows.append(("Long reasoning", lab, 1.3, {ALIAS.get(s, s): (g.loc[s, 1.0], g.loc[s, 1.3]) for s in g.index if ALIAS.get(s, s) in ORDER}))
    for f, lab in [("reason2_cells", "AIME + HMMT 2025–26, 6 models"), ("reason_cells", "AIME 2025–26 + MATH-500 L5, 4 models")]:
        g = pd.read_parquet(f"{LG}/{f}.parquet").pivot_table(index="sampler", columns="T", values="acc", aggfunc="mean")
        rows.append(("Long reasoning", lab, 1.3, {ALIAS.get(s, s): (g.loc[s, 1.0], g.loc[s, 1.3]) for s in g.index
                                                 if ALIAS.get(s, s) in ORDER and not np.isnan(g.loc[s].get(1.3, np.nan))}))
    aw = pd.DataFrame(json.load(open(f"{R}/cr_arenawrite.json"))).set_index(["sampler", "T"]).healthy
    lb = pd.read_parquet(f"{R}/lbwrite_cells.parquet").groupby(["sampler", "T"]).S.mean()
    wb = pd.read_parquet(f"{R}/wbench_cells.parquet").groupby(["sampler", "T"]).score.mean()
    met, win = qcells("r1_16k")
    fw = win[win.start == 0].merge(met[K + ["sampler", "T"]], on=K).groupby(["sampler", "T"]).j_overall.mean()
    for ser, lab in [(aw, "Arena-Write, healthy texts, 9 models"), (lb, "LongBench-Write score, 8 models"),
                     (wb, "WritingBench critic, 6 models"), (fw, "Forced 16K fiction, first window, 10 models")]:
        rows.append(("Long writing", lab, 2.0, {s: (ser.loc[(s, 1.0)], ser.loc[(s, 2.0)]) for s in ORDER if (s, 2.0) in ser.index}))
    lp = pd.read_parquet(f"{LG}/longproc_cells.parquet").groupby(["sampler", "T"]).score.mean()
    rows.append(("Long writing", "LongProc procedural output, 6 models", 1.5, {s: (lp.loc[(s, 1.0)], lp.loc[(s, 1.5)]) for s in ORDER}))
    l2 = pd.read_parquet(f"{LG}/lb2_cells.parquet").groupby(["sampler", "T"]).acc.mean()
    ni = pd.read_parquet(f"{LG}/niah_gen.parquet")
    ni = ni[ni.L == 1000000].groupby(["sampler", "T"]).correct.mean()
    rows.append(("Long-context reading", "LongBench v2 (up to 120K), 6 models", 1.5, {s: (l2.loc[(s, 1.0)], l2.loc[(s, 1.5)]) for s in ORDER}))
    rows.append(("Long-context reading", "Needle retrieval at 1M tokens, 4 models", 1.5, {s: (ni.loc[(s, 1.0)], ni.loc[(s, 1.5)]) for s in ORDER}))
    return rows


def fig_scoreboard():
    rows = scoreboard_rows()
    out = [dict(group=g, row=l, T=T, **{s: round(100 * hi / lo, 1) for s, (lo, hi) in d.items()}) for g, l, T, d in rows]
    json.dump(out, open(f"{R}/cr_scoreboard.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    y, ys, groups, last = 0, [], [], None
    for g, lab, T, d in rows:
        if g != last:
            y += 0.7 if last else 0
            groups.append((g, y)); last = g
        ys.append(y); y += 1
    off = {k: (i - 2.5) * 0.14 for i, k in enumerate(["topp0.9", "topk50", "toph0.4", "minp0.1", "pless", "nsigma1"])}
    for (g, lab, T, d), yy in zip(rows, ys):
        ax.axhspan(yy - 0.45, yy + 0.45, color="#f6f5f1", zorder=0, lw=0)
        for s, (lo, hi) in d.items():
            r = 100 * hi / lo if lo > 0 else np.nan
            ax.plot(min(r, 148), yy + off[s], marker=S[s][2], color=S[s][1], ms=4.6, mec="white", mew=0.5, ls="", zorder=3)
            if r > 148: ax.annotate("", (152, yy + off[s]), (147, yy + off[s]), arrowprops=dict(arrowstyle="-|>", color=S[s][1], lw=0.8))
        ax.text(-3, yy, lab, ha="right", va="center", fontsize=7, color=INK)
        ax.text(156, yy, f"$T{{=}}{T:g}$", ha="left", va="center", fontsize=6.6, color=INK2)
    for g, yy in groups:
        ax.text(-3, yy - 0.75, g.replace("\n", " "), ha="right", va="center", fontsize=7.4, color=INK, weight="bold")
    ax.axvline(100, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(100, -1.55, "as good as at $T{=}1$", ha="center", fontsize=6.6, color=INK2)
    ax.set_xlim(0, 160); ax.set_xticks([0, 25, 50, 75, 100, 125, 150]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%", "125%", "150%"])
    ax.set_ylim(y - 0.4, -1.9); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.6); ax.grid(False, axis="y")
    ax.set_xlabel("Score at high temperature, as a share of the same sampler's score at $T{=}1$")
    order = ["nsigma1", "pless", "minp0.1", "toph0.4", "topk50", "topp0.9"]
    ax.legend(handles=[mpl.lines.Line2D([], [], color=S[k][1], marker=S[k][2], mec="white", ls="", ms=5.5, label=S[k][0]) for k in order],
              ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.4, handletextpad=0.2, columnspacing=1.4)
    save(fig, "fig_scoreboard")


# ------------------------------------------------------------------ NIAH by context length
def fig_niah():
    ni = pd.read_parquet(f"{LG}/niah_gen.parquet")
    ni = ni[ni.model != "qwen3.5-35b-a3b"]                     # the four models measured at every length
    d = ni.groupby(["sampler", "T", "L"]).correct.mean().reset_index()
    d.to_csv(f"{R}/cr_niah_bylen.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), sharey=True)
    for ax, T in zip(axes, (1.0, 1.5)):
        for k in ORDER + ["temp"]:
            g = d[(d.sampler == k) & (d["T"] == T)].sort_values("L")
            line(ax, g.L.values, 100 * g.correct.values, k, label=False)
        ax.set_xscale("log"); ax.set_xticks([8192, 32768, 131072, 262144, 524288, 1048576])
        ax.set_xticklabels(["8K", "32K", "128K", "256K", "512K", "1M"]); ax.minorticks_off()
        ax.set_title(f"$T = {T:g}$"); ax.set_xlabel("Context length (tokens)"); ax.set_ylim(40, 100)
    axes[0].set_ylabel("Needles retrieved (%)")
    fig.legend(handles=handles(LEG7), ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=7.2, handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_niah")


# ------------------------------------------------------------------ benchmarks, five panels
def fig_bench():
    aw = pd.DataFrame(json.load(open(f"{R}/cr_arenawrite.json")))
    lb = pd.read_parquet(f"{R}/lbwrite_cells.parquet").groupby(["sampler", "T"]).S.mean().reset_index()
    wb = pd.read_parquet(f"{R}/wbench_cells.parquet").groupby(["sampler", "T"]).score.mean().reset_index()
    lp = pd.read_parquet(f"{LG}/longproc_cells.parquet").groupby(["sampler", "T"]).score.mean().reset_index()
    l2 = pd.read_parquet(f"{LG}/lb2_cells.parquet").groupby(["sampler", "T"]).acc.mean().reset_index()
    lp["score"] *= 100; l2["acc"] *= 100; aw["healthy"] *= 100
    panels = [(aw, "healthy", "Arena-Write (9 models)\nhealthy texts (%)", (-3, 100)),
              (lb, "S", "LongBench-Write (8 models)\nofficial score", (0, 90)),
              (wb, "score", "WritingBench (6 models)\ncritic score (1–10)", (0, 8.5)),
              (lp, "score", "LongProc (6 models)\nofficial score (%)", (0, 35)),
              (l2, "acc", "LongBench v2 (6 models)\naccuracy (%)", (15, 42))]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.0))
    for ax, (d, col, yl, ylim) in zip(axes, panels):
        for k in ORDER + ["temp"]:
            g = d[(d.sampler == k) & (d["T"] >= 0.7)].sort_values("T")
            if len(g): line(ax, g["T"].values, g[col].values, k, label=False, lw=1.4)
        ax.set_xticks([1.0, 1.5, 2.0]); ax.set_xlim(0.6, 2.1); ax.set_ylim(*ylim); ax.set_xlabel("Temperature")
        ax.set_title(yl, fontsize=7.0, loc="center", weight="normal")
    fig.legend(handles=handles(LEG7), ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.14), fontsize=6.8, handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.36, bottom=0.27)
    save(fig, "fig_bench")


# ------------------------------------------------------------------ appendix: disputed issues
def fig_disputed():
    inst = pd.read_parquet(f"{R}/swe_inst.parquet")
    inst["sampler"] = inst.config.str.split("@").str[0]; inst["T"] = inst.config.str.split("@T").str[1].astype(float)
    cfg = [("r3a_swe", "qwen3.5-9b", "Qwen3.5-9B"), ("r7_swe", "qwen3.5-27b", "Qwen3.5-27B"), ("r7_swe", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B"),
           ("r18_swe_new100", "qwen3.5-9b", "Qwen3.5-9B, fresh"), ("r20_swe_new100", "qwen3.5-27b", "Qwen3.5-27B, fresh"),
           ("r20_swe_new100", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B, fresh")]
    col = {"both": "#9db8de", "ns": S["nsigma1"][1], "tp": S["topp0.9"][1], "none": "#e9e8e3"}
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.75))
    for ax, (e, m, lab) in zip(axes, cfg):
        g = inst[(inst.exp == e) & (inst.model == m) & (inst["T"] == 2.0)]
        a = g[g.sampler == "nsigma1"].set_index("instance").resolved.astype(bool)
        b = g[g.sampler == "topp0.9"].set_index("instance").resolved.astype(bool)
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(False)
        cats = sorted(np.where(j.a & j.b, "both", np.where(j.a, "ns", np.where(j.b, "tp", "none"))), key=["both", "ns", "tp", "none"].index)
        n = len(cats); w = 10
        for i, c in enumerate(cats):
            ax.add_patch(mpl.patches.FancyBboxPatch((i % w + 0.08, -(i // w) - 0.92), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.18",
                                                    color=col[c], lw=0))
        ax.set_xlim(0, w); ax.set_ylim(-10.2, 0.1); ax.set_aspect("equal"); ax.axis("off")
        ns, tp = int((j.a & ~j.b).sum()), int((j.b & ~j.a).sum())
        ax.set_title(f"{lab}\n{ns}–{tp} of {n}", fontsize=6.6, loc="center", weight="normal")
    lab = [("ns", "Solved only with top-$n\\sigma$"), ("tp", "Solved only with top-$p$ (none)"), ("none", "Solved by neither")]
    fig.legend(handles=[mpl.patches.Patch(color=col[c], label=l) for c, l in lab], ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=7)
    fig.subplots_adjust(wspace=0.12, bottom=0.12)
    save(fig, "fig_disputed")


# ------------------------------------------------------------------ appendix: how agents fail
def fig_swe_anatomy():
    inst = pd.read_parquet(f"{R}/swe_inst.parquet")
    inst = inst[inst.model.str.startswith("qwen3.5") & inst.exp.isin(["r3a_swe", "r7_swe"]) & ~inst.config.str.contains(r"\+")]
    inst["sampler"] = inst.config.str.split("@").str[0]; inst["T"] = inst.config.str.split("@T").str[1].astype(float)
    inst["submitted"] = inst["exit"] == "Submitted"
    d = inst.groupby(["sampler", "T"]).agg(fe=("format_errors", "mean"), sub=("submitted", "mean"), steps=("steps", "mean")).reset_index()
    d.to_csv(f"{R}/cr_swe_anatomy.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for k in ORDER + ["temp"]:
        g = d[d.sampler == k].sort_values("T")
        line(axes[0], g["T"].values, g.fe.values + 0.01, k, label=False)
        line(axes[1], g["T"].values, 100 * g["sub"].values, k, label=False)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Malformed actions per episode"); axes[0].set_ylim(0.01, 30)
    axes[0].set_yticks([0.01, 0.1, 1, 10]); axes[0].set_yticklabels(["0", "0.1", "1", "10"])
    axes[1].set_ylabel("Episodes that submit a patch (%)"); axes[1].set_ylim(-3, 103)
    for ax in axes: ax.set_xticks([1.0, 1.5, 2.0]); ax.set_xlim(0.9, 2.1); ax.set_xlabel("Temperature")
    axes[0].set_title("(a) Tail tokens break the command format", fontsize=8)
    axes[1].set_title("(b) …so the agent never finishes", fontsize=8)
    fig.legend(handles=handles(LEG7), ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=7, handlelength=1.6)
    fig.subplots_adjust(wspace=0.3, bottom=0.27)
    save(fig, "fig_swe_anatomy")


# ------------------------------------------------------------------ appendix: candidate-set distribution
def fig_pooldist():
    acc = {}
    for f in glob.glob(f"{H}/runs/r1_16k/*/stats/*.npz"):
        mm = re.match(r".*__(topp0\.9|topk50|toph0\.4|pless|minp0\.1|nsigma1)@T([0-9.]+)__s\d+\.p0\.npz", os.path.basename(f))
        if not mm: continue
        s = np.load(f)["stats"][:2048, 3]; s = s[np.isfinite(s)]
        acc.setdefault((mm.group(1), float(mm.group(2))), []).append(s)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharey=True)
    summ = []
    for ax, T in zip(axes, (1.0, 1.5, 2.0)):
        for k in ORDER:
            if (k, T) not in acc: continue
            x = np.sort(np.concatenate(acc[(k, T)]))
            u, c = np.unique(x, return_counts=True); cdf = np.cumsum(c) / len(x)
            ax.step(np.r_[1, u], 100 * np.r_[0, cdf], where="post", color=S[k][1], lw=1.5)
            summ.append(dict(sampler=k, T=T, p50=np.median(x), p90=np.percentile(x, 90), frac_le10=float((x <= 10).mean())))
        ax.set_xscale("log"); ax.set_xlim(0.8, 2e5); ax.set_title(f"$T = {T:g}$"); ax.set_xlabel("Tokens admitted at a step")
        ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5]); ax.set_xticklabels(["1", "10", "100", "1K", "10K", "100K"]); ax.minorticks_off()
        ax.grid(True, axis="x", color=GRID, lw=0.5)
    pd.DataFrame(summ).to_csv(f"{R}/cr_pooldist.csv", index=False)
    axes[0].set_ylabel("Share of decoding steps (%)")
    fig.legend(handles=handles(["nsigma1", "minp0.1", "pless", "toph0.4", "topk50", "topp0.9"]), ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.07), fontsize=7.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_pooldist")


# ------------------------------------------------------------------ appendix: second judge, second seed
def _glm(exp):
    rows = []
    for run in glob.glob(f"{H}/runs/{exp}/*"):
        f = f"{run}/judge_glm45air/window.jsonl"
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if isinstance(r.get("scores"), dict) and "overall" in r["scores"]:
                    rows.append(dict(model=os.path.basename(run), uid=r["uid"], start=r.get("start", -1), g=r["scores"]["overall"]))
    return pd.DataFrame(rows)


def fig_agree():
    met, win = qcells("r1_16k")
    w = win.merge(_glm("r1_16k"), on=K + ["start"]).merge(met[K + ["sampler", "T"]], on=K)
    c = w.groupby(["sampler", "T"]).agg(a=("j_overall", "mean"), b=("g", "mean")).reset_index()
    rho = spearmanr(c.a, c.b)[0]
    m9, _ = qcells("r9_16k_s2")
    s1 = met.groupby(["model", "sampler", "T"]).q.mean(); s2 = m9.groupby(["model", "sampler", "T"]).q.mean()
    ss = pd.concat([s1.rename("a"), s2.rename("b")], axis=1, join="inner").reset_index()
    rho2 = spearmanr(ss.a, ss.b)[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, d, xl, yl, title, r in [(axes[0], c, "gpt-oss-120b judge", "GLM-4.5-Air judge", "(a) Two judges, 22 sampler × temperature cells", rho),
                                     (axes[1], ss, "Seed 1", "Seed 2", "(b) Two seeds, model × sampler × temperature cells", rho2)]:
        lo, hi = 0.8, max(d.a.max(), d.b.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color=GRID, lw=1, zorder=1)
        for k in ORDER + ["temp", "greedy"]:
            g = d[d.sampler == k]
            if g.empty: continue
            col, mk = (S[k][1], S[k][2] or "o") if k in S else ("#3d3d3a", "x")
            ax.scatter(g.a, g.b, s=10 + 10 * g["T"], color=col, marker=mk, edgecolor="white", lw=0.4, zorder=3, alpha=0.9)
        ax.text(0.04, 0.93, f"Spearman $\\rho = {r:.2f}$", transform=ax.transAxes, fontsize=7.5, color=INK)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=8)
        ax.grid(True, axis="both", color=GRID, lw=0.5)
    fig.legend(handles=[mpl.lines.Line2D([], [], color=S[k][1], marker=S[k][2] or "o", mec="white", ls="", ms=5, label=S[k][0]) for k in LEG7], ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.subplots_adjust(wspace=0.28, bottom=0.22)
    json.dump(dict(rho_judge_cell=rho, rho_seed_cell=rho2, n_seed=len(ss)), open(f"{R}/cr_agree_fig.json", "w"))
    save(fig, "fig_agree")


# ------------------------------------------------------------------ appendix: position curves for other protocols
def hero_like(exp, temps, name, title_pre="", ylim=(0.8, 7.6), bin_k=2):
    met, win = qcells(exp)
    w = win.merge(met[K + ["sampler", "T"]], on=K)
    w["pos"] = (w.start // (bin_k * 1000)) * bin_k + bin_k / 2
    fig, axes = plt.subplots(1, len(temps), figsize=(7.0, 2.2), sharey=True, squeeze=False)
    for ax, T in zip(axes[0], temps):
        for k in ORDER + ["temp"]:
            g = w[(w.sampler == k) & (w["T"] == T)]
            if g.empty: continue
            m = g.groupby("pos").j_overall.mean()
            line(ax, m.index.values, m.values, k, label=False)
        ax.set_title(f"{title_pre}$T = {T:g}$"); ax.set_ylim(*ylim); ax.set_xlabel("Position in generation (K tokens)")
    axes[0][0].set_ylabel("Judged window quality (1–10)")
    fig.legend(handles=handles(LEG7), ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=7.2, handlelength=1.8)
    fig.subplots_adjust(wspace=0.08)
    save(fig, name)


def fig_natural():
    hero_like("r4_mt", (1.0, 1.5, 2.0), "fig_natural", bin_k=4)


def fig_nonfic():
    hero_like("r2_nonfic16k", (1.0, 1.5), "fig_nonfic", bin_k=4)


def fig_large():
    met, _ = qcells("r10_large")
    names = {"gemma4-31b": "Gemma-4-31B", "olmo3.1-32b": "OLMo-3.1-32B", "qwen3.5-27b": "Qwen3.5-27B",
             "nemotron3-30b-a3b": "Nemotron-3-Nano-30B-A3B", "glm4.5-air": "GLM-4.5-Air", "qwen3.5-122b-a10b": "Qwen3.5-122B-A10B"}
    ms = [m for m in names if m in set(met.model)]
    fig, axes = plt.subplots(1, len(ms), figsize=(7.0, 1.9), sharey=True)
    for ax, m in zip(axes, ms):
        g0 = met[met.model == m]
        for k in ORDER + ["temp"]:
            g = g0[g0.sampler == k].groupby("T").q.mean()
            if len(g): line(ax, g.index.values, g.values, k, label=False, lw=1.3)
        ax.set_title(names[m], fontsize=6.8, loc="center", weight="normal"); ax.set_xticks([1, 1.5, 2]); ax.set_xlim(0.9, 2.1)
        ax.set_xlabel("Temperature"); ax.set_ylim(0.8, 8)
    axes[0].set_ylabel("Judged quality (1–10)")
    fig.legend(handles=handles(LEG7), ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=7)
    fig.subplots_adjust(wspace=0.1, bottom=0.28)
    save(fig, "fig_large")


# ------------------------------------------------------------------ appendix: reasoning and code
def fig_reason():
    lcb = pd.read_parquet(f"{R}/lcb_cells.parquet")
    r2 = pd.read_parquet(f"{LG}/reason2_cells.parquet"); r1 = pd.read_parquet(f"{LG}/reason_cells.parquet")
    panels = [(lcb[lcb.model == "qwen3.8-27b"], "LiveCodeBench v6\nQwen3.8-27B"), (lcb[lcb.model == "qwen3.6-27b"], "LiveCodeBench v6\nQwen3.6-27B"),
              (r2, "AIME + HMMT 2025–26\n6 models"), (r1, "AIME 2025–26 + MATH-500 L5\n4 models")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    for ax, (d, title) in zip(axes, panels):
        d = d.assign(sampler=d.sampler.replace(ALIAS)).groupby(["sampler", "T"]).acc.mean().reset_index()
        for k in ["topp0.9", "toph0.4", "pless", "minp0.1", "nsigma1", "temp"]:
            g = d[(d.sampler == k) & (d["T"] > 0)].sort_values("T")
            if len(g): line(ax, g["T"].values, 100 * g.acc.values, k, label=False, lw=1.4)
        gr = d[d.sampler == "greedy"].acc
        if len(gr): ax.axhline(100 * gr.iloc[0], color=INK2, lw=0.7, ls=(0, (1, 1.5)))
        ax.set_title(title, fontsize=7, loc="center", weight="normal"); ax.set_xticks([0.6, 1.0, 1.3, 1.6]); ax.set_xlim(0.5, 1.7)
        ax.set_xlabel("Temperature")
    axes[0].set_ylabel("pass@1 / accuracy (%)")
    fig.legend(handles=handles(["nsigma1", "minp0.1", "pless", "toph0.4", "topp0.9", "temp"]) +
               [mpl.lines.Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.5)), label="Greedy")],
               ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=7)
    fig.subplots_adjust(wspace=0.32, bottom=0.28)
    save(fig, "fig_reason")


# ------------------------------------------------------------------ appendix: LongProc
def fig_longproc():
    lp = pd.read_parquet(f"{LG}/longproc_cells.parquet")
    tn = {"countdown_8k": "Countdown", "html_to_tsv_8k": "HTML to TSV", "path_traversal_8k": "Path traversal",
          "pseudo_to_code_2k": "Pseudocode to code", "tom_tracking_8k": "Theory-of-mind tracking", "travel_planning_8k": "Travel planning"}
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.95), sharey=False)
    for ax, (t, nm) in zip(axes, tn.items()):
        g0 = lp[lp.task == t].groupby(["sampler", "T"]).score.mean().reset_index()
        for k in ORDER + ["temp"]:
            g = g0[(g0.sampler == k) & (g0["T"] > 0)].sort_values("T")
            if len(g): line(ax, g["T"].values, 100 * g.score.values, k, label=False, lw=1.3)
        ax.set_title(nm, fontsize=6.8, loc="center", weight="normal"); ax.set_xticks([0.7, 1.0, 1.5]); ax.set_xlim(0.6, 1.6)
        ax.set_xlabel("Temperature"); ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Official score (%)")
    fig.legend(handles=handles(LEG7), ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=7)
    fig.subplots_adjust(wspace=0.38, bottom=0.28)
    save(fig, "fig_longproc")


def fig_lphp():
    h = pd.read_parquet(f"{LG}/longproc_hp_cells.parquet").groupby(["sampler", "T"]).score.mean()
    fams = [("Top-$p$", "topp", "topp0.9", "p"), ("Top-$k$", "topk", "topk50", "k"), ("Top-H", "toph", "toph0.4", "\\alpha"),
            ("$\\eta$-sampling", "eta", None, "\\eta"), ("$\\epsilon$-sampling", "eps", None, "\\epsilon"), ("Typical", "typical", None, "\\tau"),
            ("Min-$p$", "minp", "minp0.1", "p"), ("Top-$n\\sigma$", "nsigma", "nsigma1", "n")]
    fig, axes = plt.subplots(1, 8, figsize=(7.0, 1.85), sharey=True)
    for ax, (title, pre, key, sym) in zip(axes, fams):
        names = sorted({s for s, _ in h.index if re.fullmatch(pre + r"[0-9.]+", s)}, key=lambda s: float(s[len(pre):]))
        base = mpl.colors.to_rgb(S[key][1] if key else "#55534e"); mk = S[key][2] if key else "o"
        for i, s in enumerate(names):
            tt = (i + 1) / len(names); col = tuple(1 - (1 - c) * (0.30 + 0.70 * tt) for c in base)
            g = h.loc[s]; g = g[g.index > 0]
            ax.plot(g.index, 100 * g.values, color=col, lw=1.3, marker=mk, ms=3, mec="white", mew=0.4, zorder=3)
        ax.text(0, 1.01, f"${sym}$: " + ", ".join(n[len(pre):].lstrip("0") or "0" for n in names), transform=ax.transAxes, fontsize=4.6, color=INK2, va="bottom")
        ax.set_title(title, color=S[key][1] if key else INK, pad=9, fontsize=7.5); ax.set_xticks([1, 1.5, 2]); ax.set_xlim(0.85, 2.15)
        ax.set_xticklabels(["1", "1.5", "2"]); ax.set_xlabel("Temperature", fontsize=7)
    axes[0].set_ylabel("LongProc score (%)")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_lphp")


def fig_lptl():
    h = pd.read_parquet(f"{LG}/longproc_hp_cells.parquet").groupby(["sampler", "T"]).score.mean()
    panels = [("topp0.95", "Top-$p$ 0.95", "topp0.9"), ("minp0.05", "Min-$p$ 0.05", "minp0.1"), ("minp0.1", "Min-$p$ 0.1", "minp0.1"),
              ("minp0.2", "Min-$p$ 0.2", "minp0.1"), ("llamacpp", "llama.cpp default chain", None)]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.8), sharey=True)
    for ax, (s, title, key) in zip(axes, panels):
        c = S[key][1] if key else "#55534e"; mk = S[key][2] if key else "o"
        a = h.loc[s]; a = a[a.index > 0]; b = h.loc[s + "~tl"]
        ax.plot(a.index, 100 * a.values, color=c, lw=1.4, ls="--", alpha=0.6, marker=mk, ms=3, mfc="white", mec=c, mew=0.8)
        ax.plot(b.index, 100 * b.values, color=c, lw=1.8, marker=mk, ms=3.5, mec="white", mew=0.5)
        ax.set_title(title, color=c if key else INK, fontsize=7.6); ax.set_xticks([1, 1.5, 2]); ax.set_xlim(0.85, 2.15); ax.set_xlabel("Temperature")
    axes[0].set_ylabel("LongProc score (%)")
    hh = [mpl.lines.Line2D([], [], color=INK2, ls="--", lw=1.4, marker="o", mfc="white", ms=3, label="Temperature first (standard)"),
          mpl.lines.Line2D([], [], color=INK2, ls="-", lw=1.8, marker="o", mec="white", ms=3.5, label="Candidate set fixed at $T{=}1$, sample at $t$")]
    fig.legend(handles=hh, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=7.2)
    fig.subplots_adjust(wspace=0.1)
    save(fig, "fig_lptl")


# ------------------------------------------------------------------ the candidate set predicts quality (R5, every setting)
def fig_predict():
    met, win = qcells("r5_hp")
    rows = []
    for f in glob.glob(f"{H}/runs/r5_hp/*/stats/*.npz"):
        b = os.path.basename(f)
        mm = re.match(r"(.*__([a-z]+[0-9.]*)@T([0-9.]+)__s\d+)\.p0\.npz", b)
        if not mm: continue
        s = np.load(f)["stats"][:2048, 3]; s = s[np.isfinite(s)]
        if len(s): rows.append(dict(model=f.split("/")[-3], uid=mm.group(1), med=np.median(s), big=float(np.log10(s).mean())))
    st = pd.DataFrame(rows)
    st["uid"] = st.uid.str.replace("nsigma1.0@", "nsigma1@").str.replace(r"nsigma(\d)\.0@", r"nsigma\1@", regex=True)
    d = met.merge(st, on=K, how="inner")
    c = d.groupby(["sampler", "T"]).agg(q=("q", "mean"), med=("med", "median"), big=("big", "mean"), n=("q", "size")).reset_index()
    c["gm"] = 10 ** c.big
    c = c[c.n >= 20]
    c.to_csv(f"{R}/cr_predict.csv", index=False)
    rho = spearmanr(c.big, c.q)[0]
    fam = {"topp": "topp0.9", "topk": "topk50", "toph": "toph0.4", "minp": "minp0.1", "nsigma": "nsigma1", "pless": "pless", "temp": "temp"}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for pre, key in fam.items():
        g = c[c.sampler.str.match(pre + r"[0-9.]*$")]
        col, mk = S[key][1], (S[key][2] or "o")
        ax.scatter(g.gm, g.q, s=8 + 9 * g["T"], color=col, marker=mk, edgecolor="white", lw=0.4, zorder=3, alpha=0.92)
    ax.set_xscale("log"); ax.set_xlim(0.8, 1.5e5); ax.set_ylim(0.8, 5.2)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5]); ax.set_xticklabels(["1", "10", "100", "1K", "10K", "100K"]); ax.minorticks_off()
    ax.set_xlabel("Typical candidate set (geometric mean, tokens per step)"); ax.set_ylabel("Judged quality (1–10)")
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.text(0.03, 0.06, f"Spearman $\\rho = {rho:.2f}$\n{len(c)} sampler settings × temperatures", transform=ax.transAxes, fontsize=6.6, color=INK)
    ax.legend(handles=[mpl.lines.Line2D([], [], color=S[k][1], marker=S[k][2] or "o", mec="white", ls="", ms=4.5, label=S[k][0]) for k in ["nsigma1", "minp0.1", "pless", "toph0.4", "topk50", "topp0.9", "temp"]], fontsize=6.2, loc="upper right",
              handletextpad=0.1, borderaxespad=0.2, labelspacing=0.25)
    json.dump(dict(rho=rho, n=len(c)), open(f"{R}/cr_predict.json", "w"))
    save(fig, "fig_predict")


if __name__ == "__main__":
    style()
    want = sys.argv[1:] or ["scoreboard", "niah", "bench", "disputed", "swe_anatomy", "pooldist", "agree", "natural", "nonfic",
                            "large", "reason", "longproc", "lphp", "lptl", "predict"]
    for w in want:
        try:
            globals()[f"fig_{w}"]()
        except Exception as ex:
            import traceback; traceback.print_exc(); print("FAILED", w, repr(ex)[:300])
