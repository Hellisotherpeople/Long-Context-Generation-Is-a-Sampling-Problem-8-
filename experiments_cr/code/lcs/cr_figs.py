"""All camera-ready figures, generated from run files (no hand-entered numbers).
python -m lcs.cr_figs [hero pool map tl swe] -> ~/lcg-cr/figs/fig_*.pdf (+ .png previews)

Palette: validated reference categorical order (dataviz skill, light mode, all checks pass), mapped so the
tail-admitting rules take warm hues and the tail-excluding rules take cool/green hues. Every series also has a
distinct marker and a direct label, because three slots sit below 3:1 contrast on white.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from .cr_tables import load

H = os.path.expanduser("~/lcg-cr")
OUT = f"{H}/figs"
INK, INK2, GRID = "#1f1f1e", "#62615d", "#e6e5e0"
S = {  # key: (label, color, marker)
    "nsigma1": ("Top-$n\\sigma$", "#2a78d6", "o"),
    "topp0.9": ("Top-$p$", "#eb6834", "s"),
    "minp0.1": ("Min-$p$", "#1baf7a", "^"),
    "topk50": ("Top-$k$", "#eda100", "D"),
    "toph0.4": ("Top-H", "#e87ba4", "v"),
    "pless": ("P-less", "#008300", "P"),
    "temp": ("Pure sampling", "#9a9993", ""),
}
ORDER = ["topp0.9", "topk50", "toph0.4", "pless", "minp0.1", "nsigma1"]


def style():
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.6, "lines.markersize": 4,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=220)
    plt.close(fig); print("wrote", name)


def line(ax, x, y, key, label=True, lw=1.6, alpha=1.0, dx=0.0):
    lab, c, m = S[key]
    ax.plot(x, y, color=c, marker=m, lw=lw, ms=4 if m else 0, alpha=alpha, mec="white", mew=0.6,
            ls=":" if key == "temp" else "-", zorder=3, solid_capstyle="round")
    if label:
        ax.annotate(lab, (x[-1], y[-1]), xytext=(4 + dx, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK)


def spread_labels(ax, ends, min_gap):
    """Place right-edge direct labels without collisions. ends: list of (y, key)."""
    ends = sorted(ends)
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    shift = max(0, ys[-1] - max(e[0] for e in ends)) / 2 if ys else 0
    return [(y - shift, k) for y, (_, k) in zip(ys, ends)]


# ---------------------------------------------------------------- 1. hero
def fig_hero():
    """Judged quality by position in the text, per sampler, at T = 1.0 / 1.5 / 2.0 (R1, ten models)."""
    met, win, _ = load("r1_16k")
    win = win.merge(met[["model", "uid", "sampler", "T", "loop"]], on=["model", "uid"])
    win["pos"] = win.start.where(win.start < 15000, 14464) / 1000 + 0.75   # window centre in K tokens
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    for ax, T in zip(axes, (1.0, 1.5, 2.0)):
        ends = []
        for k in ORDER + ["temp"]:
            g = win[(win.sampler == k) & (win["T"] == T)]
            if g.empty: continue
            m = g.groupby("pos").j_overall.mean()
            line(ax, m.index.values, m.values, k, label=False)
            ends.append((m.values[-1], k))
        ax.set_title(f"$T = {T:g}$", color=INK)
        ax.set_xlim(0, 15.6); ax.set_ylim(0.8, 5.4); ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_xlabel("Position in generation (K tokens)")
    axes[0].set_ylabel("Judged window quality (1–10)")
    handles = [mpl.lines.Line2D([], [], color=S[k][1], marker=S[k][2], mec="white", lw=1.6, ls=":" if k == "temp" else "-",
                                label=S[k][0]) for k in ["nsigma1", "minp0.1", "pless", "toph0.4", "topk50", "topp0.9", "temp"]]
    fig.legend(handles=handles, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7.2, handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_hero")


# ---------------------------------------------------------------- 2. pool
def fig_pool():
    """Median realized candidate-set size vs temperature (R5 stats, first 2K tokens, four models) + R1 loop rate."""
    rows = []
    for f in glob.glob(f"{H}/runs/r5_hp/*/stats/*.npz"):
        u = os.path.basename(f)
        mm = re.match(r".*__(topp0\.9|topk50|toph0\.4|pless|minp0\.1|nsigma1\.0)@T([0-9.]+)__s\d+\.p0\.npz", u)
        if not mm: continue
        s = np.load(f)["stats"][:2048, 3]; s = s[np.isfinite(s)]
        if len(s): rows.append(dict(sampler=mm.group(1).replace("nsigma1.0", "nsigma1"), T=float(mm.group(2)), med=np.median(s), mean=s.mean()))
    d = pd.DataFrame(rows).groupby(["sampler", "T"]).agg(med=("med", "median"), mean=("mean", "median")).reset_index()
    d.to_csv(f"{H}/results/cr_pool_r5.csv", index=False)
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.45))
    ends = []
    for k in ORDER:
        g = d[d.sampler == k].sort_values("T")
        line(ax, g["T"].values, g["med"].values, k, label=False)
        ends.append((np.log10(g["med"].values[-1]), k))
    ax.set_yscale("log"); ax.set_ylim(0.8, 3e5)
    for y, k in spread_labels(ax, ends, 0.32):
        ax.text(3.08, 10 ** y, S[k][0], fontsize=6.8, va="center", color=S[k][1])
    ax.set_xlim(0.65, 3.0); ax.set_xticks([0.7, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel("Temperature"); ax.set_ylabel("Median tokens admitted per step")
    ax.grid(True, axis="y", which="major")
    save(fig, "fig_pool")


# ---------------------------------------------------------------- 3. map (R5 redo)
def fig_map():
    met, win, _ = load("r5_hp"); K = ["model", "uid"]
    met = met.join(win.groupby(K).j_overall.mean().rename("q"), on=K)
    q = met.groupby(["sampler", "T"]).q.mean()
    fams = [("Top-$p$", "topp", "topp0.9"), ("Top-$k$", "topk", "topk50"), ("Top-H", "toph", "toph0.4"),
            ("Min-$p$", "minp", "minp0.1"), ("Top-$n\\sigma$", "nsigma", "nsigma1")]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.85), sharey=True)
    for ax, (title, pre, key) in zip(axes, fams):
        names = sorted({s for s, _ in q.index if s.startswith(pre)}, key=lambda s: float(s[len(pre):]))
        base = mpl.colors.to_rgb(S[key][1])
        for i, s in enumerate(names):
            t = (i + 1) / len(names)
            col = tuple(1 - (1 - c) * (0.30 + 0.70 * t) for c in base)
            g = q.loc[s]
            ax.plot(g.index, g.values, color=col, lw=1.4, marker=S[key][2], ms=3, mec="white", mew=0.4, zorder=3)
            ax.lines[-1].set_label(s[len(pre):])
        tq = q.loc["temp"]; ax.plot(tq.index, tq.values, color="#9a9993", ls=":", lw=1, zorder=2)
        sym = {"topp": "p", "topk": "k", "toph": "\\alpha", "minp": "p", "nsigma": "n"}[pre]
        ax.text(0.0, 1.005, f"${sym}$ = " + ", ".join(n[len(pre):] for n in names),
                transform=ax.transAxes, fontsize=5.8, color=INK2, va="bottom")
        ax.set_title(title, color=S[key][1], pad=10); ax.set_xlim(0.6, 3.1); ax.set_xticks([1, 2, 3]); ax.set_xlabel("Temperature")
        ax.set_ylim(0.8, 5.2)
    axes[0].set_ylabel("Judged quality (1–10)")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_map")


# ---------------------------------------------------------------- 4. temperature-last ablation
def fig_tl():
    tl = load("r8_tl")[0:2]; met, win = tl; K = ["model", "uid"]
    met = met.join(win.groupby(K).j_overall.mean().rename("q"), on=K)
    m5, w5, _ = load("r5_hp"); m5 = m5.join(w5.groupby(K).j_overall.mean().rename("q"), on=K)
    m5 = m5[m5.model.isin(met.model.unique())]
    met["base"] = met.sampler.str.replace("_tl", "")
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.75), sharey=True)
    for ax, k in zip(axes, ["topp0.9", "topk50", "toph0.4", "minp0.1", "pless"]):
        a = m5[m5.sampler == k].groupby("T").q.mean().loc[[1.0, 1.3, 1.6, 2.0, 2.5, 3.0]]
        b = met[met.base == k].groupby("T").q.mean()
        b = pd.concat([pd.Series({1.0: a.loc[1.0]}), b])
        c = S[k][1]
        ax.plot(a.index, a.values, color=c, lw=1.4, ls="--", alpha=0.55, marker=S[k][2], ms=3, mfc="white", mec=c, mew=0.8)
        ax.plot(b.index, b.values, color=c, lw=1.8, marker=S[k][2], ms=3.5, mec="white", mew=0.5)
        ax.set_title(S[k][0], color=c); ax.set_xticks([1, 2, 3]); ax.set_xlim(0.85, 3.15); ax.set_ylim(0.8, 5.2)
        ax.set_xlabel("Temperature")
    axes[0].set_ylabel("Judged quality")
    axes[0].set_ylabel("Judged quality (1–10)")
    h = [mpl.lines.Line2D([], [], color=INK2, ls="--", lw=1.4, marker="o", mfc="white", ms=3, label="Temperature first (standard)"),
         mpl.lines.Line2D([], [], color=INK2, ls="-", lw=1.8, marker="o", mec="white", ms=3.5, label="Candidate set fixed at $T{=}1$, sample at $t$")]
    fig.legend(handles=h, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.1), fontsize=7.2)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_tl")


# ---------------------------------------------------------------- 5. agentic
def fig_swe():
    t = pd.DataFrame(json.load(open(f"{H}/results/cr_swe.json")))
    t = t[t.graded >= 0.9 * t.n]
    panels = [(e, m) for e, m in [("r3a_swe", "qwen3.5-9b"), ("r3b_swelong", "qwen3.5-9b"), ("r7_swe", "qwen3.5-27b"),
                                  ("r7_swe", "qwen3.5-35b-a3b"), ("r3c_swe", "gemma4-31b"), ("r7_swe", "gemma4-e4b")]
              if not t[(t.exp == e) & (t.model == m)].empty]
    title = {("r3a_swe", "qwen3.5-9b"): "Qwen3.5-9B\n100 tasks, 250 steps", ("r3b_swelong", "qwen3.5-9b"): "Qwen3.5-9B\n50 tasks, 500 steps",
             ("r7_swe", "qwen3.5-27b"): "Qwen3.5-27B\n50 tasks, 500 steps", ("r7_swe", "qwen3.5-35b-a3b"): "Qwen3.5-35B-A3B\n50 tasks, 500 steps",
             ("r3c_swe", "gemma4-31b"): "Gemma-4-31B\n50 tasks, 250 steps", ("r7_swe", "gemma4-e4b"): "Gemma-4-E4B\n50 tasks, 500 steps"}
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.1), sharey=True, squeeze=False)
    for ax, (e, m) in zip(axes[0], panels):
        g = t[(t.exp == e) & (t.model == m)]
        for k in ORDER:
            r = g[g.sampler == k].sort_values("T")
            if len(r): ax.plot(r["T"], 100 * r.resolved, color=S[k][1], marker=S[k][2], ms=4, mec="white", mew=0.5, lw=1.5, zorder=3)
        ax.set_title(title[(e, m)], fontsize=6.8, loc="center"); ax.set_xlabel("Temperature")
        ax.set_xticks([1.0, 1.5, 2.0]); ax.set_xlim(0.9, 2.1); ax.set_ylim(-3, 78)
    axes[0][0].set_ylabel("SWE-bench Verified resolved (%)")
    handles = [mpl.lines.Line2D([], [], color=S[k][1], marker=S[k][2], mec="white", lw=1.5, label=S[k][0]) for k in ORDER]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=7, handlelength=1.6, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.15, bottom=0.27)
    save(fig, "fig_swe")


def fig_bench():
    aw = pd.DataFrame(json.load(open(f"{H}/results/cr_arenawrite.json")))
    lb = pd.read_parquet(f"{H}/results/lbwrite_cells.parquet").groupby(["sampler", "T"]).S.mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
    for ax, (d, col, yl, ylim) in zip(axes, [(aw, "healthy", "Arena-Write: healthy generations", (-0.03, 0.85)),
                                              (lb, "S", "LongBench-Write score $S$", (0, 90))]):
        for k in ORDER + ["temp"]:
            g = d[d.sampler == k].sort_values("T"); g = g[g["T"] > 0]
            if len(g): line(ax, g["T"].values, g[col].values, k, label=False)
        ax.set_xticks([1.0, 1.5, 2.0]); ax.set_xlim(0.9, 2.1); ax.set_ylim(*ylim); ax.set_xlabel("Temperature"); ax.set_title(yl, fontsize=8)
    handles = [mpl.lines.Line2D([], [], color=S[k][1], marker=S[k][2], mec="white", lw=1.6, ls=":" if k == "temp" else "-", label=S[k][0])
               for k in ["nsigma1", "pless", "minp0.1", "toph0.4", "topk50", "topp0.9", "temp"]]
    fig.legend(handles=handles, ncol=1, loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=7)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_bench")


if __name__ == "__main__":
    style()
    want = sys.argv[1:] or ["hero", "pool", "map", "tl", "swe", "bench"]
    for w in want:
        try:
            globals()[f"fig_{w}"]()
        except Exception as ex:
            print("FAILED", w, repr(ex)[:300])
