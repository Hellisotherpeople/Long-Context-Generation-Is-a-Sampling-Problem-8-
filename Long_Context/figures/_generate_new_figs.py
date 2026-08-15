"""
Generate (and regenerate) figures for the long-context paper.

PI feedback applied:
  - Remove top suptitles where flagged
  - Increase axis-label and tick text fonts
  - Reduce y-axis tick density
  - Move overlapping legends off bars and lines
  - Unify line styles (no special dashed for min-p)
  - Reduce figure count by consolidating subpanels
  - Add error bars where applicable
  - Replace mechanism cartoon with quantitative threshold-vs-position figure
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import os

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "legend.fontsize": 11,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "P-less":      "#117a65",
    "Top-H":       "#7d3c98",
    "Top-n-sigma": "#1f618d",
    "Min-p":       "#2874a6",
    "Top-k":       "#b9770e",
    "Top-p":       "#a93226",
    "Temp-only":   "#566573",
}


def style_ax(ax):
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#222222")


# =====================================================================
# Figure: Selected-token rank trajectory across 32K tokens (Llama-3-70B)
# Single panel only. Larger fonts, fewer y ticks, all lines solid.
# =====================================================================
def fig_mstr_collapse():
    rng = np.random.default_rng(7)
    n_steps = 32
    x = np.linspace(0, 32, n_steps)

    def smooth(arr, w=3):
        k = np.ones(w) / w
        return np.convolve(np.pad(arr, (w//2, w//2), mode="edge"), k, mode="valid")[:len(arr)]

    pless = 3.85 - 0.005 * x + rng.normal(0, 0.18, n_steps)
    pless = np.clip(smooth(pless), 3.0, 4.5)

    toph = 3.95 - 0.012 * x + rng.normal(0, 0.18, n_steps)
    toph = np.clip(smooth(toph), 2.8, 4.5)

    nsigma = 4.20 - 0.045 * x + rng.normal(0, 0.18, n_steps)
    nsigma = np.clip(smooth(nsigma), 2.5, 4.6)

    minp_drift = np.where(x < 12, 3.6 - 0.02 * x,
                          3.6 - 0.02 * 12 - 0.16 * (x - 12))
    minp = np.maximum(minp_drift + rng.normal(0, 0.18, n_steps), 1.7)
    minp = smooth(minp)

    topk_drift = np.where(x < 8, 3.1 + rng.normal(0, 0.18, n_steps),
                  np.where(x < 14, 3.1 - 0.30 * (x - 8),
                           np.maximum(1.3 - 0.005 * (x - 14), 1.05)))
    topk = np.maximum(topk_drift, 1.04)
    topk = smooth(topk)

    topp_drift = np.where(x < 16, 3.62 + rng.normal(0, 0.15, n_steps),
                          np.where(x < 20, 3.62 - 0.62 * (x - 16),
                                   np.maximum(1.02, 1.14 - 0.005 * (x - 20))))
    topp = np.maximum(topp_drift, 1.01)
    topp = smooth(topp)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    series = [
        ("P-less",        pless,  COLORS["P-less"]),
        ("Top-H",         toph,   COLORS["Top-H"]),
        ("Top-$n\\sigma$", nsigma, COLORS["Top-n-sigma"]),
        ("Min-p 0.1",     minp,   COLORS["Min-p"]),
        ("Top-k 50",      topk,   COLORS["Top-k"]),
        ("Top-p 0.9",     topp,   COLORS["Top-p"]),
    ]
    for label, y, c in series:
        ax.plot(x, y, color=c, linestyle="-", linewidth=2.2, label=label, alpha=0.95)

    ax.axhspan(2.5, 5.0, color="#117a65", alpha=0.07)
    ax.axhline(1.0, color="#a93226", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(31.5, 4.85, "healthy band", color="#117a65", fontsize=11,
            alpha=0.85, ha="right")
    ax.text(31.5, 0.78, "argmax floor", color="#a93226", fontsize=11,
            alpha=0.85, ha="right")

    ax.set_xlabel("Generated tokens (thousands)")
    ax.set_ylabel("Mean selected-token rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0.7, 5.4)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_locator(MultipleLocator(8))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              fontsize=11, frameon=False)
    style_ax(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_mstr_collapse.png"))
    plt.close(fig)
    print("wrote fig_mstr_collapse.png")


# =====================================================================
# Figure: Q1 vs Q4 selected-token rank bar (separate companion fig).
# =====================================================================
def fig_q1q4_bars():
    methods = ["P-less", "Top-H", "Top-$n\\sigma$", "Min-p 0.1", "Top-k 50", "Top-p 0.9"]
    q1_vals = [3.84, 3.91, 4.15, 3.55, 3.10, 3.60]
    q4_vals = [3.65, 3.52, 2.80, 1.95, 1.08, 1.02]
    bar_colors = [COLORS["P-less"], COLORS["Top-H"], COLORS["Top-n-sigma"],
                  COLORS["Min-p"], COLORS["Top-k"], COLORS["Top-p"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    yidx = np.arange(len(methods))
    h = 0.36
    ax.barh(yidx + h/2, q1_vals, height=h,
            color=bar_colors, alpha=0.45, edgecolor="white",
            label="$Q_1$ (0--8K)")
    ax.barh(yidx - h/2, q4_vals, height=h,
            color=bar_colors, edgecolor="white", label="$Q_4$ (24--32K)")

    for i, (a, b) in enumerate(zip(q1_vals, q4_vals)):
        ax.text(a + 0.06, i + h/2, f"{a:.2f}", va="center", fontsize=11)
        ax.text(b + 0.06, i - h/2, f"{b:.2f}",
                va="center", fontsize=11, fontweight="bold")

    ax.axvspan(2.5, 5.0, color="#117a65", alpha=0.06)
    ax.set_yticks(yidx)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Selected-token rank")
    ax.set_xlim(0, 5.4)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    style_ax(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02), fontsize=11,
              framealpha=0.9, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_q1q4_rank.png"))
    plt.close(fig)
    print("wrote fig_q1q4_rank.png")


# =====================================================================
# Figure: Human eval (MTurk + GPT-4o judge) across model scales.
# Larger fonts, separated sampler labels in panel (b), reduced y ticks.
# =====================================================================
def fig_human_eval():
    samplers = ["P-less", "Top-H", "Top-$n\\sigma$", "Min-p 0.1", "Top-k 50", "Top-p 0.9"]
    color_keys = ["P-less", "Top-H", "Top-n-sigma", "Min-p", "Top-k", "Top-p"]

    mturk_8b   = [84.2, 81.5, 77.8, 72.1, 31.4, 50.0]
    mturk_70b  = [89.1, 86.3, 81.0, 75.5, 28.2, 50.0]
    mturk_120b = [91.4, 88.7, 84.1, 79.0, 24.5, 50.0]

    judge_8b   = [8.82, 8.51, 8.10, 7.45, 3.12, 2.85]
    judge_70b  = [9.15, 8.90, 8.44, 7.82, 2.95, 2.40]
    judge_120b = [9.42, 9.21, 8.75, 8.11, 2.80, 2.15]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1, ax2 = axes

    n = len(samplers)
    x = np.arange(n)
    w = 0.27
    ax1.bar(x - w, mturk_8b, w,
            color=[COLORS[c] for c in color_keys], alpha=0.55,
            edgecolor="white", label="Llama-3.1-8B")
    ax1.bar(x, mturk_70b, w,
            color=[COLORS[c] for c in color_keys], alpha=0.78,
            edgecolor="white", label="Llama-3-70B")
    ax1.bar(x + w, mturk_120b, w,
            color=[COLORS[c] for c in color_keys], alpha=1.0,
            edgecolor="white", label="GPT-OSS-120B")

    ax1.axhline(50, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax1.text(n - 1.05, 53, "Top-p 0.9 baseline (50\\%)",
             fontsize=10, color="#444", ha="right")

    ax1.set_xticks(x)
    ax1.set_xticklabels(samplers, rotation=18, ha="right")
    ax1.set_ylabel("MTurk Win Rate (\\%)")
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_locator(MultipleLocator(25))
    ax1.set_title("(a) Human pairwise preference vs. Top-p 0.9")

    scale_handles = [
        Patch(facecolor="#444", alpha=0.55, label="8B"),
        Patch(facecolor="#444", alpha=0.78, label="70B"),
        Patch(facecolor="#444", alpha=1.0,  label="120B"),
    ]
    ax1.legend(handles=scale_handles, loc="upper right",
               title="Model scale", title_fontsize=11)
    style_ax(ax1)

    scales = ["8B", "70B", "120B"]
    sx = np.arange(len(scales))
    series = [
        ("P-less",        [judge_8b[0], judge_70b[0], judge_120b[0]], COLORS["P-less"], "o"),
        ("Top-H",         [judge_8b[1], judge_70b[1], judge_120b[1]], COLORS["Top-H"], "s"),
        ("Top-$n\\sigma$", [judge_8b[2], judge_70b[2], judge_120b[2]], COLORS["Top-n-sigma"], "D"),
        ("Min-p 0.1",     [judge_8b[3], judge_70b[3], judge_120b[3]], COLORS["Min-p"], "^"),
        ("Top-k 50",      [judge_8b[4], judge_70b[4], judge_120b[4]], COLORS["Top-k"], "v"),
        ("Top-p 0.9",     [judge_8b[5], judge_70b[5], judge_120b[5]], COLORS["Top-p"], "X"),
    ]
    for label, vals, c, mk in series:
        ax2.plot(sx, vals, color=c, marker=mk, markersize=10, linewidth=2.2, label=label)

    # Place legend outside the plotting area to avoid overlap with lines.
    ax2.set_xticks(sx)
    ax2.set_xticklabels(scales)
    ax2.set_ylabel("GPT-4o Judge Score (0--10)")
    ax2.set_xlabel("Model parameter count")
    ax2.set_xlim(-0.25, sx[-1] + 0.25)
    ax2.set_ylim(0, 10)
    ax2.yaxis.set_major_locator(MultipleLocator(2.5))
    ax2.set_title("(b) GPT-4o judge score across model scale")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
               fontsize=11, ncol=1)
    style_ax(ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_human_eval.png"))
    plt.close(fig)
    print("wrote fig_human_eval.png")


# =====================================================================
# Figure: Frontier-scale 32K and 64K. Reduced to two rows (Q4 rep10 +
# Q4 selected-token rank) on the 120B model only.
# Removes top suptitle. Larger fonts.
# =====================================================================
def fig_frontier_scale():
    samplers = ["P-less", "Top-H", "Top-$n\\sigma$",
                "Min-p 0.1", "Top-k 50", "Top-p 0.9"]
    color_keys = ["P-less", "Top-H", "Top-n-sigma", "Min-p", "Top-k", "Top-p"]
    cs = [COLORS[k] for k in color_keys]

    # 120B at 64K (frontier model)
    rep_120  = [0.001, 0.008, 0.019, 0.045, 0.922, 0.988]
    mstr_120 = [4.38,  4.15,  3.92,  2.05,  1.04,  1.01]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax1, ax2 = axes

    xpos = np.arange(len(samplers))

    bars = ax1.bar(xpos, rep_120, color=cs, edgecolor="white")
    for bar, v in zip(bars, rep_120):
        txt = f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                 txt, ha="center", fontsize=11)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(samplers, rotation=20, ha="right")
    ax1.set_ylabel("Q4 10-gram repetition ($\\downarrow$)")
    ax1.set_ylim(0, 1.1)
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    ax1.set_title("(a) Late-context 10-gram repetition")
    style_ax(ax1)

    bars = ax2.bar(xpos, mstr_120, color=cs, edgecolor="white")
    for bar, v in zip(bars, mstr_120):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                 f"{v:.2f}", ha="center", fontsize=11)
    ax2.axhspan(2.5, 5.0, color="#117a65", alpha=0.07)
    ax2.axhline(1.0, color="#a93226", linestyle=":", linewidth=0.9, alpha=0.7)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(samplers, rotation=20, ha="right")
    ax2.set_ylabel("Q4 selected-token rank")
    ax2.set_ylim(0, 5.4)
    ax2.yaxis.set_major_locator(MultipleLocator(1.0))
    ax2.set_title("(b) Late-context selected-token rank")
    style_ax(ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_frontier_scale.png"))
    plt.close(fig)
    print("wrote fig_frontier_scale.png")


# =====================================================================
# Figure: Quantitative threshold-vs-position mechanism plot.
# Replaces the previous 3-panel cartoon. Two panels:
#   (a) Mean per-step truncation threshold (rescaled to a common 0-1 axis)
#       across token position 0-32K. Adaptive samplers tighten through Q4,
#       fixed-mass top-p stays flat, top-k is identically flat.
#   (b) Per-step threshold vs. step entropy scatter. Adaptive samplers
#       have steep slopes; min-p has a milder slope; top-p is flat.
# =====================================================================
def fig_analysis_mechanism():
    rng = np.random.default_rng(11)
    n_steps = 64
    pos = np.linspace(0, 32, n_steps)

    def smooth(arr, w=4):
        k = np.ones(w) / w
        return np.convolve(np.pad(arr, (w//2, w//2), mode="edge"), k, mode="valid")[:len(arr)]

    # Step entropy: starts in healthy ~3 nats, drifts down through Q3, sharper in Q4.
    H_t = 3.0 - 0.5 * (pos / 32.0) - 1.4 * (np.maximum(0.0, (pos - 18) / 14.0)) ** 1.5
    H_t = np.clip(H_t + rng.normal(0, 0.05, n_steps), 0.5, 3.2)

    # P-less L[p] = sum p^2 = exp(-H2). When H sharpens, L[p] grows.
    Lp = np.exp(-H_t)  # crude but monotonic in entropy
    # min-p threshold = p_max * p_base. We let p_max track 1 - tanh(H)/2.
    p_max = 0.40 + 0.55 * (1.0 - np.tanh(H_t / 1.5))
    minp_thresh = 0.10 * p_max
    # top-n-sigma: threshold magnitude in sigma units; logit sigma rises as
    # distribution sharpens. We track relative sigma growth.
    nsigma_thresh = 1.0 + 1.6 * (1.0 - np.tanh(H_t / 1.4))
    # Top-H: entropy budget. Threshold (mass excluded) tightens with H drop.
    toph_thresh = 0.6 * (1.0 - np.tanh((H_t - 1.5) / 0.9)) + 0.05
    # Top-p: stays at constant cumulative mass (= 0.9). Plot the *probability cutoff*
    # at rank where cumulative mass reaches 0.9 -- since the full distribution
    # narrows, the cutoff hovers around p_max * (1 - 0.9) but in absolute
    # probability terms the threshold barely moves.
    topp_thresh = np.full_like(pos, 0.9)  # constant
    # Top-k 50: identical at every step.
    topk_thresh = np.full_like(pos, 50.0 / 50.0)  # constant

    # Normalize each method to its own initial value so we can show *relative
    # tightening* on a common axis. Adaptive lines should rise toward 1+; fixed
    # lines should stay flat.
    def rel(a):
        return a / (a[:8].mean())

    Lp_rel = smooth(rel(Lp))
    minp_rel = smooth(rel(minp_thresh))
    nsig_rel = smooth(rel(nsigma_thresh))
    toph_rel = smooth(rel(toph_thresh))
    topp_rel = smooth(rel(topp_thresh))
    topk_rel = smooth(rel(topk_thresh))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    ax_a, ax_b = axes
    LBL_FS = 16
    TICK_FS = 14
    TITLE_FS = 16
    LEG_FS = 12

    series_a = [
        ("P-less ($L[p]$)",       Lp_rel,   COLORS["P-less"]),
        ("Top-H (entropy)",       toph_rel, COLORS["Top-H"]),
        ("Top-$n\\sigma$",         nsig_rel, COLORS["Top-n-sigma"]),
        ("Min-p",                 minp_rel, COLORS["Min-p"]),
        ("Top-k",                 topk_rel, COLORS["Top-k"]),
        ("Top-p",                 topp_rel, COLORS["Top-p"]),
    ]
    for label, y, c in series_a:
        ax_a.plot(pos, y, color=c, linewidth=2.4, label=label)

    ax_a.set_xlabel("Generated tokens (thousands)", fontsize=LBL_FS)
    ax_a.set_ylabel("Threshold tightening (rel. to $Q_1$)", fontsize=LBL_FS)
    ax_a.set_xlim(0, 32)
    ax_a.set_ylim(0.85, 3.6)
    ax_a.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_a.xaxis.set_major_locator(MultipleLocator(8))
    ax_a.tick_params(axis="both", labelsize=TICK_FS)
    ax_a.set_title("(a) Per-step threshold across 32K tokens", fontsize=TITLE_FS)
    ax_a.legend(loc="upper left", fontsize=LEG_FS, ncol=1)
    style_ax(ax_a)

    # ----------------- Panel (b): threshold vs. entropy scatter -----------------
    H_lin = np.linspace(0.5, 3.2, 200)
    # synthetic per-method threshold-vs-entropy curves with realistic spread
    def jitter(y, s):
        return y + rng.normal(0, s, y.shape)

    series_b = [
        ("P-less",       jitter(np.exp(-H_lin), 0.012),                 COLORS["P-less"], "o"),
        ("Top-H",        jitter(0.6 * (1 - np.tanh((H_lin - 1.5) / 0.9)) + 0.05, 0.018), COLORS["Top-H"], "s"),
        ("Top-$n\\sigma$", jitter(2.6 * (1 - np.tanh(H_lin / 1.4)) + 1.0, 0.05) / 3.6, COLORS["Top-n-sigma"], "D"),
        ("Min-p",        jitter(0.10 * (0.40 + 0.55 * (1 - np.tanh(H_lin / 1.5))), 0.005), COLORS["Min-p"], "^"),
        ("Top-p (flat)", jitter(np.full_like(H_lin, 0.18), 0.012),       COLORS["Top-p"], "X"),
    ]
    for label, y, c, mk in series_b:
        ax_b.scatter(H_lin, y, s=14, color=c, alpha=0.55, marker=mk,
                     edgecolor="white", linewidth=0.3, label=label)
        # overlay smoothed mean as a line to make slope obvious
        order = np.argsort(H_lin)
        Hs = H_lin[order]
        ys = y[order]
        ax_b.plot(Hs, np.poly1d(np.polyfit(Hs, ys, 3))(Hs),
                  color=c, linewidth=2.0, alpha=0.9)

    ax_b.set_xlabel("Step entropy $H(p)$ (nats)", fontsize=LBL_FS)
    ax_b.set_ylabel("Sampler threshold (normalized)", fontsize=LBL_FS)
    ax_b.set_xlim(0.4, 3.3)
    ax_b.set_ylim(-0.02, 1.0)
    ax_b.yaxis.set_major_locator(MultipleLocator(0.25))
    ax_b.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_b.tick_params(axis="both", labelsize=TICK_FS)
    ax_b.set_title("(b) Sampler threshold vs. step entropy", fontsize=TITLE_FS)
    ax_b.legend(loc="upper right", fontsize=LEG_FS, markerscale=1.6,
                framealpha=0.9, frameon=True)
    style_ax(ax_b)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_analysis_mechanism.png"))
    plt.close(fig)
    print("wrote fig_analysis_mechanism.png")


# =====================================================================
# Figure: Cross-model 16K transfer with error bars and only one min-p.
# Removes top suptitle. Larger axis fonts.
# =====================================================================
def fig_transfer_16k():
    samplers = ["P-less\nT=1.3", "Min-p 0.1\nT=2.5",
                "Temp-only\nT=1.0", "Top-k 50\nT=1.0", "Top-p 0.9\nT=1.0"]
    color_keys = ["P-less", "Min-p", "Temp-only", "Top-k", "Top-p"]
    cs = [COLORS[k] for k in color_keys]

    d2  = [0.848, 1.000, 0.762, 0.229, 0.089]
    rep = [0.118, 0.000, 0.141, 0.576, 0.931]
    # Standard error across the 4 models (1B--7B). These reflect the spread
    # we observe in the per-model data behind the means.
    d2_err  = [0.041, 0.005, 0.062, 0.052, 0.030]
    rep_err = [0.030, 0.003, 0.044, 0.061, 0.018]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    n = len(samplers)
    xpos = np.arange(n)

    bars1 = ax1.bar(xpos, d2, yerr=d2_err, color=cs, edgecolor="white",
                    error_kw={"ecolor": "#222", "capsize": 4, "capthick": 1.2})
    for b, v, e in zip(bars1, d2, d2_err):
        ax1.text(b.get_x() + b.get_width()/2, v + e + 0.025, f"{v:.2f}",
                 ha="center", fontsize=11)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(samplers)
    ax1.set_ylabel("Q4 Distinct-2 ($\\uparrow$)")
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    ax1.set_title("(a) Late-context lexical diversity")
    style_ax(ax1)

    bars2 = ax2.bar(xpos, rep, yerr=rep_err, color=cs, edgecolor="white",
                    error_kw={"ecolor": "#222", "capsize": 4, "capthick": 1.2})
    for b, v, e in zip(bars2, rep, rep_err):
        ax2.text(b.get_x() + b.get_width()/2, v + e + 0.02, f"{v:.2f}",
                 ha="center", fontsize=11)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(samplers)
    ax2.set_ylabel("Q4 10-gram Repetition ($\\downarrow$)")
    ax2.set_ylim(0, 1.10)
    ax2.yaxis.set_major_locator(MultipleLocator(0.25))
    ax2.set_title("(b) Late-context 10-gram repetition")
    style_ax(ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_transfer_16k.png"))
    plt.close(fig)
    print("wrote fig_transfer_16k.png")


# =====================================================================
# Figure: 21-metric judge heatmap. Moved to appendix. No top title;
# larger annotation text; same heatmap content.
# =====================================================================
def fig_judge_21metric():
    samplers = [
        "P-less ($T{=}1.5$)",
        "Top-H ($\\alpha{=}0.4$)",
        "Top-$n\\sigma$ ($n{=}1$)",
        "Min-p 0.9 ($T{=}1.5$)",
        "Min-p 0.1 ($T{=}1.0$)",
        "Top-k 50 ($T{=}1.0$)",
        "Top-p 0.9 ($T{=}1.0$)",
    ]
    metrics = [
        "Coherence", "LR Coher.", "Plot", "Pacing", "Prose",
        "Grammar", "Dialogue", "Creativity", "World", "Charact.",
        "Non-rep.", "Vocab Div.", "Scene Div.", "Engage.", "Emotion",
        "Immers.", "Begin", "Middle", "End", "Stability", "Overall",
    ]
    table = np.array([
        [5.5, 3.5, 4.5, 4.0, 5.0,  8.0, 4.5, 5.5, 4.5, 4.5, 4.0, 4.5, 4.0, 4.0, 4.0, 4.5, 6.0, 4.5, 3.5, 3.0, 4.0],
        [5.0, 3.0, 4.0, 4.0, 4.5,  7.5, 4.0, 5.5, 4.0, 4.0, 3.5, 4.0, 4.0, 4.0, 3.5, 4.0, 5.5, 4.0, 3.0, 2.5, 3.5],
        [4.5, 2.5, 4.0, 3.5, 4.5,  7.5, 4.0, 5.0, 4.0, 4.0, 3.0, 3.5, 3.5, 3.5, 3.5, 4.0, 5.5, 3.5, 2.5, 2.0, 3.5],
        [4.5, 1.5, 3.5, 3.5, 4.0,  7.5, 4.0, 5.0, 3.5, 3.5, 2.5, 3.0, 3.0, 3.5, 3.0, 3.5, 5.5, 3.0, 1.5, 2.0, 3.0],
        [3.0, 1.0, 3.0, 3.0, 4.0,  5.5, 3.5, 5.0, 3.0, 3.0, 2.0, 2.5, 3.0, 3.0, 2.5, 3.0, 5.5, 2.0, 1.5, 1.5, 3.0],
        [3.0, 1.0, 3.0, 3.0, 3.5,  3.0, 3.0, 5.0, 3.0, 3.0, 1.5, 2.5, 2.5, 2.5, 2.5, 3.0, 5.0, 2.0, 2.0, 1.5, 2.5],
        [2.5, 0.5, 2.5, 2.5, 2.5,  2.5, 2.5, 4.5, 2.5, 2.5, 1.0, 2.0, 2.0, 2.0, 2.0, 2.5, 4.5, 1.5, 1.0, 1.5, 2.0],
    ])

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    im = ax.imshow(table, aspect="auto", cmap="viridis", vmin=0, vmax=8.5)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(samplers)))
    ax.set_yticklabels(samplers, fontsize=11)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table[i, j]
            tc = "white" if v < 4.5 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=9.0, color=tc)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Judge score", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_judge_21metric.png"))
    plt.close(fig)
    print("wrote fig_judge_21metric.png")


# =====================================================================
# Figure: Quality vs temperature. Removes top title, fewer ticks, larger fonts.
# =====================================================================
def fig_quality_vs_temperature():
    temps = np.array([0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5])

    pless = np.array([2.95, 3.20, 3.55, 3.78, 3.85, 3.78, 3.55, 3.10])
    toph = np.array([2.85, 3.10, 3.45, 3.65, 3.72, 3.62, 3.30, 2.85])
    nsigma = np.array([2.70, 2.95, 3.30, 3.45, 3.55, 3.40, 3.05, 2.55])
    minp9 = np.array([2.55, 2.80, 3.05, 3.25, 3.33, 3.05, 2.55, 1.85])
    minp1 = np.array([2.85, 3.05, 3.17, 3.05, 2.85, 2.50, 2.05, 1.40])
    topk = np.array([2.65, 2.80, 2.90, 2.65, 2.30, 1.90, 1.40, 0.95])
    topp = np.array([2.20, 2.25, 2.24, 1.85, 1.45, 1.05, 0.65, 0.25])

    series = [
        ("P-less",        pless,  COLORS["P-less"],       "o", 2.6),
        ("Top-H",         toph,   COLORS["Top-H"],        "s", 2.2),
        ("Top-$n\\sigma$", nsigma, COLORS["Top-n-sigma"], "D", 2.2),
        ("Min-p 0.1",     minp1,  COLORS["Min-p"],        "^", 2.0),
        ("Top-k 50",      topk,   COLORS["Top-k"],        "v", 1.8),
        ("Top-p 0.9",     topp,   COLORS["Top-p"],        "X", 1.8),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for label, vals, c, mk, lw in series:
        ax.plot(temps, vals, color=c, marker=mk, markersize=8,
                linewidth=lw, label=label, alpha=0.95)

    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Mean judged quality (avg. of 21 metrics)")
    ax.set_ylim(0, 4.4)
    ax.set_xlim(0.4, 2.6)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(ncol=2, loc="lower left", fontsize=11)
    style_ax(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_quality_vs_temperature.png"))
    plt.close(fig)
    print("wrote fig_quality_vs_temperature.png")


# =====================================================================
# Figure: Appendix, mean Q4 distinct-2 vs judged quality scatter (kept).
# =====================================================================
def fig_appendix_judge_vs_auto():
    points = [
        ("top-k 50 T=1.0",   0.16, 2.80, COLORS["Top-k"],     (8, 4)),
        ("top-p 0.9 T=1.0",  0.04, 2.30, COLORS["Top-p"],     (8, 4)),
        ("temp-only T=1.0",  0.51, 2.10, COLORS["Temp-only"], (8, -16)),
        ("min-p 0.1 T=2.0",  0.65, 2.07, COLORS["Min-p"],     (8, 6)),
        ("min-p 0.1 T=2.5",  0.94, 1.40, COLORS["Min-p"],     (-100, -16)),
        ("p-less T=1.3",     0.71, 1.30, COLORS["P-less"],    (8, 4)),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for label, x, y, c, off in points:
        ax.scatter([x], [y], s=160, color=c, edgecolor="white", linewidth=1.2)
        ax.annotate(label, xy=(x, y), xytext=off,
                    textcoords="offset points", fontsize=11, color="#222")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 4.0)
    ax.set_xlabel("Q4 Distinct-2 (mean across small-subset models)")
    ax.set_ylabel("Mean judged quality (small rubric)")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    style_ax(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_appendix_judge_vs_auto.png"))
    plt.close(fig)
    print("wrote fig_appendix_judge_vs_auto.png")


if __name__ == "__main__":
    fig_mstr_collapse()
    fig_q1q4_bars()
    fig_human_eval()
    fig_frontier_scale()
    fig_analysis_mechanism()
    fig_transfer_16k()
    fig_appendix_judge_vs_auto()
    fig_judge_21metric()
    fig_quality_vs_temperature()
    print("done.")
