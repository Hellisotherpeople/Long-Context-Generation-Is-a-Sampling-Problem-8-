"""Teaser and supplementary figures (imported by cr_figs)."""
import json
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def fig_teaser(H, S, INK, save):
    d = json.load(open(f"{H}/results/teaser_dist.json"))
    sw = pd.DataFrame(json.load(open(f"{H}/results/cr_swe.json")))
    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], hspace=0.75, wspace=0.3)
    K = 12
    for row, T in enumerate((1.0, 2.0)):
        ax = fig.add_subplot(gs[row, 0]); e = d[f"T{T}"]; pr = np.array(e["probs"][:K]); kp = e["kept"]; x = np.arange(K)
        ax.bar(x, pr, width=0.8, color="#dedcd6", zorder=2)
        ax.bar(x[:kp["top_nsigma"]], pr[:kp["top_nsigma"]], width=0.8, color=S["nsigma1"][1], zorder=3)
        ax.set_xlim(-0.7, K - 0.3); ax.set_xticks(x)
        ax.set_xticklabels(["\u201c" + (t.strip() or "\\n") + "\u201d" for t in e["tokens"][:K]], fontsize=5.8, rotation=40, ha="right")
        ax.set_ylabel(f"$p$ at $T{{=}}{T:g}$", fontsize=7.2); ax.grid(False); ax.set_ylim(0, 0.55)
        ax.set_yticks([0, 0.25, 0.5])
        lines = [(S["nsigma1"][1], f"top-$n\\sigma$ keeps {kp['top_nsigma']}"),
                 (S["minp0.1"][1], f"min-$p$ keeps {kp['min_p']}"),
                 (S["topp0.9"][1], f"top-$p$ keeps {kp['top_p']:,}")]
        for i_, (c, t) in enumerate(lines):
            ax.text(K - 0.6, 0.52 - i_ * 0.13, t, ha="right", va="top", fontsize=6.6, color=c, fontweight="bold" if i_ == 2 and T == 2.0 else "normal")
        if row == 0:
            ax.set_title("(a) A real next-token distribution, Llama-3.1-8B mid-story", fontsize=7.4)
    ax = fig.add_subplot(gs[:, 1])
    sets = [("r3a_swe", "qwen3.5-9b", "9B"), ("r18_swe_new100", "qwen3.5-9b", "9B\n(new tasks)"),
            ("r20_swe_new100", "qwen3.5-27b", "27B\n(new tasks)"), ("r20_swe_new100", "qwen3.5-35b-a3b", "35B-A3B\n(new tasks)")]
    keys = ["topp0.9", "toph0.4", "minp0.1", "nsigma1"]; w = 0.2
    for j, k in enumerate(keys):
        vals = []
        for e_, m_, _ in sets:
            r = sw[(sw.exp == e_) & (sw.model == m_) & (sw.sampler == k) & (sw["T"] == 2.0)]
            vals.append(100 * r.resolved.iloc[0] if len(r) else np.nan)
        xs = np.arange(len(sets)) + (j - 1.5) * w
        ax.bar(xs, vals, width=w * 0.9, color=S[k][1], label=S[k][0], zorder=3)
        for xx, v in zip(xs, vals):
            if v == 0: ax.text(xx, 0.8, "0", ha="center", fontsize=6, color=S[k][1], fontweight="bold")
    ax.set_xticks(np.arange(len(sets))); ax.set_xticklabels([s_[2] for s_ in sets], fontsize=6.3)
    ax.set_ylabel("SWE-bench Verified resolved (%)", fontsize=7.2); ax.set_ylim(0, 72)
    ax.set_title("(b) Qwen3.5 coding agents at $T{=}2$", fontsize=7.4)
    ax.legend(fontsize=6.3, ncol=2, loc="upper left", handlelength=1, columnspacing=0.8, frameon=False)
    save(fig, "fig_teaser")
