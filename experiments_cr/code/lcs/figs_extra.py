"""Teaser and supplementary figures (imported by cr_figs)."""
import json
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _chips(ax, x, y, parts, fs=7.2):
    """Draw a run of text segments left to right in axes coords. parts: (text, style dict or None)."""
    fig = ax.figure; r = fig.canvas.get_renderer(); inv = ax.transAxes.inverted()
    for txt, sty in parts:
        kw = dict(fontsize=fs, va="baseline", ha="left", transform=ax.transAxes, color="#1f1f1e")
        if sty: kw.update(sty)
        t = ax.text(x, y, txt, **kw)
        bb = t.get_window_extent(renderer=r)
        x = inv.transform((bb.x1, bb.y0))[0] + (0.006 if sty and "bbox" in sty else 0.0)
    return x


def fig_teaser(H, S, INK, save):
    """Figure 1. (a) one real decoding step from Llama-3.1-8B: the story context, the candidate tokens, and which
    tokens each rule keeps at T=1 and T=2; (b) SWE-bench Verified resolve rates at T=2 for Qwen3.5 agents."""
    d = json.load(open(f"{H}/results/teaser_dist.json"))
    sw = pd.DataFrame(json.load(open(f"{H}/results/cr_swe.json")))
    NS, MP, TP, GRAY = S["nsigma1"][1], S["minp0.1"][1], S["topp0.9"][1], "#d9d7d0"
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.22], height_ratios=[0.30, 1.0], hspace=0.22, wspace=0.32,
                          left=0.06, right=0.995, top=0.90, bottom=0.12)
    # ---- context strip
    axc = fig.add_subplot(gs[0, 0:2]); axc.axis("off")
    axc.add_patch(mpl.patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.06",
                  transform=axc.transAxes, fc="#f6f5f1", ec="#cfcdc6", lw=0.6, clip_on=False))
    axc.text(0.015, 0.93, "(a) Llama-3.1-8B-Instruct writing a story (step 329)", transform=axc.transAxes,
             fontsize=7.6, fontweight="bold", va="top", color=INK)
    ctx = d["context"].strip(); tail = ctx.rsplit(". ", 1)[1]
    hi = dict(bbox=dict(boxstyle="round,pad=0.2", fc=NS, ec="none"), color="white", fontweight="bold")
    x = _chips(axc, 0.015, 0.30, [("\u2026 " + tail + " ", dict(style="italic", fontsize=7.6))])
    toks = [t.strip() for t in d["T1.0"]["tokens"][:3]]
    x = _chips(axc, x + 0.004, 0.30, [(toks[0], hi), ("  ", None), (toks[1], hi), ("  ", None), (toks[2], hi)])
    # ---- token bars at T=1 and T=2, with a dot matrix of which tokens each rule keeps
    K = 10; rules = [("top-$n\\sigma$", NS, "top_nsigma"), ("min-$p$", MP, "min_p"), ("top-$p$", TP, "top_p")]
    for col, T in enumerate((1.0, 2.0)):
        ax = fig.add_subplot(gs[1, col]); e = d[f"T{T}"]; pr = np.array(e["probs"][:K]); kp = e["kept"]; y = np.arange(K)
        ax.barh(y, pr, height=0.72, color=[NS if i < kp["top_nsigma"] else GRAY for i in range(K)], zorder=3)
        for i in range(K):
            strong = i < kp["top_nsigma"]
            ax.text(-0.012, i, e["tokens"][i].strip() or "\\n", ha="right", va="center", fontsize=6.6,
                    color=INK if strong else "#62615d", fontweight="bold" if strong else "normal")
            ax.text(pr[i] + 0.008, i, f"{pr[i]:.2f}", va="center", fontsize=5.4, color="#62615d")
        for j, (name, c, key) in enumerate(rules):
            xb = 0.62 + 0.10 * j; n = kp[key]
            ax.text(xb - 0.02, -0.85, name, ha="left", va="bottom", rotation=35, fontsize=6.0, color=c, fontweight="bold")
            for i in range(K):
                kept = i < n
                ax.scatter([xb], [i], s=15, marker="o", color=c if kept else "white", edgecolor=c if kept else "#cfcdc6", lw=0.6, zorder=4)
            lab = f"{n:,}" if n <= K else f"{n:,}"
            ax.text(xb, K + 0.15, lab, ha="center", va="top", fontsize=6.4 if n <= K else 6.6, color=c, fontweight="bold")
        ax.text(0.545, K + 0.15, "kept:", ha="right", va="top", fontsize=5.9, color="#62615d")
        ax.set_ylim(K + 1.25, -2.2); ax.set_yticks([]); ax.set_xlim(0, 0.88); ax.set_xticks([0, 0.25, 0.5])
        ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
        ax.set_xlabel("Probability", fontsize=6.8, x=0.32)
        ax.text(0.25, -0.9, f"Temperature {T:g}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
    # ---- (b) agents
    ax = fig.add_subplot(gs[:, 2])
    sets = [("r3a_swe", "qwen3.5-9b", "9B"), ("r18_swe_new100", "qwen3.5-9b", "9B\nfresh"),
            ("r20_swe_new100", "qwen3.5-27b", "27B\nfresh"), ("r20_swe_new100", "qwen3.5-35b-a3b", "35B-A3B\nfresh")]
    keys = ["topp0.9", "topk50", "toph0.4", "pless", "minp0.1", "nsigma1"]; w = 0.135
    for j, k in enumerate(keys):
        xs = np.arange(len(sets)) + (j - 2.5) * w
        for xx, (e_, m_, _) in zip(xs, sets):
            r = sw[(sw.exp == e_) & (sw.model == m_) & (sw.sampler == k) & (sw["T"] == 2.0)]
            if not len(r):
                ax.text(xx, 0.9, "\u2013", ha="center", va="bottom", fontsize=6.5, color="#a9a7a0"); continue
            v = 100 * r.resolved.iloc[0]
            ax.bar(xx, v, width=w * 0.92, color=S[k][1], zorder=3)
            if v == 0: ax.text(xx, 0.9, "0", ha="center", va="bottom", fontsize=5.6, color=S[k][1], fontweight="bold")
            elif k == "nsigma1": ax.text(xx, v + 1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=6.0, color=S[k][1], fontweight="bold")
    ax.set_xticks(np.arange(len(sets))); ax.set_xticklabels([s_[2] for s_ in sets], fontsize=6.4)
    ax.set_ylabel("Qwen3.5 SWE-bench Verified issues resolved (%)", fontsize=6.8); ax.set_ylim(0, 82); ax.set_xlim(-0.5, len(sets) - 0.5)
    ax.set_title("(b) Coding agents at temperature 2", fontsize=7.6, loc="left", fontweight="bold")
    hs = [mpl.patches.Patch(color=S[k][1], label=S[k][0]) for k in keys]
    ax.legend(handles=hs, fontsize=6.2, ncol=3, loc="upper left", handlelength=0.9, columnspacing=0.7, frameon=False)
    ax.text(0.015, 0.80, "\u2013 not run on the fresh issue sets", transform=ax.transAxes, ha="left", fontsize=5.8, color="#8a8983")
    save(fig, "fig_teaser")
