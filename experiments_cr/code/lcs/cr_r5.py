"""R5 healthy-region map: per sampler setting x T, healthy rate (no terminal loop AND judge window coherence >= 6 in every
window) and mean window overall. Also the pre-registered primary quantity: width of the usable-T interval, defined as the
set of T where mean window overall >= 4.0 (= within ~0.5 of the T<=1 level for every rule).
python -m lcs.cr_r5 -> results/cr_r5.txt, tables/cr_r5.tex, figs/fig_r5.pdf"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .cr_tables import load

H = os.path.expanduser("~/lcg-cr")
FAM = [("Top-$p$", "topp"), ("Top-$k$", "topk"), ("Top-H", "toph"), ("Min-$p$", "minp"), ("Top-$n\\sigma$", "nsigma")]


def main():
    met, win, _ = load("r5_hp")
    K = ["model", "uid"]
    w = win.groupby(K).agg(cmin=("j_coherence", "min"), omean=("j_overall", "mean"))
    met = met.join(w, on=K)
    met["healthy"] = (~met.loop) & (met.cmin >= 6)
    h = met.pivot_table(index="sampler", columns="T", values="healthy", aggfunc="mean")
    q = met.pivot_table(index="sampler", columns="T", values="omean", aggfunc="mean")
    lp = met.pivot_table(index="sampler", columns="T", values="loop", aggfunc="mean")
    usable = (q >= 4.0)
    width = usable.apply(lambda r: list(r.index[r.values]), axis=1)
    txt = ("healthy rate\n" + h.round(2).to_string() + "\n\nmean window overall\n" + q.round(2).to_string() +
           "\n\nloop rate\n" + lp.round(2).to_string() + "\n\nusable T (mean window overall >= 4.0)\n" + width.to_string() +
           f"\n\nn per cell: {met.groupby(['sampler','T']).size().min()}-{met.groupby(['sampler','T']).size().max()}, models: {met.model.nunique()}")
    open(f"{H}/results/cr_r5.txt", "w").write(txt + "\n"); print(txt)
    # figure: mean window quality vs T, one panel per family
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.1), sharey=True)
    for ax, (title, pre) in zip(axes, FAM):
        rows = [s for s in q.index if s.startswith(pre)]
        rows = sorted(rows, key=lambda s: float(s[len(pre):]))
        cm = plt.cm.viridis(np.linspace(0.1, 0.85, len(rows)))
        for c, s in zip(cm, rows):
            ax.plot(q.columns, q.loc[s].values, "-o", ms=3, color=c, label=s[len(pre):])
        ax.plot(q.columns, q.loc["temp"].values, ":", color="gray", lw=1)
        ax.set_title(title); ax.set_xlabel("Temperature"); ax.axhline(4.0, color="k", lw=0.5, alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Judged window quality (1-10)")
    plt.tight_layout(); os.makedirs(f"{H}/figs", exist_ok=True)
    plt.savefig(f"{H}/figs/fig_r5.pdf"); plt.savefig(f"{H}/figs/fig_r5.png", dpi=200)


if __name__ == "__main__":
    main()
