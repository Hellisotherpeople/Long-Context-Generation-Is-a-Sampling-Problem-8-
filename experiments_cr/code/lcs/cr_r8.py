"""R8 temperature-last ablation vs R5 temperature-first on the same 4 models.
python -m lcs.cr_r8 -> tables/cr_r8.tex, results/cr_r8.txt"""
import os
from .cr_tables import load
H = os.path.expanduser("~/lcg-cr")
NAME = {"topp0.9": "Top-$p$ 0.9", "topk50": "Top-$k$ 50", "toph0.4": "Top-H 0.4", "minp0.1": "Min-$p$ 0.1", "pless": "\\pless{}"}


def q(exp, models=None):
    met, win, _ = load(exp); K = ["model", "uid"]
    met = met.join(win.groupby(K).j_overall.mean().rename("om"), on=K)
    if models is not None:
        met = met[met.model.isin(models)]
    return met


def main():
    tl = q("r8_tl"); tf = q("r5_hp", tl.model.unique())
    tl["base"] = tl.sampler.str.replace("_tl", "")
    A = tf.groupby(["sampler", "T"]).om.mean(); B = tl.groupby(["base", "T"]).om.mean()
    L = tl.groupby(["base", "T"]).loop.mean()
    rows = ["\\begin{tabular}{lccccccc}", "\\toprule",
            " & $T{=}1$ & \\multicolumn{3}{c}{Temperature first} & \\multicolumn{3}{c}{Candidate set fixed at $T{=}1$} \\\\",
            "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
            "Sampler & & $T{=}1.6$ & $T{=}2$ & $T{=}3$ & $t{=}1.5$ & $t{=}2$ & $t{=}3$ \\\\", "\\midrule"]
    txt = []
    for s in ["topp0.9", "topk50", "toph0.4", "minp0.1", "pless"]:
        c = [A[(s, 1.0)], A[(s, 1.6)], A[(s, 2.0)], A[(s, 3.0)], B[(s, 1.5)], B[(s, 2.0)], B[(s, 3.0)]]
        rows.append(f"{NAME[s]} & " + " & ".join(f"{x:.1f}" for x in c) + " \\\\")
        txt.append(f"{s} " + " ".join(f"{x:.2f}" for x in c) + " | loop_tl " + " ".join(f"{L[(s,t)]:.2f}" for t in (1.5, 2.0, 3.0)))
    rows += ["\\bottomrule", "\\end{tabular}"]
    open(f"{H}/tables/cr_r8.tex", "w").write("\n".join(rows) + "\n")
    open(f"{H}/results/cr_r8.txt", "w").write("cols: T1 TF1.6 TF2 TF3 TL1.5 TL2 TL3\n" + "\n".join(txt) + f"\nn/cell={tl.groupby(['base','T']).size().min()} models={sorted(tl.model.unique())}\n")
    print(open(f"{H}/results/cr_r8.txt").read())


if __name__ == "__main__":
    main()
