"""Appendix tables: R11 sampler stacks vs bare rule, R12 context depth, R10 large models, seed-2 replication.
python -m lcs.cr_extra -> tables/cr_stack.tex, cr_depth.tex, cr_r10_large.tex, cr_r9_16k_s2.tex"""
import json, os
import pandas as pd
from .cr_tables import load
from .cr_latex import main_table
H = os.path.expanduser("~/lcg-cr"); K = ["model", "uid"]
NM = {"topp0.9": "Top-$p$ 0.9", "minp0.1": "Min-$p$ 0.1", "toph0.4": "Top-H 0.4", "pless": "\\pless{}", "nsigma1": "Top-$n\\sigma$ 1"}


def q(exp):
    m, w, _ = load(exp)
    return m.join(w.groupby(K).j_overall.mean().rename("q"), on=K)


def stack():
    a = q("r11_stack"); b = q("r1_16k"); b = b[b.model.isin(a.model.unique())]
    L = ["\\begin{tabular}{ll" + "cc" * 3 + "}", "\\toprule", " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$T{{=}}{t:g}$}}" for t in (1, 1.5, 2)) + " \\\\",
         "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}", "Rule & Stack & " + " & ".join(["Loop & Quality"] * 3) + " \\\\", "\\midrule"]
    out = []
    for r in NM:
        for lab, d, key in (("none", b, r), ("+ DRY", a, f"dry+{r}"), ("+ top-2 floor", a, f"k2+{r}")):
            cells = []
            for t in (1.0, 1.5, 2.0):
                g = d[(d.sampler == key) & (d["T"] == t)]
                cells += ([f"{g.loop.mean():.2f}", f"{g.q.mean():.1f}"] if len(g) else ["--", "--"])
            L.append(f"{NM[r] if lab == 'none' else ''} & {lab} & " + " & ".join(cells) + " \\\\"); out.append(f"{key} {cells}")
        L.append("\\addlinespace[2pt]")
    L[-1:] = ["\\bottomrule", "\\end{tabular}"]
    open(f"{H}/tables/cr_stack.tex", "w").write("\n".join(L) + "\n")
    open(f"{H}/results/cr_stack.txt", "w").write("\n".join(out) + f"\nmodels={sorted(a.model.unique())} n/cell>={a.groupby(['sampler','T']).size().min()}\n")


def depth():
    md = pd.read_csv(f"{H}/results/cr_depth_gen.csv"); md = md[md.grp != "greedy"]
    g = md.groupby(["T", "D", "grp"]).q.mean().unstack("grp"); g["gap"] = g.tight - g.tailadm
    e = md[md["T"] == 1.0].groupby(["model", "D"]).H1.mean().unstack("D")
    Ds = sorted(md.D.unique())
    L = ["\\begin{tabular}{l" + "c" * len(Ds) + "}", "\\toprule", "Context depth & " + " & ".join(f"{d//1024}K" for d in Ds) + " \\\\", "\\midrule"]
    for t in (1.0, 1.3, 1.6):
        L.append(f"Quality gap, $T{{=}}{t:g}$ & " + " & ".join(f"{g.loc[(t, d), 'gap']:+.2f}" for d in Ds) + " \\\\")
    L.append("\\midrule")
    for m in e.index:
        L.append(f"Entropy at $T{{=}}1$, \\texttt{{{m}}} & " + " & ".join("--" if pd.isna(e.loc[m, d]) else f"{e.loc[m, d]:.2f}" for d in Ds) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{H}/tables/cr_depth.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    stack(); depth()
    for exp, temps in (("r10_large", [1.0, 1.5, 2.0]), ("r9_16k_s2", [1.0, 1.5, 2.0])):
        print(exp, main_table(exp, temps, f"{H}/tables/cr_{exp}.tex"))
    print(open(f"{H}/results/cr_stack.txt").read())
