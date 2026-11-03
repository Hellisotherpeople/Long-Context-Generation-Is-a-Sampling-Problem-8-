"""LiveCodeBench v6 (thinking, 32K) table for one model: pass@1 and paired W/L vs top-p 0.95 at the same T.
python -m lcs.cr_lcb MODEL -> tables/cr_lcb.tex, results/cr_lcb_paired.csv"""
import os, sys
import pandas as pd
from scipy.stats import binomtest
H = os.path.expanduser("~/lcg-cr"); M = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b"
NM = {"nsigma1": "Top-$n\\sigma$ 1", "minp0.1": "Min-$p$ 0.1", "pless": "\\pless{}", "toph0.4": "Top-H 0.4"}
g = pd.read_parquet(f"{H}/results/lcb_gen.parquet"); g = g[g.model == M]; idx = ["pid", "i"]; TS = (0.6, 1.0, 1.3)
rows = []
for T in TS:
    b = g[(g.sampler == "topp0.95") & (g["T"] == T)].set_index(idx)["pass"].astype(float)
    for s in NM:
        a = g[(g.sampler == s) & (g["T"] == T)].set_index(idx)["pass"].astype(float)
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        w = int(((j.a == 1) & (j.b == 0)).sum()); l = int(((j.a == 0) & (j.b == 1)).sum())
        rows.append(dict(T=T, sampler=s, acc=j.a.mean(), topp=j.b.mean(), w=w, l=l, n=len(j),
                         p=binomtest(w, w + l).pvalue if w + l else 1.0))
d = pd.DataFrame(rows); d.to_csv(f"{H}/results/cr_lcb_paired_{M}.csv", index=False)
top = {T: d[d["T"] == T].topp.iloc[0] for T in TS}
L = ["\\begin{tabular}{lcccccc}", "\\toprule",
     " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$T{{=}}{t:g}$}}" for t in TS) + " \\\\",
     "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
     "Sampler & " + " & ".join(["pass@1 & W/L"] * 3) + " \\\\", "\\midrule",
     "Top-$p$ 0.95 & " + " & ".join(f"{100 * top[t]:.1f} & base" for t in TS) + " \\\\"]
for s in NM:
    c = []
    for t in TS:
        r = d[(d.sampler == s) & (d["T"] == t)].iloc[0]
        c.append(f"{100 * r.acc:.1f} & {r.w}/{r.l}" + ("$^{*}$" if r.p < 0.05 else ""))
    L.append(f"{NM[s]} & " + " & ".join(c) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{H}/tables/cr_lcb_{M}.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("n per cell", d.n.min(), d.n.max())
