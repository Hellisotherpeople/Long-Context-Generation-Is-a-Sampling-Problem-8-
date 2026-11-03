"""Paired SWE-bench comparisons (pre-registered: each config vs top-p 0.9 at the same T, same instance).
python -m lcs.cr_swe -> results/cr_swe.txt/.json, tables/cr_swe_<exp>.tex"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest

H = os.path.expanduser("~/lcg-cr")
NAME = {"temp": "Pure sampling", "topk50": "Top-$k$ 50", "topp0.9": "Top-$p$ 0.9", "minp0.1": "Min-$p$ 0.1",
        "toph0.4": "Top-H 0.4", "pless": "\\pless{}", "nsigma1": "Top-$n\\sigma$ 1"}
ORDER = list(NAME)


def main():
    df = pd.read_parquet(f"{H}/results/swe_inst.parquet")
    df = df[df.exp.isin(["r3a_swe", "r3b_swelong", "r3c_swe", "r7_swe", "r16_swe_s2", "r18_swe_new100", "r20_swe_new100", "r21_swe_new100"])].copy()
    df["sampler"] = df.config.str.split("@").str[0]
    df["T"] = df.config.str.extract(r"@T([0-9.]+)").astype(float)
    out, txt = [], []
    for (exp, model), g in df.groupby(["exp", "model"]):
        for T, gt in g.groupby("T"):
            base = gt[gt.sampler == "topp0.9"].set_index("instance").resolved
            for s, gs in gt.groupby("sampler"):
                r = gs.set_index("instance").resolved
                n_graded = int(r.notna().sum())
                row = dict(exp=exp, model=model, T=T, sampler=s, n=len(r), graded=n_graded,
                           resolved=float(r.mean()) if n_graded else np.nan,
                           loop=float(gs.loop.mean()), steps=float(gs.steps.mean()),
                           format_err=float(gs.format_errors.mean()))
                if s != "topp0.9" and len(base):
                    j = pd.concat([r.rename("a"), base.rename("b")], axis=1).dropna()
                    w, l = int(((j.a == 1) & (j.b == 0)).sum()), int(((j.a == 0) & (j.b == 1)).sum())
                    row.update(pairs=len(j), wins=w, losses=l,
                               p_sign=float(binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0)
                out.append(row)
    t = pd.DataFrame(out)
    t.to_json(f"{H}/results/cr_swe.json", orient="records", indent=1)
    s = t.round(3).to_string(index=False)
    open(f"{H}/results/cr_swe.txt", "w").write(s + "\n"); print(s)
    nmod = t.groupby("exp").model.nunique()
    for (exp, model), g in t.groupby(["exp", "model"]):
        temps = sorted(g["T"].unique())
        L = ["\\begin{tabular}{l" + "cc" * len(temps) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$T{{=}}{x}$}}" for x in temps) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(temps))),
             "Sampler & " + " & ".join(["Resolved & W/L vs top-$p$"] * len(temps)) + " \\\\", "\\midrule"]
        for smp in ORDER:
            cells = []
            for x in temps:
                r = g[(g.sampler == smp) & (g["T"] == x)]
                if r.empty or r.graded.iloc[0] < 0.9 * r.n.iloc[0]:
                    cells += ["--", "--"]; continue
                r = r.iloc[0]
                res = f"{100*r.resolved:.0f}\\%"
                if smp == "topp0.9":
                    wl = "base"
                else:
                    star = "$^{*}$" if r.p_sign < 0.05 else ""
                    wl = f"{int(r.wins)}/{int(r.losses)}{star}"
                cells += [res, wl]
            if any(c != "--" for c in cells):
                L.append(f"{NAME[smp]} & " + " & ".join(cells) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(f"{H}/tables/cr_swe_{exp}__{model}.tex", "w").write("\n".join(L) + "\n")
        if nmod[exp] == 1:   # single-model waves keep the short name used by main.tex
            open(f"{H}/tables/cr_swe_{exp}.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
