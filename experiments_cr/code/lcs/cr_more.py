"""Tables for the October-campaign benchmarks (LongProc, LongBench v2, NIAH, AIME/HMMT/MATH) and paired tests.
python -m lcs.cr_more -> ~/lcg-cr/tables/cr_more_*.tex
"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
H = os.path.expanduser("~/lcg-cr"); LG = os.path.expanduser("~/lcgen/results")
NAME = {"greedy": "Greedy", "temp": "Pure sampling", "topk50": "Top-$k$ 50", "topp0.9": "Top-$p$ 0.9", "topp0.95": "Top-$p$ 0.95",
        "toph0.4": "Top-H 0.4", "minp0.1": "Min-$p$ 0.1", "pless": "P-less", "nsigma1": "Top-$n\\sigma$ 1"}
ROWS = ["topp0.9", "topk50", "toph0.4", "minp0.1", "pless", "nsigma1", "temp"]


def paired(d, keys, val, ref, T, s):
    r = d[(d.sampler == ref) & (d["T"] == T)].set_index(keys)[val]
    x = d[(d.sampler == s) & (d["T"] == T)].set_index(keys)[val]
    j = pd.concat([x.rename("a"), r.rename("b")], axis=1, join="inner")
    w, l = int((j.a > j.b).sum()), int((j.a < j.b).sum())
    return w, l, (binomtest(w, w + l).pvalue if w + l else 1.0)


def wl(w, l, p):
    return f"{w}--{l}" + ("$^{*}$" if p < 0.05 else "")


def bench_table(gen, keys, val, temps, ref, rows, scale=100):
    out = ["\\begin{tabular}{l" + "rr" * len(temps) + "}", "\\toprule",
           "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$T{{=}}{T:g}$}}" for T in temps) + " \\\\",
           " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(temps))),
           "Sampler & " + " & ".join(["Score & W/L"] * len(temps)) + " \\\\", "\\midrule"]
    m = gen.groupby(["sampler", "T"])[val].mean()
    for s in rows:
        cells = []
        for T in temps:
            if (s, T) not in m.index: cells += ["--", ""]; continue
            cells.append(f"{scale * m.loc[(s, T)]:.1f}")
            cells.append("" if s == ref else wl(*paired(gen, keys, val, ref, T, s)))
        out.append(f"{NAME[s]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main():
    os.makedirs(f"{H}/tables", exist_ok=True)
    lp = pd.read_parquet(f"{LG}/longproc_gen.parquet")
    open(f"{H}/tables/cr_more_longproc.tex", "w").write(bench_table(lp, ["model", "task", "item"], "score", [0.7, 1.0, 1.5], "topp0.9", ROWS))
    lb = pd.read_parquet(f"{LG}/lb2_gen.parquet"); lb["c"] = lb.correct.astype(float)
    open(f"{H}/tables/cr_more_lb2.tex", "w").write(bench_table(lb, ["model", "pid"], "c", [0.7, 1.0, 1.5], "topp0.9", ROWS))
    ni = pd.read_parquet(f"{LG}/niah_gen.parquet"); ni["c"] = ni.correct.astype(float)
    open(f"{H}/tables/cr_more_niah.tex", "w").write(bench_table(ni, ["model", "rope", "L", "inst", "key", "i"], "c", [0.7, 1.0, 1.5], "topp0.9", ROWS))
    rr = ["topp0.95", "toph0.4", "minp0.1", "pless", "nsigma1"]
    for f, nm in [("reason2_gen", "reason2"), ("reason_gen", "reason")]:
        r = pd.read_parquet(f"{LG}/{f}.parquet"); r["c"] = r.correct.astype(float)
        open(f"{H}/tables/cr_more_{nm}.tex", "w").write(bench_table(r, ["model", "set", "pid", "i"], "c", [0.6, 1.0, 1.3], "topp0.95", rr))
    print("ok")


if __name__ == "__main__":
    main()
