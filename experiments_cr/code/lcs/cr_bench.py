"""Published-benchmark table: Arena-Write (healthy rate, last-window judge) + LongBench-Write official S, by sampler x T.
python -m lcs.cr_bench -> tables/cr_bench.tex, results/cr_bench.txt"""
import json, os
import pandas as pd
H = os.path.expanduser("~/lcg-cr")
NAME = {"temp": "Pure sampling", "topk50": "Top-$k$ 50", "topp0.9": "Top-$p$ 0.9", "minp0.1": "Min-$p$ 0.1",
        "toph0.4": "Top-H 0.4", "pless": "\\pless{}", "nsigma1": "Top-$n\\sigma$ 1"}
ORDER = list(NAME); TS = [1.0, 1.5, 2.0]


def main():
    aw = {(r["sampler"], r["T"]): r for r in json.load(open(f"{H}/results/cr_arenawrite.json"))}
    lb = pd.read_parquet(f"{H}/results/lbwrite_cells.parquet").groupby(["sampler", "T"]).S.mean()
    wb = pd.read_parquet(f"{H}/results/wbench_cells.parquet").groupby(["sampler", "T"]).score.mean()
    m_wb = pd.read_parquet(f"{H}/results/wbench_cells.parquet").model.nunique()
    n_aw = sorted({r["n"] for r in aw.values()}); m_aw = max(r["n_models"] for r in aw.values())
    m_lb = pd.read_parquet(f"{H}/results/lbwrite_cells.parquet").model.nunique()
    L = ["\\begin{tabular}{l" + "ccc" * 4 + "}", "\\toprule",
         " & \\multicolumn{3}{c}{Arena-Write healthy} & \\multicolumn{3}{c}{Arena-Write last window} & \\multicolumn{3}{c}{LongBench-Write $S$} & \\multicolumn{3}{c}{WritingBench} \\\\",
         "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}",
         "Sampler & " + " & ".join([f"$T{{=}}{t:g}$" for t in TS] * 4) + " \\\\", "\\midrule"]
    txt = []
    for s in ORDER:
        a = [aw.get((s, t), {}).get("healthy") for t in TS]; b = [aw.get((s, t), {}).get("w_last") for t in TS]
        c = [lb.get((s, t)) for t in TS]; w = [wb.get((s, t)) for t in TS]
        f = lambda x, d: "--" if x is None or x != x else f"{x:.{d}f}"
        L.append(f"{NAME[s]} & " + " & ".join([f(x, 2) for x in a] + [f(x, 1) for x in b] + [f(x, 1) for x in c] + [f(x, 2) for x in w]) + " \\\\")
        txt.append(f"{s} healthy {a} last {b} lbwS {c} wb {w}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{H}/tables/cr_bench.tex", "w").write("\n".join(L) + "\n")
    meta = f"Arena-Write n/cell {n_aw[0]}-{n_aw[-1]}, models {m_aw}; LongBench-Write models {m_lb}; WritingBench models {m_wb}"
    open(f"{H}/results/cr_bench.txt", "w").write("\n".join(txt) + "\n" + meta + "\n"); print("\n".join(txt)); print(meta)


if __name__ == "__main__":
    main()
