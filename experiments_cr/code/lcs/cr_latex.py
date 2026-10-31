"""Emit LaTeX tables from results/cr_*.json so no number in the paper is hand-typed.
python -m lcs.cr_latex -> ~/lcg-cr/tables/cr_*.tex"""
import json, os, re
H = os.path.expanduser("~/lcg-cr")
NAME = {"greedy": "Greedy", "temp": "Pure sampling", "topk50": "Top-$k$ 50", "topp0.9": "Top-$p$ 0.9",
        "minp0.1": "Min-$p$ 0.1", "toph0.4": "Top-H 0.4", "pless": "\\pless{}", "nsigma1": "Top-$n\\sigma$ 1"}
ORDER = ["temp", "topk50", "topp0.9", "minp0.1", "toph0.4", "pless", "nsigma1"]


def f(x, d=2):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def main_table(exp, temps, out):
    R = {(r["sampler"], r["T"]): r for r in json.load(open(f"{H}/results/cr_{exp}.json"))}
    cols = "l" + "ccc" * len(temps)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{$T{{=}}{t}$}}" for t in temps)
    sub = " & ".join(["Loop & First & Last"] * len(temps))
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", f" & {head} \\\\",
             " ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(temps))), f"Sampler & {sub} \\\\", "\\midrule"]
    for s in ORDER:
        cells = []
        for t in temps:
            r = R.get((s, t), {})
            cells += [f(r.get("loop")), f(r.get("w_first"), 1), f(r.get("w_last"), 1)]
        lines.append(f"{NAME[s]} & " + " & ".join(cells) + " \\\\")
    g = R.get(("greedy", 0.0))
    if g:
        lines.append("\\midrule")
        lines.append(f"Greedy & \\multicolumn{{3}}{{c}}{{{f(g['loop'])} / {f(g['w_first'],1)} / {f(g['w_last'],1)}}}" + " & " * (3 * len(temps) - 3) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(out, "w").write("\n".join(lines) + "\n")
    n = sorted({r["n"] for r in R.values()}); nm = max(r["n_models"] for r in R.values())
    return n, nm


def pool_table(exp, out):
    txt = open(f"{H}/results/cr_pool_{exp}.txt").read().split("\n\n")[0].splitlines()[1:]
    P = {}
    for l in txt:
        p = l.split()
        if len(p) >= 6:
            P[(p[0], float(p[1]))] = (float(p[3]), float(p[4]))
    temps = sorted({t for (_, t) in P if t > 0})
    lines = ["\\begin{tabular}{l" + "c" * len(temps) + "c}", "\\toprule",
             "Sampler & " + " & ".join(f"$T{{=}}{t}$" for t in temps) + " & Singleton at $T{=}1$ \\\\", "\\midrule"]
    for s in ORDER:
        meds = [P.get((s, t), (None, None))[0] for t in temps]
        sf = P.get((s, 1.0), (None, None))[1]
        lines.append(f"{NAME[s]} & " + " & ".join("--" if m is None else ("all" if m > 100000 else f"{m:,.0f}") for m in meds) + f" & {f(sf)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    os.makedirs(f"{H}/tables", exist_ok=True)
    meta = {}
    for exp, temps in (("r1_16k", [1.0, 1.5, 2.0]), ("r2_nonfic16k", [1.0, 1.5]), ("r4_mt", [1.0, 1.5, 2.0])):
        if os.path.exists(f"{H}/results/cr_{exp}.json"):
            meta[exp] = main_table(exp, temps, f"{H}/tables/cr_{exp}.tex")
            if os.path.exists(f"{H}/results/cr_pool_{exp}.txt"):
                pool_table(exp, f"{H}/tables/cr_pool_{exp}.tex")
    json.dump({k: {"n_per_cell": v[0], "n_models": v[1]} for k, v in meta.items()}, open(f"{H}/tables/cr_meta.json", "w"), indent=1)
    print(meta)
