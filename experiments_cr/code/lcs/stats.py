"""Statistics helpers: clustered bootstrap CIs and regression tests for the paper's claims."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def boot_ci(df, value, by, cluster="pid", n_boot=2000, seed=0, alpha=0.05):
    """Mean of `value` per group `by` with a cluster bootstrap CI (resampling clusters, e.g. prompts)."""
    rng = np.random.default_rng(seed)
    out = []
    for key, g in df.groupby(by):
        clusters = g[cluster].unique()
        means = g.groupby(cluster)[value].mean()
        sizes = g.groupby(cluster)[value].size()
        idx = rng.integers(0, len(clusters), (n_boot, len(clusters)))
        m = means.reindex(clusters).values
        w = sizes.reindex(clusters).values
        bs = (m[idx] * w[idx]).sum(1) / w[idx].sum(1)
        lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
        rec = dict(zip(by if isinstance(by, (list, tuple)) else [by], key if isinstance(key, tuple) else [key]))
        rec.update({"mean": float(g[value].mean()), "lo": float(lo), "hi": float(hi), "n": int(len(g))})
        out.append(rec)
    return pd.DataFrame(out)


def temp_slope(df, outcome, sampler_col="sampler", t_col="T", cluster="cid"):
    """Per-sampler logistic regression outcome ~ T (cluster-robust SEs).  Returns slope, CI, p."""
    import statsmodels.formula.api as smf
    rows = []
    for s, g in df.groupby(sampler_col):
        g = g[g[t_col] > 0]
        if g[outcome].nunique() < 2 or g[t_col].nunique() < 2:
            continue
        g = g.assign(y=g[outcome].astype(int))
        try:
            m = smf.logit(f"y ~ {t_col}", data=g).fit(disp=0, cov_type="cluster",
                                                     cov_kwds={"groups": pd.factorize(g[cluster])[0]})
            b, (lo, hi) = m.params[t_col], m.conf_int().loc[t_col].values
            rows.append({"sampler": s, "slope": b, "lo": lo, "hi": hi, "p": m.pvalues[t_col], "n": len(g)})
        except Exception as e:
            rows.append({"sampler": s, "slope": np.nan, "err": repr(e)[:80]})
    return pd.DataFrame(rows)


def _nan(v):
    return np.array([np.nan if x is None else x for x in v], float)


def claim_end_pressure(exp="core32k"):
    """Fraction of terminal loops that start at/after the first end spike (p_end > 0.5), per model and pooled,
    with a bootstrap CI clustered by prompt."""
    from .figs import load
    df = load(exp)
    rows = []
    for _, r in df.iterrows():
        pe = _nan(r["win"].get("pend", []))
        lo = r["loop_onset_win"]
        if not len(pe) or np.isnan(pe).all() or lo is None or (isinstance(lo, float) and np.isnan(lo)):
            continue
        hi = np.where(pe > 0.5)[0]
        rows.append({"model": r["model"], "pid": r["pid"], "has_end": len(hi) > 0,
                     "after": bool(len(hi) and lo >= hi[0]), "gap": (lo - hi[0]) if len(hi) else np.nan})
    d = pd.DataFrame(rows)
    w = d[d.has_end].assign(all="all")
    out = pd.concat([boot_ci(w, "after", "model"), boot_ci(w, "after", "all").rename(columns={"all": "model"})])
    noend = d.groupby("model").has_end.apply(lambda x: 1 - x.mean()).rename("frac_loops_without_end")
    return out.merge(noend, left_on="model", right_index=True, how="left"), float(np.nanmedian(w.gap))


def claim_k2(base="mt32k", k2="mtk2_32k", temps=(1.0, 1.6)):
    """Paired effect of keep_top=2 (same prompt/seed/T/base rule): change in loop and healthy rates, bootstrap CI
    over prompts."""
    from .figs import load, load_judge, outcome_judge
    frames = []
    for exp in (base, k2):
        J = load_judge(exp)
        df = load(exp)
        df = df[df["T"] > 0].copy()
        df["out"] = df.apply(lambda r: outcome_judge(r, J), axis=1)
        frames.append(df[df.out.notna()])
    b, k = frames
    k = k.assign(rule=k.sampler.str.replace("+k2", "", regex=False))
    m = b.merge(k, left_on=["model", "pid", "seed", "T", "sampler"], right_on=["model", "pid", "seed", "T", "rule"],
                suffixes=("_b", "_k"))
    m = m[m["T"].isin(temps)]
    m["d_loop"] = (m.out_k == "loop").astype(float) - (m.out_b == "loop").astype(float)
    m["d_healthy"] = (m.out_k == "healthy").astype(float) - (m.out_b == "healthy").astype(float)
    a = boot_ci(m, "d_loop", ["model", "rule", "T"]).rename(columns={"mean": "d_loop", "lo": "loop_lo", "hi": "loop_hi"})
    h = boot_ci(m, "d_healthy", ["model", "rule", "T"])[["mean", "lo", "hi"]]
    a[["d_healthy", "healthy_lo", "healthy_hi"]] = h.values
    pooled = boot_ci(m.assign(all="pooled"), "d_healthy", ["all", "T"])
    return a, pooled


def claim_niah(L=1000000, T=1.5):
    """Accuracy minus greedy at the longest context, bootstrap over (haystack instance, needle) clusters."""
    g = pd.read_parquet(os.path.expanduser("~/lcg-cr/results/niah_gen.parquet"))
    g = g[g.L == L]
    g["cl"] = g["inst"].astype(str) + "_" + g["key"].astype(str)
    out = []
    for model, x in g.groupby("model"):
        gr = x[x.sampler == "greedy"].groupby("cl").correct.mean()
        y = x[(x["T"] == T)].copy()
        y["d"] = y.correct.astype(float) - y.cl.map(gr)
        out.append(boot_ci(y.dropna(subset=["d"]).assign(model=model), "d", ["model", "sampler"], cluster="cl"))
    return pd.concat(out) if out else pd.DataFrame()


def claim_reason():
    """Accuracy and loop rate per model/set/sampler/T with bootstrap CIs over problems."""
    g = pd.read_parquet(os.path.expanduser("~/lcg-cr/results/reason_gen.parquet"))
    a = boot_ci(g.assign(correct=g.correct.astype(float)), "correct", ["model", "set", "sampler", "T"])
    l = boot_ci(g.assign(looped=g.looped.astype(float)), "looped", ["model", "set", "sampler", "T"])
    a[["loop", "loop_lo", "loop_hi"]] = l[["mean", "lo", "hi"]].values
    return a


def claim_book_slopes(exp="book32k"):
    """Per-rule logistic slope of terminal looping on temperature in base-model book continuation (cluster: book)."""
    from .figs import load
    df = load(exp)
    df = df[df["T"] > 0].assign(looped=lambda d: d.loop_onset_win.notna())
    df["cid"] = df["model"] + "_" + df["pid"]
    return temp_slope(df, "looped")


def claim_ordering(exp="core32k", rules=("temp", "topk50", "topp0.95", "topp0.9", "minp0.05", "minp0.1", "toph0.4",
                                         "nsigma1", "pless"), n_perm=5000, seed=0):
    """Is the ordering of rules by loop-proneness shared across models?  Each model ranks the rules by its terminal-loop
    rate averaged over temperature; Kendall's W measures agreement (permutation p-value: shuffle ranks within model)."""
    from .figs import load
    df = load(exp)
    df = df[(df["T"] > 0) & df.sampler.isin(rules)].assign(looped=lambda d: d.loop_onset_win.notna())
    M = df.groupby(["model", "sampler"]).looped.mean().unstack("sampler")[list(rules)].dropna()
    R = M.rank(axis=1).values                    # models x rules
    m, n = R.shape

    def W(r):
        S = ((r.sum(0) - m * (n + 1) / 2) ** 2).sum()
        return 12 * S / (m ** 2 * (n ** 3 - n))
    w = W(R)
    rng = np.random.default_rng(seed)
    null = [W(np.array([rng.permutation(row) for row in R])) for _ in range(n_perm)]
    p = (1 + sum(x >= w for x in null)) / (1 + n_perm)
    mean_rank = pd.Series(R.mean(0), index=list(rules)).sort_values()
    return {"W": float(w), "p": float(p), "n_models": int(m), "models": list(M.index)}, mean_rank, M.round(2)


if __name__ == "__main__":
    import os
    import sys
    pd.set_option("display.width", 250, "display.max_rows", 2000)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("loop", "all"):
        lg = pd.read_parquet(os.path.expanduser("~/lcg-cr/results/loop_gen.parquet"))
        k8 = lg[lg.k == 8]
        print("stay ~ T at k=8 (logit slope per unit T, cluster-robust by context):")
        print(temp_slope(k8, "stay").round(3).to_string(index=False))
        k8 = k8.assign(single=(k8.kept0 <= 1))
        print("singleton ~ T at k=8:")
        print(temp_slope(k8, "single").round(3).to_string(index=False))
    if what in ("end", "all"):
        t, med = claim_end_pressure()
        print("\n== loops starting after the first end spike (core32k); median gap (K tokens):", med)
        print(t.round(3).to_string(index=False))
    if what in ("k2", "all"):
        a, pooled = claim_k2()
        print("\n== keep_top=2 paired effects (mt32k vs mtk2_32k)")
        print(a.round(3).to_string(index=False))
        print(pooled.round(3).to_string(index=False))
    if what in ("niah", "all"):
        print("\n== NIAH 1M: accuracy - greedy at T=1.5 / T=1.0")
        print(claim_niah(T=1.5).round(3).to_string(index=False))
        print(claim_niah(T=1.0).round(3).to_string(index=False))
    if what in ("reason", "all"):
        print("\n== reasoning accuracy / loop rate with CIs")
        print(claim_reason().round(3).to_string(index=False))
    if what in ("order", "all"):
        for exp in ("core32k", "mt32k", "book32k"):
            res, mr, M = claim_ordering(exp)
            print(f"\n== rule ordering by loop-proneness ({exp}):", res)
            print(mr.round(2).to_string())
    if what in ("book", "all"):
        print("\n== book32k: looping ~ T per rule")
        print(claim_book_slopes().round(3).to_string(index=False))
