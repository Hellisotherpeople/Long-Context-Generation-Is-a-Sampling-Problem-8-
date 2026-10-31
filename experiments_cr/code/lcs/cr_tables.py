"""Camera-ready tables from the new waves (analysis venv).

python -m lcs.cr_tables r1_16k [r2_nonfic16k ...]  -> ~/lcg-cr/results/cr_<exp>.txt / .json
Per (sampler, T): n, terminal-loop rate, Q4 rep10, judge window overall by depth (first / last window),
judge global overall, and a per-model-averaged version (each model weighted equally).
Bootstrap 95% CIs over prompts x models (2000 resamples, fixed seed).
"""
from __future__ import annotations

import glob, json, os, sys
import numpy as np, pandas as pd

H = os.path.expanduser("~/lcg-cr")


def load(exp):
    met, win, glo = [], [], []
    for run in sorted(glob.glob(f"{H}/runs/{exp}/*")):
        m = os.path.basename(run)
        if os.path.exists(f"{run}/metrics.jsonl"):
            for l in open(f"{run}/metrics.jsonl"):
                r = json.loads(l)
                met.append(dict(model=m, uid=r["uid"], sampler=r["sampler"], T=r["T"], pid=r["pid"],
                                loop=r.get("loop_onset_win") is not None, q4rep10=r.get("Q4_rep10"),
                                ntok=r.get("n_tokens")))
        for f, out in ((f"{run}/judge_gptoss/window.jsonl", win), (f"{run}/judge_gptoss/global.jsonl", glo)):
            if os.path.exists(f):
                for l in open(f):
                    r = json.loads(l)
                    if not isinstance(r.get("scores"), dict) or "overall" not in r["scores"]:
                        continue
                    out.append(dict(model=m, uid=r["uid"], start=r.get("start", -1), **{f"j_{k}": v for k, v in r["scores"].items()}))
    return pd.DataFrame(met), pd.DataFrame(win), pd.DataFrame(glo)


def boot(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed).choice(x, (B, len(x))).mean(1)
    return tuple(np.round(np.percentile(r, [2.5, 97.5]), 3))


def main(exps):
    for exp in exps:
        met, win, glo = load(exp)
        if met.empty:
            print(exp, "no metrics"); continue
        if not win.empty:
            K = ["model", "uid"]
            first = win[win.start == 0].groupby(K).j_overall.mean().rename("w_first")
            last = win.sort_values("start").groupby(K).j_overall.last().rename("w_last")
            wmean = win.groupby(K).j_overall.mean().rename("w_mean")
            wcoh_min = win.groupby(K).j_coherence.min().rename("w_coh_min")
            met = met.join(first, on=K).join(last, on=K).join(wmean, on=K).join(wcoh_min, on=K)
        if not glo.empty:
            met = met.join(glo.set_index(["model", "uid"]).j_overall.rename("g_overall"), on=["model", "uid"])
        cols = [c for c in ("loop", "q4rep10", "w_first", "w_last", "w_mean", "g_overall") if c in met]
        if "w_coh_min" in met:
            met["healthy"] = (~met.loop) & (met.w_coh_min >= 6)
            cols.append("healthy")
        assert not met.duplicated(["model", "uid"]).any(), "duplicate generations after join"
        rows = []
        for (s, T), g in met.groupby(["sampler", "T"]):
            r = dict(sampler=s, T=T, n=len(g), n_models=g.model.nunique())
            for c in cols:
                v = g[c].astype(float)
                r[c] = round(v.mean(), 3)
                r[c + "_ci"] = boot(v)
                r[c + "_permodel"] = round(g.groupby("model")[c].mean().astype(float).mean(), 3)
            rows.append(r)
        t = pd.DataFrame(rows).sort_values(["T", "sampler"])
        os.makedirs(f"{H}/results", exist_ok=True)
        t.to_json(f"{H}/results/cr_{exp}.json", orient="records", indent=1)
        show = ["sampler", "T", "n", "n_models"] + [c for c in cols] + [c + "_ci" for c in cols if c in ("w_last", "healthy")]
        txt = t[[c for c in show if c in t]].to_string(index=False)
        open(f"{H}/results/cr_{exp}.txt", "w").write(txt + "\n")
        print(f"== {exp}\n{txt}")
        met.drop(columns=[], errors="ignore").to_csv(f"{H}/results/cr_{exp}_gen.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
