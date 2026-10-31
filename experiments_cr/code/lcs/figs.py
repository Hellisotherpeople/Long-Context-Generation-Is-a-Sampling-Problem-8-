"""Paper figures (run with the analysis venv on the server; writes PDFs/PNGs to ~/lcg-cr/figs)."""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = os.path.expanduser("~/lcg-cr/runs")
OUT = os.path.expanduser("~/lcg-cr/figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
                     "savefig.dpi": 200, "legend.frameon": False})

SAMPLER_ORDER = ["temp", "topk50", "topp0.95", "topp0.9", "minp0.05", "minp0.1", "toph0.4", "pless", "nsigma1", "mink3"]
LABEL = {"temp": "pure sampling", "topk50": "top-$k$ 50", "topp0.9": "top-$p$ 0.9", "topp0.95": "top-$p$ 0.95",
         "minp0.05": "min-$p$ 0.05", "minp0.1": "min-$p$ 0.1", "toph0.4": "Top-H 0.4", "pless": "P-less",
         "nsigma1": "top-$n\\sigma$ 1", "mink3": "min-$k$", "greedy": "greedy"}
COL = {"temp": "#566573", "topk50": "#b9770e", "topp0.9": "#a93226", "topp0.95": "#e74c3c", "minp0.05": "#5dade2",
       "minp0.1": "#2874a6", "toph0.4": "#7d3c98", "pless": "#117a65", "nsigma1": "#1f618d", "mink3": "#45b39d",
       "greedy": "#000000"}


def load(exp, models=None):
    rows = []
    for f in sorted(glob.glob(f"{RUNS}/{exp}/*/metrics.jsonl")):
        model = f.split("/")[-2]
        if models and model not in models:
            continue
        ev = {}
        evf = f.replace("metrics.jsonl", "evalppl_q35a3b.jsonl")
        if os.path.exists(evf):
            for l in open(evf):
                e = json.loads(l)
                ev[e["uid"]] = e["win_nll"]
        for l in open(f):
            r = json.loads(l)
            r["model"] = model
            r["evnll"] = ev.get(r["uid"])
            rows.append(r)
    return pd.DataFrame(rows)


def load_judge(exp, judge="gptoss", dense=False):
    """uid -> {"win": {start: scores}, "global": scores}"""
    out = {}
    for f in glob.glob(f"{RUNS}/{exp}/*/judge_{judge}/{'window_dense' if dense else 'window'}.jsonl"):
        model = f.split("/")[-3]
        for l in open(f):
            r = json.loads(l)
            out.setdefault((model, r["uid"]), {"win": {}, "global": None})["win"][r["start"]] = r["scores"]
    for f in glob.glob(f"{RUNS}/{exp}/*/judge_{judge}/global.jsonl"):
        model = f.split("/")[-3]
        for l in open(f):
            r = json.loads(l)
            out.setdefault((model, r["uid"]), {"win": {}, "global": None})["global"] = r["scores"]
    return out


def outcome_judge(r, J, thr=4):
    """loop (terminal repetition) / degenerate (judge coherence of the last window < thr) / healthy."""
    lo = r["loop_onset_win"]
    if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
        return "loop"
    j = J.get((r["model"], r["uid"]))
    if not j or not j["win"]:
        return None
    last = j["win"][max(j["win"])] or {}
    c = last.get("coherence")
    if c is None:
        return None
    return "healthy" if c >= thr else "degenerate"


def outcome(r, garbage_nll=5.0, garbage_surpr=5.0):
    """loop / degenerate (incoherent) / healthy, judged on the last quarter."""
    if r["loop_onset_win"] is not None and not (isinstance(r["loop_onset_win"], float) and np.isnan(r["loop_onset_win"])):
        return "loop"
    w = r["win"]
    n = len(w["rep32"])
    q4 = slice(3 * n // 4, n)
    if r.get("evnll") is not None:
        v = np.array([np.nan if x is None else x for x in r["evnll"][q4]], float)
        if np.nanmean(v) > garbage_nll:
            return "degenerate"
    else:
        s = np.array([np.nan if x is None else x for x in w["surpr"][q4]], float)
        if np.nanmean(s) > garbage_surpr:
            return "degenerate"
    if np.nanmean(np.array(w["rep32"][q4], float)) > 0.3:
        return "degenerate"
    return "healthy"


def fig_phase(exp="core32k", models=None, name="fig_phase", use_judge=True):
    df = load(exp, models)
    J = load_judge(exp) if use_judge else {}
    if J:
        df["out"] = df.apply(lambda r: outcome_judge(r, J), axis=1)
        df = df[df.out.notna()]
        print(f"judge-based outcomes for {len(df)} generations")
    else:
        df["out"] = df.apply(outcome, axis=1)
    samplers = [s for s in SAMPLER_ORDER if s in set(df.sampler)]
    temps = sorted(t for t in df["T"].unique() if t > 0)
    fig, axes = plt.subplots(1, len(samplers), figsize=(1.45 * len(samplers), 2.6), sharey=True)
    for ax, s in zip(axes, samplers):
        sub = df[df.sampler == s]
        fr = sub.groupby("T").out.value_counts(normalize=True).unstack().reindex(temps).fillna(0)
        bottom = np.zeros(len(temps))
        for o, c in [("loop", "#a93226"), ("degenerate", "#d4ac0d"), ("healthy", "#117a65")]:
            v = fr[o].values if o in fr else np.zeros(len(temps))
            ax.bar(range(len(temps)), v, bottom=bottom, color=c, width=0.85, label=o)
            bottom += v
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([f"{t:g}" for t in temps], fontsize=7, rotation=90)
        ax.set_title(LABEL.get(s, s), fontsize=9)
        ax.grid(False)
    axes[0].set_ylabel("fraction of generations")
    fig.supxlabel("temperature", fontsize=10, y=-0.02)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    tab = df.groupby(["sampler", "T"]).out.value_counts(normalize=True).unstack().fillna(0).round(3)
    tab.to_csv(f"{OUT}/{name}.csv")
    print(tab.to_string())


def fig_trajectories(exp="core32k", models=None, cfgs=("topp0.9@T1.0", "minp0.1@T1.6", "toph0.4@T1.6", "pless@T1.0",
                                                       "nsigma1@T1.0", "topk50@T1.3"), name="fig_traj"):
    df = load(exp, models)
    df["cfg"] = df.apply(lambda r: "greedy" if r["sampler"] == "greedy" else f"{r['sampler']}@T{r['T']}", axis=1)
    keys = [("rep32", "32-gram repetition"), ("H1", "model entropy $H(p)$ (nats)"),
            ("single_frac", "singleton candidate set"), ("surpr", "surprisal of sampled token")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.6))
    for ax, (k, lab) in zip(axes, keys):
        for c in cfgs:
            sub = df[df.cfg == c]
            if not len(sub) or k not in sub.iloc[0]["win"]:
                continue
            rows_ = [[np.nan if x is None else x for x in w[k]] for w in sub.win]
            L_ = max(len(r_) for r_ in rows_)
            M = np.array([r_ + [np.nan] * (L_ - len(r_)) for r_ in rows_], float)
            mu = np.nanmean(M, 0)
            s = c.split("@")[0]
            ax.plot(np.arange(len(mu)) + 0.5, mu, color=COL.get(s, "k"), label=f"{LABEL.get(s, s)}, {c.split('@')[1]}", lw=1.6)
        ax.set_xlabel("generated tokens (K)")
        ax.set_title(lab, fontsize=9)
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_end_alignment(exp="core32k", models=None, name="fig_end"):
    df = load(exp, models)
    gaps, ends, loops = [], [], []
    for _, r in df.iterrows():
        pe = np.array([np.nan if x is None else x for x in r["win"].get("pend", [])], float)
        if not len(pe) or np.isnan(pe).all():
            continue
        hi = np.where(pe > 0.5)[0]
        lo = r["loop_onset_win"]
        if len(hi):
            ends.append(hi[0])
        if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
            loops.append(lo)
            if len(hi):
                gaps.append(lo - hi[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    axes[0].hist(ends, bins=np.arange(0, 33), alpha=0.6, color="#2874a6", label="first end spike")
    axes[0].hist(loops, bins=np.arange(0, 33), alpha=0.6, color="#a93226", label="loop onset")
    axes[0].set_xlabel("position (K tokens)"); axes[0].legend(fontsize=8)
    axes[1].hist(gaps, bins=np.arange(-16, 25), color="#7d3c98")
    axes[1].axvline(0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("loop onset $-$ first end spike (K tokens)")
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    g = np.array(gaps)
    print(f"n_loops={len(loops)} with_end={len(g)} frac_after={np.mean(g >= 0):.2f} median_gap={np.median(g):.1f}")




PARAMS = {"qwen3.5-0.8b": 0.9, "qwen3.5-2b": 2.3, "qwen3.5-4b": 4.7, "qwen3.5-9b": 9.7, "qwen3.5-27b": 27.8,
          "qwen3.5-35b-a3b": 36.0, "qwen3.5-122b-a10b": 125.0}


def escape_T(sub):
    """Lowest temperature at which the loop rate drops below 0.5 (linear interpolation); inf if never."""
    g = sub.groupby("T").looped.mean().sort_index()
    ts, ls = g.index.values, g.values
    for i in range(len(ts)):
        if ls[i] < 0.5:
            if i == 0:
                return ts[0]
            t0, t1, l0, l1 = ts[i - 1], ts[i], ls[i - 1], ls[i]
            return t0 + (l0 - 0.5) / (l0 - l1) * (t1 - t0)
    return np.inf


def fig_scale(exp="core32k", name="fig_scale"):
    df = load(exp, list(PARAMS))
    df = df[df["T"] > 0]
    df["looped"] = df.loop_onset_win.notna()
    df["out"] = df.apply(outcome, axis=1)
    rows = []
    for (m, s), sub in df.groupby(["model", "sampler"]):
        hf = sub.groupby("T").out.apply(lambda x: (x == "healthy").mean())
        rows.append({"model": m, "params": PARAMS[m], "sampler": s, "T_escape": escape_T(sub),
                     "best_healthy": hf.max(), "T_best": hf.idxmax()})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/{name}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for s in [x for x in SAMPLER_ORDER if x in set(t.sampler)]:
        sub = t[t.sampler == s].sort_values("params")
        y = sub.T_escape.replace(np.inf, 2.3)
        axes[0].plot(sub.params, y, "o-", color=COL.get(s), label=LABEL.get(s, s), lw=1.4, ms=3)
        axes[1].plot(sub.params, sub.best_healthy, "o-", color=COL.get(s), lw=1.4, ms=3)
    axes[0].axhline(2.3, color="k", lw=0.6, ls=":")
    axes[0].text(1.0, 2.33, "never escapes (T$\\leq$2)", fontsize=7)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("parameters (B)")
    axes[0].set_ylabel("escape temperature"); axes[1].set_ylabel("best healthy fraction over T")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    print(t.pivot_table(index="sampler", columns="model", values="T_escape").round(2).to_string())
    print(t.pivot_table(index="sampler", columns="model", values="best_healthy").round(2).to_string())


def fig_loop(name="fig_loop"):
    c = pd.read_parquet(os.path.expanduser("~/lcg-cr/results/loop_cells.parquet"))
    samplers = [x for x in SAMPLER_ORDER if x in set(c.sampler)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7))
    a = c[c["T"] == 1.0].groupby(["sampler", "k"]).stay.mean().unstack()
    for s_ in samplers:
        if s_ in a.index:
            axes[0].plot(a.columns, a.loc[s_], "o-", color=COL[s_], label=LABEL[s_], ms=3, lw=1.4)
    g = c[c.sampler == "greedy"].groupby("k").stay.mean()
    axes[0].plot(g.index, g.values, "k--", lw=1, label="greedy")
    axes[0].set_xscale("symlog", linthresh=1); axes[0].set_xticks([0, 1, 2, 4, 8, 16]); axes[0].set_xticklabels([0, 1, 2, 4, 8, 16])
    axes[0].set_xlabel("repetitions $k$ in context"); axes[0].set_ylabel("P(repeat again)"); axes[0].set_title("T = 1.0", fontsize=9)
    b = c[(c.k == 8) & (c["T"] > 0)].groupby(["sampler", "T"]).stay.mean().unstack()
    for s_ in samplers:
        if s_ in b.index:
            axes[1].plot(b.columns, b.loc[s_], "o-", color=COL[s_], ms=3, lw=1.4)
    axes[1].set_xlabel("temperature"); axes[1].set_title("$k$ = 8", fontsize=9)
    d = c[(c.k == 8) & (c["T"] > 0)].groupby(["sampler", "T"]).singleton0.mean().unstack()
    for s_ in samplers:
        if s_ in d.index:
            axes[2].plot(d.columns, d.loc[s_], "o-", color=COL[s_], ms=3, lw=1.4)
    axes[2].set_xlabel("temperature"); axes[2].set_title("$k$ = 8: P(singleton candidate set)", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_theory(name="fig_theory", V=150000, seed=0):
    """Collapse regions in the (T, Delta) plane for a synthetic LM-like logit vector:
    z_(1) = z_(2) + Delta, z_(2) = 0, tail ~ N(-16, 2^2) (truncated below z_(2)); tail mass ~0.1 at Delta=0, T=1."""
    import torch
    os.environ["LCS_NO_COMPILE"] = "1"
    from .samplers import process_rows
    g = np.random.default_rng(seed)
    tail = np.minimum(g.normal(-16, 2, V - 2), -0.5).astype(np.float32)
    deltas = np.linspace(0.0, 12.0, 241)
    temps = np.linspace(0.5, 3.0, 126)
    rules = [("top_p", 0.9, "top-$p$ 0.9"), ("min_p", 0.1, "min-$p$ 0.1"), ("p_less", 0.0, "P-less"),
             ("top_h", 0.4, "Top-H 0.4"), ("top_nsigma", 1.0, "top-$n\\sigma$ 1"), ("eta", 9e-4, "$\\eta$ 9e-4")]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    cols = ["#a93226", "#2874a6", "#117a65", "#7d3c98", "#1f618d", "#b9770e"]
    for (m, a, lab), c in zip(rules, cols):
        Z = np.zeros((len(deltas), len(temps)), bool)
        for i, d in enumerate(deltas):
            z = torch.tensor(np.concatenate([[d, 0.0], tail]))[None, :].repeat(len(temps), 1)
            T = torch.tensor(temps, dtype=torch.float32)
            final, *_ = process_rows(z.float(), T, [(m, torch.full((len(temps),), a))], False, False)
            Z[i] = torch.isfinite(final).sum(-1).numpy() <= 1
        ax.contour(temps, deltas, Z.astype(float), levels=[0.5], colors=[c], linewidths=1.6)
        ax.plot([], [], color=c, label=lab)
    ax.set_xlabel("temperature $T$"); ax.set_ylabel("top-2 logit gap $\\Delta$")
    ax.set_title("collapse (singleton set) above each curve", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def human_band(model):
    """Per-1K-window mean surprisal / log-rank / argmax-fraction of the human continuation under `model`."""
    out = {"surpr": [], "logrank": [], "argmax": []}
    for f in sorted(glob.glob(f"{RUNS}/human/{model}/*.npz")):
        d = np.load(f)
        lp, rk = d["lp"], d["rank"]
        n = len(lp) // 1024
        out["surpr"].append([-np.nanmean(lp[i * 1024:(i + 1) * 1024]) for i in range(n)])
        r = rk.astype(float)
        r[r <= 0] = np.nan
        out["logrank"].append([np.nanmean(np.log(r[i * 1024:(i + 1) * 1024])) for i in range(n)])
        out["argmax"].append([np.nanmean(r[i * 1024:(i + 1) * 1024] == 1) for i in range(n)])
    return {k: np.nanmean(np.array(v), 0) if v else None for k, v in out.items()}


def fig_book(models=("qwen3.5-4b-base", "qwen3.5-35b-a3b-base"), cfgs=("topp0.9@T1.0", "minp0.1@T1.0", "pless@T1.0",
             "nsigma1@T1.0", "toph0.4@T1.0", "temp@T1.0", "topk50@T1.0"), name="fig_book"):
    keys = [("surpr", "surprisal (nats)"), ("logrank_mean", "mean log-rank"), ("argmax_frac", "argmax fraction")]
    hk = {"surpr": "surpr", "logrank_mean": "logrank", "argmax_frac": "argmax"}
    fig, axes = plt.subplots(len(models), len(keys), figsize=(3.3 * len(keys), 2.5 * len(models)), squeeze=False)
    for mi, model in enumerate(models):
        df = load("book32k", [model])
        if not len(df):
            continue
        df["cfg"] = df.apply(lambda r: "greedy" if r["sampler"] == "greedy" else f"{r['sampler']}@T{r['T']}", axis=1)
        hb = human_band(model)
        for ki, (k, lab) in enumerate(keys):
            ax = axes[mi, ki]
            for c in cfgs:
                sub = df[df.cfg == c]
                if not len(sub):
                    continue
                M = np.array([[np.nan if x is None else x for x in w[k]] for w in sub.win], float)
                s_ = c.split("@")[0]
                ax.plot(np.arange(M.shape[1]) + 0.5, np.nanmean(M, 0), color=COL.get(s_, "k"), lw=1.4,
                        label=f"{LABEL.get(s_, s_)} {c.split('@')[1]}")
            if hb.get(hk[k]) is not None:
                ax.plot(np.arange(len(hb[hk[k]])) + 0.5, hb[hk[k]], "k--", lw=1.6, label="human continuation")
            if mi == 0:
                ax.set_title(lab, fontsize=9)
            if mi == len(models) - 1:
                ax.set_xlabel("generated tokens (K)")
        axes[mi, 0].set_ylabel(model.replace("-base", " (base)"), fontsize=9)
    fig.subplots_adjust(hspace=0.25, wspace=0.28)
    axes[0, -1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_anatomy(run=f"{RUNS}/core32k/qwen3.5-9b", uid="fic_fantasy_0__topp0.9@T1.0__s0", name="fig_anatomy", smooth=256):
    from .metrics import ngram_rep
    toks = np.load(f"{run}/gen/{uid}.npz")["tokens"].astype(np.int64)
    L = len(toks)
    st = np.full((L, 7), np.nan)
    for p in sorted(glob.glob(f"{run}/stats/{uid}.p*.npz")):
        d = np.load(p)
        pos, S = d["pos"], d["stats"]
        st[pos[pos < L], :S.shape[1]] = S[pos < L]
    rep = ngram_rep(toks, 32).astype(float)
    k = np.ones(smooth) / smooth
    sm = lambda x: np.convolve(np.nan_to_num(x), k, mode="same")
    x = np.arange(L) / 1000
    fig, axes = plt.subplots(4, 1, figsize=(7, 4.6), sharex=True)
    axes[0].plot(x, np.nan_to_num(st[:, 6]), color="#2874a6", lw=0.5); axes[0].set_ylabel("$p_{end}$", fontsize=8)
    axes[0].set_ylim(0, 1.02)
    axes[1].plot(x, sm(st[:, 0]), color="#7d3c98", lw=1.2); axes[1].set_ylabel("$H(p)$", fontsize=8)
    axes[2].plot(x, sm((st[:, 3] <= 1).astype(float)), color="#117a65", lw=1.2); axes[2].set_ylabel("singleton", fontsize=8)
    axes[3].plot(x, sm(rep), color="#a93226", lw=1.2); axes[3].set_ylabel("rep$_{32}$", fontsize=8)
    axes[3].set_xlabel("generated tokens (K)")
    hi = np.where(st[:, 6] > 0.5)[0]
    for ax in axes:
        if len(hi):
            ax.axvline(hi[0] / 1000, color="k", ls=":", lw=0.8)
    axes[0].set_title(f"{os.path.basename(run)}, {uid.split('__')[1]}", fontsize=9)
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_temp_robust(exp="core32k", models=None, name="fig_temp_robust"):
    """(a) judged quality of the FIRST window vs T (short-range temperature robustness);
    (b) judged quality of the LAST window vs T; (c) terminal-loop rate vs T."""
    df = load(exp, models)
    J = load_judge(exp)
    def wq(r, which):
        j = J.get((r["model"], r["uid"]))
        if not j or not j["win"]:
            return np.nan
        k = min(j["win"]) if which == "first" else max(j["win"])
        return (j["win"][k] or {}).get("overall", np.nan)
    df["q_first"] = df.apply(lambda r: wq(r, "first"), axis=1)
    df["q_last"] = df.apply(lambda r: wq(r, "last"), axis=1)
    df["looped"] = df.loop_onset_win.notna()
    df = df[df["T"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7))
    for s_ in [x for x in SAMPLER_ORDER if x in set(df.sampler)]:
        sub = df[df.sampler == s_].groupby("T")
        for ax, col in zip(axes, ["q_first", "q_last", "looped"]):
            v = sub[col].mean()
            ax.plot(v.index, v.values, "o-", color=COL[s_], label=LABEL[s_], ms=3, lw=1.4)
    axes[0].set_title("judged quality, first 1.5K tokens", fontsize=9)
    axes[1].set_title("judged quality, last 1.5K tokens", fontsize=9)
    axes[2].set_title("terminal-loop rate", fontsize=9)
    for ax in axes:
        ax.set_xlabel("temperature")
    axes[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    tab = df.groupby(["sampler", "T"])[["q_first", "q_last", "looped"]].mean().round(2)
    tab.to_csv(f"{OUT}/{name}.csv")


def failure_window(r, nll_hi=5.5):
    """First 1K window at which the generation has failed: terminal loop onset, or the first of two consecutive
    windows with evaluator NLL > nll_hi (incoherence).  Returns n_windows if it never fails."""
    nw = r["n_windows"]
    lo = r["loop_onset_win"]
    lo = nw if lo is None or (isinstance(lo, float) and np.isnan(lo)) else int(lo)
    inc = nw
    ev = r.get("evnll")
    if ev:
        v = np.array([np.nan if x is None else x for x in ev], float)
        for i in range(len(v) - 1):
            if v[i] > nll_hi and v[i + 1] > nll_hi:
                inc = i
                break
    return min(lo, inc), ("loop" if lo <= inc and lo < nw else ("incoherent" if inc < nw else "survived"))


def fig_survival(exp="core32k", models=None, temps=(1.0, 1.6), name="fig_survival"):
    df = load(exp, models)
    res = df.apply(lambda r: failure_window(r), axis=1)
    df["fail_win"] = [x[0] for x in res]
    df["fail_mode"] = [x[1] for x in res]
    fig, axes = plt.subplots(1, len(temps), figsize=(4.2 * len(temps), 2.8), squeeze=False)
    rows = []
    for ax, T in zip(axes[0], temps):
        for s_ in [x for x in SAMPLER_ORDER if x in set(df.sampler)]:
            sub = df[(df.sampler == s_) & (df["T"] == T)]
            if not len(sub):
                continue
            nw = int(sub.n_windows.max())
            surv = [(sub.fail_win > w).mean() for w in range(nw + 1)]
            ax.step(range(nw + 1), surv, where="post", color=COL[s_], label=LABEL[s_], lw=1.4)
            rows.append({"sampler": s_, "T": T, "median_fail_k": float(np.median(sub.fail_win)),
                         "survive_8k": float((sub.fail_win >= 8).mean()), "survive_16k": float((sub.fail_win >= 16).mean()),
                         "survive_32k": float((sub.fail_win >= nw).mean()),
                         "loop_frac": float((sub.fail_mode == "loop").mean()),
                         "incoh_frac": float((sub.fail_mode == "incoherent").mean())})
        ax.set_title(f"T = {T}", fontsize=9); ax.set_xlabel("generated tokens (K)"); ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("fraction not yet failed")
    axes[0][-1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/{name}.csv", index=False)
    print(t.round(2).to_string(index=False))


def fig_natural(exp="mt32k", models=("qwen3.5-4b", "qwen3.5-9b"), name="fig_natural"):
    """Per-sampler curves vs temperature under a natural protocol: loop rate, healthy fraction, global judged quality."""
    df = load(exp, list(models))
    J = load_judge(exp)
    df["looped"] = df.loop_onset_win.notna()
    df["healthy"] = df.apply(lambda r: outcome_judge(r, J), axis=1) == "healthy"
    df["gov"] = df.apply(lambda r: ((J.get((r["model"], r["uid"])) or {}).get("global") or {}).get("overall", np.nan), axis=1)
    df = df[df["T"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7))
    for s_ in [x for x in SAMPLER_ORDER if x in set(df.sampler)]:
        g = df[df.sampler == s_].groupby("T")
        for ax, col in zip(axes, ["looped", "healthy", "gov"]):
            v = g[col].mean()
            ax.plot(v.index, v.values, "o-", color=COL[s_], label=LABEL[s_], ms=3, lw=1.4)
    for ax, t in zip(axes, ["terminal-loop rate", "healthy at 32K (no loop, coherent end)", "global judged quality (1-10)"]):
        ax.set_title(t, fontsize=9); ax.set_xlabel("temperature")
    axes[2].axhline(4.12, color="k", ls=":", lw=0.8)
    axes[2].text(0.72, 4.2, "human book text", fontsize=7)
    axes[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    tab = df.groupby(["sampler", "T"])[["looped", "healthy", "gov"]].mean().round(3)
    tab.to_csv(f"{OUT}/{name}.csv")


def fig_niah(models=None, name="fig_niah"):
    c = pd.read_parquet(os.path.expanduser("~/lcg-cr/results/niah_cells.parquet"))
    g = pd.read_parquet(os.path.expanduser("~/lcg-cr/results/niah_gen.parquet"))
    if models:
        c = c[c.model.isin(models)]; g = g[g.model.isin(models)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for ax, T in zip(axes[:2], (1.0, 1.5)):
        sub = c[(c["T"] == T) | (c.sampler == "greedy")]
        for s_ in ["greedy"] + [x for x in SAMPLER_ORDER if x in set(sub.sampler)]:
            v = sub[sub.sampler == s_].groupby("L").acc.mean()
            ax.plot(v.index / 1000, v.values, "o-" if s_ != "greedy" else "k--", color=COL.get(s_, "k"),
                    label=LABEL.get(s_, s_), ms=3, lw=1.4)
        ax.set_xscale("log"); ax.set_xlabel("context length (K tokens)"); ax.set_title(f"retrieval accuracy, T = {T}", fontsize=9)
    # accuracy relative to greedy vs the model's answer-token entropy, per (model, length) cell, T = 1.0
    ent = g[g.sampler == "greedy"].groupby(["model", "L"]).H1_first.mean()
    base = c[c.sampler == "greedy"].set_index(["model", "L"]).acc
    for s_ in ["temp", "topp0.9", "topk50", "minp0.1", "toph0.4", "nsigma1", "pless"]:
        sub = c[(c.sampler == s_) & (c["T"] == 1.0)].set_index(["model", "L"])
        idx = sub.index.intersection(ent.index).intersection(base.index)
        if not len(idx):
            continue
        x, y = ent.loc[idx].values, (sub.loc[idx].acc - base.loc[idx]).values
        o = np.argsort(x)
        axes[2].plot(x[o], y[o], "o-", color=COL[s_], ms=3, lw=1.0, label=LABEL[s_])
    axes[2].axhline(0, color="k", lw=0.6, ls=":")
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set_xlabel("model entropy at answer token (nats)"); axes[2].set_ylabel("accuracy $-$ greedy")
    axes[2].set_title("T = 1.0: loss grows with answer entropy", fontsize=9)
    axes[1].legend(fontsize=6.5, loc="lower left")
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_reason(name="fig_reason", panels=(("r1-distill-qwen-1.5b", "math5"), ("qwen3.5-2b", "math5"), ("qwen3.5-4b", "aime"),
                                          ("qwen3.5-4b", "math5"), ("nemotron3-4b", "aime"))):
    """Long-CoT reasoning (32K budget): accuracy (top) and terminal-loop rate (bottom) vs temperature, per model/set."""
    g = pd.read_parquet(os.path.expanduser("~/lcg-cr/results/reason_gen.parquet"))
    panels = [p for p in panels if len(g[(g.model == p[0]) & (g.set == p[1])])]
    fig, axes = plt.subplots(2, len(panels), figsize=(2.5 * len(panels), 3.5), sharex=True, squeeze=False)
    fig.subplots_adjust(wspace=0.38)
    rules = ["temp", "topp0.95", "minp0.1", "toph0.4", "pless", "nsigma1"]
    for j, (m, st) in enumerate(panels):
        x = g[(g.model == m) & (g.set == st)]
        gr = x[x.sampler == "greedy"]
        for s_ in rules:
            c = x[x.sampler == s_].groupby("T").agg(acc=("correct", "mean"), loop=("looped", "mean"))
            if not len(c):
                continue
            axes[0, j].plot(c.index, c.acc, "o-", color=COL[s_], label=LABEL[s_], ms=3, lw=1.3)
            axes[1, j].plot(c.index, c.loop, "o-", color=COL[s_], ms=3, lw=1.3)
        if len(gr):
            axes[0, j].axhline(gr.correct.mean(), color="k", ls=":", lw=0.8)
            axes[1, j].axhline(gr.looped.mean(), color="k", ls=":", lw=0.8)
        axes[0, j].set_title(f"{m} / {'AIME' if st == 'aime' else 'MATH L5'}", fontsize=8)
        axes[1, j].set_xlabel("temperature")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("terminal-loop rate")
    axes[0, -1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_long(name="fig_long", runs=(("long256k", "qwen3.5-4b"), ("long256k", "qwen3.5-9b"), ("long1000k", "qwen3.5-0.8b"),
                                   ("long1000k", "nemotron3-30b-a3b")),
             pid="fic_scifi_0", smooth=4):
    """Million-token generation: 32-gram repetition and model entropy per 1K-token window, one row per model/length."""
    cfgs = ["topp0.9@T1.0", "minp0.1@T1.0", "pless@T1.0", "nsigma1@T1.0", "minp0.1@T1.6", "toph0.4@T1.6", "nsigma1+k2@T1.3"]
    cols = {"topp0.9@T1.0": COL["topp0.9"], "minp0.1@T1.0": "#85c1e9", "pless@T1.0": COL["pless"],
            "nsigma1@T1.0": COL["nsigma1"], "minp0.1@T1.6": COL["minp0.1"], "toph0.4@T1.6": COL["toph0.4"],
            "nsigma1+k2@T1.3": "#e67e22"}
    labs = {c: (LABEL.get(c.split("@")[0].replace("+k2", ""), c) + (" +keep2" if "+k2" in c else "") + f", $T{{=}}{c.split('@T')[1]}$")
            for c in cfgs}
    runs = [r for r in runs if os.path.exists(f"{RUNS}/{r[0]}/{r[1]}/metrics.jsonl")]
    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 2.2 * len(runs)), squeeze=False)
    k = np.ones(smooth) / smooth
    for i, (exp, m) in enumerate(runs):
        rows = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{RUNS}/{exp}/{m}/metrics.jsonl")}
        for c in cfgs:
            r = rows.get(f"{pid}__{c}")
            if not r:
                continue
            x = (np.arange(len(r["win"]["rep32"])) + 0.5) * 1024 / 1000
            for j, key in enumerate(["rep32", "H1"]):
                y = np.array([np.nan if v is None else v for v in r["win"][key]], float)
                y = np.convolve(np.nan_to_num(y), k, mode="same") if smooth > 1 else y
                axes[i, j].plot(x, y, color=cols[c], lw=1.0, label=labs[c])
        axes[i, 0].set_ylabel(f"{m}\n32-gram repetition", fontsize=8)
        axes[i, 1].set_ylabel("model entropy (nats)", fontsize=8)
        for j in range(2):
            axes[i, j].set_xscale("log")
    for j in range(2):
        axes[-1, j].set_xlabel("position (K tokens)")
    axes[0, 1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_cheatsheet(name="fig_cheatsheet", V=150000, seed=0, show=12):
    """What each truncation rule keeps: an ordinary step vs. a confident (looping) step, and the confident step at T=1.6.
    Masks are computed with the exact sampler implementation used in all experiments (lcs.samplers)."""
    import torch
    from .samplers import process_rows
    rng = np.random.default_rng(seed)
    tail = rng.normal(-19.0, 2.5, V - show).astype(np.float32)   # most of the vocabulary sits far below the top
    ordinary = np.concatenate([np.array([0, -0.4, -0.7, -1.0, -1.3, -1.6, -1.9, -2.1, -2.4, -2.6, -2.8, -3.0], np.float32), tail])
    confident = np.concatenate([np.array([0, -3.5, -4.8, -5.4, -6.0, -6.4, -6.8, -7.2, -7.5, -7.8, -8.1, -8.3], np.float32), tail])
    rules = [("top-$p$ 0.9", [("top_p", 0.9)]), ("min-$p$ 0.1", [("min_p", 0.1)]),
             ("P-less", [("p_less", 0.0)]), ("top-$n\\sigma$ 1", [("top_nsigma", 1.0)]),
             ("P-less + keep-top-2", [("p_less", 0.0), ("keep_top", 2)])]
    cols = [("ordinary step, $T{=}1$", ordinary, 1.0), ("confident step, $T{=}1$", confident, 1.0),
            ("confident step, $T{=}1.6$", confident, 1.6)]
    fig, axes = plt.subplots(len(rules), len(cols), figsize=(7.0, 0.78 * len(rules) + 0.5), sharey="col", squeeze=False)
    for j, (title, z, T) in enumerate(cols):
        zt = torch.tensor(z)[None]
        p = torch.softmax(zt / T, -1)[0].numpy()
        for i, (lab, ops) in enumerate(rules):
            opt = [(m, torch.tensor([float(v)])) for m, v in ops]
            _, _, _, _, keep = process_rows(zt.clone(), torch.tensor([T]), opt, False, False)
            keep = keep[0].numpy()
            ax = axes[i, j]
            colors = ["#1f618d" if keep[t] else "#d5d8dc" for t in range(show)]
            ax.bar(np.arange(show), p[:show], color=colors, width=0.8)
            n_kept = int(keep.sum())
            ax.text(0.98, 0.9, f"keeps {n_kept:,} token" + ("s" if n_kept != 1 else ""), transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color="#a93226" if n_kept == 1 else "#1f3a5f",
                    fontweight="bold" if n_kept == 1 else "normal")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ("left", "top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
            if i == 0:
                ax.set_title(title, fontsize=8.5)
    fig.text(0.5, 0.005, "the 12 most likely next tokens (bar height = probability; blue = may be sampled, grey = cut)",
             ha="center", fontsize=7)
    fig.subplots_adjust(hspace=0.35, wspace=0.08, left=0.2, bottom=0.06)
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


def fig_phasemap(name="fig_phasemap", forced_models=("qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"),
                 natural_models=("qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b",
                                 "nemotron3-30b-a3b")):
    """Outcome phase map: sampler x temperature, color = mix of loop (red) / noise (purple) / healthy (green)."""
    from matplotlib.colors import to_rgb
    rules = ["temp", "topk50", "topp0.95", "topp0.9", "minp0.05", "minp0.1", "toph0.4", "pless", "nsigma1",
             "toph0.4+k2", "pless+k2", "nsigma1+k2"]
    temps = [0.7, 1.0, 1.3, 1.6, 2.0]
    cl = {"loop": np.array(to_rgb("#c0392b")), "degenerate": np.array(to_rgb("#8e44ad")), "healthy": np.array(to_rgb("#27ae60"))}

    def table(exps, models):
        frames = []
        for e in exps:
            J = load_judge(e)
            d = load(e, list(models))
            if not len(d):
                continue
            d = d[d["T"] > 0].copy()
            d["out"] = d.apply(lambda r: outcome_judge(r, J), axis=1)
            frames.append(d[d.out.notna()])
        d = pd.concat(frames)
        return d.groupby(["sampler", "T"]).out.value_counts(normalize=True).unstack(fill_value=0)

    panels = [("Forced 32K generation (end-of-turn banned)", table(["core32k", "k2_32k"], forced_models)),
              ('Natural multi-turn generation ("Continue.")', table(["mt32k", "mtk2_32k"], natural_models))]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.55))
    for ax, (title, tab) in zip(axes, panels):
        img = np.ones((len(rules), len(temps), 3))
        for i, r in enumerate(rules):
            for j, T in enumerate(temps):
                if (r, T) not in tab.index:
                    img[i, j] = 0.93
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.5, color="#888")
                    continue
                row = tab.loc[(r, T)]
                c = sum(row.get(k, 0) * cl[k] for k in cl)
                img[i, j] = 1 - (1 - c) * 0.85
                h = row.get("healthy", 0)
                ax.text(j, i, f"{100 * h:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if max(row.get("loop", 0), row.get("degenerate", 0), h) > 0.55 else "black",
                        fontweight="bold" if h >= 0.8 else "normal")
        ax.imshow(img, aspect="auto")
        ax.tick_params(labelsize=6.5)
        ax.set_xticks(range(len(temps)), [str(t) for t in temps])
        ax.set_yticks(range(len(rules)), [LABEL.get(r.replace("+k2", ""), r) + (" + keep-2" if "+k2" in r else "") for r in rules])
        ax.set_xlabel("temperature $T$", fontsize=7.5)
        ax.set_title(title, fontsize=8)
        ax.axhline(8.5, color="white", lw=3)
        ax.grid(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cl["loop"], label="terminal loop"), Patch(color=cl["degenerate"], label="incoherent (noise)"),
                        Patch(color=cl["healthy"], label="healthy")], loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.955, "numbers: % of generations that are healthy at 32K tokens", ha="center", fontsize=6.5, color="#444")
    fig.subplots_adjust(wspace=0.45, bottom=0.14)
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    if what in ("phase", "all"):
        fig_phase(models=models)
    if what in ("traj", "all"):
        fig_trajectories(models=models)
    if what in ("end", "all"):
        fig_end_alignment(models=models)
    if what in ("scale", "all"):
        fig_scale()
    if what in ("loop", "all"):
        fig_loop()
    if what == "theory":
        fig_theory()
    if what == "book":
        fig_book()
    if what == "anatomy":
        fig_anatomy()
    if what == "temprobust":
        fig_temp_robust(models=models)
    if what == "natural":
        fig_natural()
    if what == "niah":
        fig_niah()
    if what == "phasemap":
        fig_phasemap()
    if what == "cheatsheet":
        fig_cheatsheet()
    if what == "long":
        fig_long()
    if what == "reason":
        fig_reason()
    if what == "survival":
        fig_survival(models=models, temps=(0.7, 1.0, 1.3, 1.6))
