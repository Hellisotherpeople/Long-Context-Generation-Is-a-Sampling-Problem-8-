"""Every number the paper quotes that is not produced by a table/figure script.

python -m lcs.paper_numbers > ~/lcg-cr/results/paper_numbers.txt   (also writes paper_numbers.json)
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from .figs import human_band, load, load_judge, outcome_judge

H = os.path.expanduser("~/lcg-cr")
RUNS = f"{H}/runs"
SKIP = ("_old", "_refusal", "smoke", "probe")
QL = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "llama3.2-1b"]
FAMILY = [("qwen", "Qwen"), ("r1-distill", "DeepSeek-R1-Distill"), ("gemma", "Gemma"), ("ministral", "Mistral"),
          ("mistral", "Mistral"), ("nemotron", "Nemotron"), ("llama", "Llama"), ("olmo", "OLMo"), ("glm", "GLM"),
          ("human", None)]
OUT = {}


def fam(name):
    for k, v in FAMILY:
        if name.startswith(k):
            return v
    return name


def counts():
    gens = toks = 0
    models, per_exp = set(), collections.Counter()
    tok_exp = collections.Counter()
    for run in glob.glob(f"{RUNS}/*/*"):
        exp, m = run.split("/")[-2], run.split("/")[-1].split("__")[0]
        if exp.startswith(SKIP) or not os.path.exists(f"{run}/meta.jsonl"):
            continue
        for line in open(f"{run}/meta.jsonl"):
            if not line.strip():
                continue
            r = json.loads(line)
            gens += 1
            toks += int(r.get("n_tokens", 0))
            per_exp[exp] += 1
            tok_exp[exp] += int(r.get("n_tokens", 0))
            if not m.startswith("human"):
                models.add(os.path.basename(r.get("model", m)))
    fams = sorted({fam(os.path.basename(x).split("__", 1)[-1].lower()) for x in models} - {None})
    OUT["n_generation_calls"] = gens
    OUT["n_generated_tokens"] = toks
    OUT["n_model_checkpoints"] = len(models)
    OUT["models"] = sorted(models)
    OUT["families"] = fams
    OUT["tokens_by_exp_M"] = {k: round(v / 1e6, 1) for k, v in sorted(tok_exp.items())}
    # GPU-hours from the scheduler log
    start, gh = {}, collections.Counter()
    pat = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d) (START|DONE|FAILED) (\S+)(?: rc=\S+)? gpus=\[([\d, ]*)\]")
    for line in open(f"{H}/logs/sched.log"):
        mm = pat.match(line.strip())
        if not mm:
            continue
        t = datetime.strptime("2026-" + mm.group(1), "%Y-%m-%d %H:%M:%S")
        name, ng = mm.group(3), len([x for x in mm.group(4).split(",") if x.strip()])
        if mm.group(2) == "START":
            start[name] = (t, ng)
        elif name in start:
            t0, n0 = start.pop(name)
            kind = name.split("_", 2)[1] if "_" in name else name
            gh[kind] += (t - t0).total_seconds() / 3600 * n0
    for name, (t0, n0) in start.items():  # still running
        gh[name.split("_", 2)[1]] += (datetime.now() - t0).total_seconds() / 3600 * n0
    OUT["gpu_hours_by_kind"] = {k: round(v, 1) for k, v in sorted(gh.items(), key=lambda t: -t[1])}
    OUT["gpu_hours_total"] = round(sum(gh.values()), 1)
    jc = 0
    for f in glob.glob(f"{RUNS}/*/*/judge_*/*.jsonl"):
        jc += sum(1 for _ in open(f))
    OUT["n_judge_calls"] = jc


def end_pressure():
    df = load("core32k")
    spikes = collections.defaultdict(list)
    for _, r in df.iterrows():
        pe = np.array([np.nan if x is None else x for x in r["win"].get("pend", [])], float)
        hi = np.where(pe > 0.5)[0]
        if len(hi):
            spikes[r["model"]].append(hi[0])
    OUT["end_spike_median_K"] = {m: float(np.median(v)) for m, v in sorted(spikes.items())}
    from .stats import claim_end_pressure
    t, med = claim_end_pressure()
    OUT["end_after_frac"] = {r["model"]: [round(r["mean"], 3), round(r["lo"], 3), round(r["hi"], 3), int(r["n"]),
                                          None if pd.isna(r.get("frac_loops_without_end")) else round(r["frac_loops_without_end"], 3)]
                             for _, r in t.iterrows()}
    OUT["end_gap_median_K"] = med


def phase():
    df = load("core32k", QL)
    J = load_judge("core32k")
    df = df[df["T"] > 0].copy()
    df["looped"] = df.loop_onset_win.notna()
    df["out"] = df.apply(lambda r: outcome_judge(r, J), axis=1)

    def wq(r, first):
        j = J.get((r["model"], r["uid"]))
        if not j or not j["win"]:
            return np.nan
        k = min(j["win"]) if first else max(j["win"])
        return (j["win"][k] or {}).get("overall", np.nan)
    df["q_first"] = df.apply(lambda r: wq(r, True), axis=1)
    OUT["phase_n_generations"] = int(len(df))
    low = df[df["T"] <= 1.0].groupby("sampler").looped.mean()
    OUT["phase_loop_rate_T<=1_by_rule"] = low.round(3).to_dict()
    cell = df[df.out.notna()].groupby(["model", "sampler", "T"]).out.apply(lambda x: (x == "healthy").mean())
    OUT["phase_max_healthy_cell"] = [round(float(cell.max()), 3), list(map(str, cell.idxmax()))]
    OUT["phase_healthy_cells_le_7pct_frac"] = round(float((cell <= 0.07).mean()), 3)
    q2 = df[df["T"] == 2.0].groupby("sampler").q_first.mean()
    OUT["phase_first_window_quality_T2"] = q2.round(2).to_dict()


def natural(models=("qwen3.5-4b", "qwen3.5-9b")):
    df = load("mt32k", list(models))
    J = load_judge("mt32k")
    df = df[df["T"] > 0].copy()
    df["out"] = df.apply(lambda r: outcome_judge(r, J), axis=1)
    df["gov"] = df.apply(lambda r: ((J.get((r["model"], r["uid"])) or {}).get("global") or {}).get("overall", np.nan), axis=1)
    df["loop"] = df.out == "loop"
    df["healthy"] = df.out == "healthy"
    g = df.groupby(["model", "sampler", "T"])[["loop", "healthy", "gov"]].mean().reset_index()
    best = g.loc[g.groupby(["model", "sampler"]).gov.idxmax()]
    OUT["natural_best"] = {f"{r['model']}|{r['sampler']}": [r["T"], round(r["healthy"], 2), round(r["gov"], 2)]
                           for _, r in best.iterrows()}
    bh = g.groupby(["model", "sampler"]).healthy.max()
    OUT["natural_best_healthy"] = {f"{m}|{s}": round(v, 2) for (m, s), v in bh.items()}
    pl = g[g.sampler == "pless"]
    OUT["natural_pless_loop_range"] = [round(pl.loop.min(), 2), round(pl.loop.max(), 2)]
    th = g[(g.sampler == "toph0.4") & (g["T"] <= 1.0)]
    OUT["natural_toph_loop_T<=1_range"] = [round(th.loop.min(), 2), round(th.loop.max(), 2)]
    hj = load_judge("book32k_human")
    vals = [s.get("overall") for j in hj.values() for s in ([j["global"]] if j["global"] else []) if s]
    OUT["human_global_quality"] = round(float(np.mean(vals)), 2) if vals else None


def book_band():
    out = {}
    for d in sorted(glob.glob(f"{RUNS}/human/*")):
        m = d.split("/")[-1]
        hb = human_band(m)
        if hb.get("surpr") is None:
            continue
        out[m] = {"surprisal": round(float(np.nanmean(hb["surpr"])), 2),
                  "geo_mean_rank": round(float(np.exp(np.nanmean(hb["logrank"]))), 2),
                  "argmax_frac": round(float(np.nanmean(hb["argmax"])), 3)}
    OUT["human_band"] = out


def main():
    for f in (counts, end_pressure, phase, natural, book_band):
        try:
            f()
        except Exception as e:  # keep going; report what failed
            OUT[f"error_{f.__name__}"] = repr(e)[:300]
    json.dump(OUT, open(f"{H}/results/paper_numbers.json", "w"), indent=1, default=str)
    for k, v in OUT.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
