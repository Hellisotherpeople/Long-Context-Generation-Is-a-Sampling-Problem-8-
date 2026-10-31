"""Hyperparameter and sampler-order ablations (user request 2026-10-11: every sampler's own parameter matters, especially
at high temperature, and so does the order in which samplers / temperature are applied).

Three experiments share one grid:
  hpdiv        quality-diversity (NoveltyBench + Infinity-Chat subset, 120 prompts x 4 samples), T in 0.7..3.0
  longproc_hp  LongProc (5 tasks, 300 items; precise long outputs), T in {1.0, 1.5, 2.0}
  mt32k_hp2    natural multi-turn 32K generation, extending mt32k_hp with T=2.0/3.0, extra values, order variants

Config names: "<rule><param>" (e.g. minp0.2, topp0.8, topk10, nsigma2, toph0.6, typical0.5, eta0.0009, eps0.0003,
pless, plessnorm, mink3), "+kN" for keep_top=N, "~tl" for temperature-last (masks computed at T=1, sampling at T),
"a>b" for chains applied in that order (each op sees the renormalised output of the previous one), "llamacpp" for the
llama.cpp default chain top-k 40 > top-p 0.95 > min-p 0.05.

python -m lcs.ablation hpdiv --models qwen3.5-9b,llama3.1-8b,gemma4-e4b --prio 054
python -m lcs.ablation longproc --models qwen3.5-9b,qwen3.5-4b --prio 055
python -m lcs.ablation mt --models qwen3.5-9b,nemotron3-30b-a3b,llama3.1-8b --prio 056
"""
from __future__ import annotations

import argparse
import copy
import json
import os

H = os.path.expanduser("~/lcg-cr")


def _g(prefix, method, vals, fmt="{:g}"):
    return [(f"{prefix}{fmt.format(v)}", [[method, v]]) for v in vals]


# temperature-first grids: every rule's own parameter
GRID = ([("temp", [])] +
        _g("topk", "top_k", [5, 10, 20, 50, 100, 500]) +
        _g("topp", "top_p", [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]) +
        _g("minp", "min_p", [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]) +
        _g("typical", "typical", [0.2, 0.5, 0.8, 0.95]) +
        _g("eta", "eta", [3e-4, 9e-4, 2e-3, 4e-3]) +
        _g("eps", "epsilon", [3e-4, 9e-4, 3e-3]) +
        _g("nsigma", "top_nsigma", [0.5, 1.0, 1.5, 2.0, 3.0]) +
        _g("toph", "top_h", [0.1, 0.2, 0.4, 0.6, 0.8]) +
        [("pless", [["p_less", 0.0]]), ("plessnorm", [["p_less_norm", 0.0]]), ("mink3", [["min_k", 3.0]])])
KEEP2 = [("pless+k2", [["p_less", 0.0], ["keep_top", 2]]), ("pless+k3", [["p_less", 0.0], ["keep_top", 3]]),
         ("nsigma1+k2", [["top_nsigma", 1.0], ["keep_top", 2]]), ("toph0.4+k2", [["top_h", 0.4], ["keep_top", 2]]),
         ("minp0.1+k2", [["min_p", 0.1], ["keep_top", 2]])]
# temperature-last variants: masks at T=1, then sample at the target temperature
TLAST = ([(f"minp{m:g}~tl", [["min_p", m]]) for m in (0.02, 0.05, 0.1, 0.2, 0.3)] +
         [(f"topp{p:g}~tl", [["top_p", p]]) for p in (0.9, 0.95)] +
         [("topk50~tl", [["top_k", 50]]), ("typical0.95~tl", [["typical", 0.95]]), ("toph0.4~tl", [["top_h", 0.4]]),
          ("pless~tl", [["p_less", 0.0]]), ("eta0.0009~tl", [["eta", 9e-4]]), ("eps0.0009~tl", [["epsilon", 9e-4]]),
          ("llamacpp~tl", [["top_k", 40], ["top_p", 0.95], ["min_p", 0.05]])])
# chain order (temperature first)
CHAINS = [("topk50>topp0.9", [["top_k", 50], ["top_p", 0.9]]), ("topp0.9>topk50", [["top_p", 0.9], ["top_k", 50]]),
          ("minp0.1>topp0.9", [["min_p", 0.1], ["top_p", 0.9]]), ("topp0.9>minp0.1", [["top_p", 0.9], ["min_p", 0.1]]),
          ("topk20>topp0.95", [["top_k", 20], ["top_p", 0.95]]), ("topp0.95>topk20", [["top_p", 0.95], ["top_k", 20]]),
          ("llamacpp", [["top_k", 40], ["top_p", 0.95], ["min_p", 0.05]]),
          ("minp0.05>topp0.95>topk40", [["min_p", 0.05], ["top_p", 0.95], ["top_k", 40]])]
ALL = GRID + KEEP2 + TLAST + CHAINS


def lcs_cfg(name, ops, T):
    """Row config; temperature-last rows truncate at T=1 and sample at T."""
    if name.endswith("~tl") or "~tl" in name:
        return {"T": 1.0, "ops": ops + [["temp_last", T]]}
    return {"T": T, "ops": ops}


def queue(jf, model_dir, tp, prio, tag, env=None):
    q = {"job": jf, "gpus": int(tp), "model_dir": model_dir, "kind": "gen"}
    if env:
        q["env"] = env
    json.dump(q, open(f"{H}/queue/pending/{prio}_{tag}.json", "w"))


def env_for(model, exps=("core32k", "mt32k")):
    env = {}
    for fn in os.listdir(f"{H}/queue/done"):
        if any(fn.endswith(f"_{e}_{model}.json") for e in exps):
            env.update(json.load(open(f"{H}/queue/done/{fn}")).get("env", {}) or {})
    return env


def build_hpdiv(models, prio, n_per_src=40, samples=4, temps=(0.7, 1.0, 1.3, 1.6, 2.0, 3.0)):
    from .exp_divx import load_prompts
    allp = load_prompts()
    prompts = []
    for src in ("novelty-curated", "novelty-wildchat", "infinity-chat"):
        prompts += [p for p in allp if p["src"] == src][:n_per_src]
    os.makedirs(f"{H}/jobs/hpdiv", exist_ok=True)
    for model in models:
        tmpl = json.load(open(f"{H}/jobs/divx/{model}.json")) if os.path.exists(f"{H}/jobs/divx/{model}.json") else None
        src_job = json.load(open(f"{H}/jobs/mt32k/{model}.json"))
        ck = src_job["requests"][0]["prompt"].get("chat_kwargs") or {}
        reqs = []
        for p in prompts:
            base = {"prompt": {"type": "chat", "messages": [{"role": "user", "content": p["prompt"]}], "chat_kwargs": ck},
                    "max_tokens": 2048, "force_len": False}
            reqs.append(dict(base, uid=f"{p['pid']}__greedy__s0", lcs={"T": 1.0, "ops": [["greedy", 0.0]]},
                             meta={"pid": p["pid"], "src": p["src"], "sampler": "greedy", "T": 0.0, "seed": 0}))
            for name, ops in ALL:
                for T in temps:
                    for sd in range(samples):
                        reqs.append(dict(base, uid=f"{p['pid']}__{name}@T{T}__s{sd}", lcs=lcs_cfg(name, ops, T),
                                         meta={"pid": p["pid"], "src": p["src"], "sampler": name, "T": T, "seed": sd}))
        eng = dict((tmpl or src_job)["engine"])
        eng["max_model_len"] = 8192
        job = {"job_id": f"hpdiv/{model}", "model": src_job["model"], "tp": src_job.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/hpdiv/{model}", "requests": reqs, "max_len": 8192}
        jf = f"{H}/jobs/hpdiv/{model}.json"
        json.dump(job, open(jf, "w"))
        queue(jf, src_job["model"], src_job.get("tp", 1), prio, f"hpdiv_{model}", env_for(model))
        print("hpdiv", model, "configs", len(ALL) * len(temps), "requests", len(reqs))


def build_longproc(models, prio, temps=(1.0, 1.5, 2.0), per_task=60):
    import random
    from .exp_longproc import MAXTOK, TASKS, load_task
    grid = ([("temp", [])] + _g("topk", "top_k", [10, 20, 50, 100]) + _g("topp", "top_p", [0.8, 0.9, 0.95, 0.98]) +
            _g("minp", "min_p", [0.02, 0.05, 0.1, 0.2, 0.3]) + _g("typical", "typical", [0.5, 0.95]) +
            _g("eta", "eta", [9e-4, 4e-3]) + _g("eps", "epsilon", [9e-4, 3e-3]) +
            _g("nsigma", "top_nsigma", [0.5, 1.0, 2.0, 3.0]) + _g("toph", "top_h", [0.2, 0.4, 0.6, 0.8]) +
            [("pless", [["p_less", 0.0]]), ("plessnorm", [["p_less_norm", 0.0]])] +
            [c for c in TLAST if c[0] in ("minp0.05~tl", "minp0.1~tl", "minp0.2~tl", "topp0.95~tl", "llamacpp~tl")] +
            [c for c in CHAINS if c[0] in ("llamacpp", "topk50>topp0.9", "topp0.9>topk50")] +
            [c for c in KEEP2 if c[0] in ("pless+k2", "nsigma1+k2", "minp0.1+k2")])
    rng = random.Random(0)
    items = []
    for t in TASKS:
        if t == "html_to_tsv_8k":
            continue
        data = load_task(t)
        idx = list(range(len(data)))
        rng.shuffle(idx)
        items += [(t, i, data[i]["input_prompt"]) for i in sorted(idx[:per_task])]
    os.makedirs(f"{H}/jobs/longproc_hp", exist_ok=True)
    for model in models:
        src = json.load(open(f"{H}/jobs/core32k/{model}.json"))
        ck = src["requests"][0]["prompt"].get("chat_kwargs") or {}
        reqs = []
        for t, i, prompt in items:
            p = {"type": "chat", "messages": [{"role": "user", "content": prompt}], "chat_kwargs": ck}
            mt = MAXTOK[t.rsplit("_", 1)[1]]
            reqs.append({"uid": f"{t}-{i}__greedy__s0", "prompt": p, "max_tokens": mt, "force_len": False,
                         "lcs": {"T": 1.0, "ops": [["greedy", 0.0]]},
                         "meta": {"pid": f"{t}-{i}", "task": t, "item": i, "sampler": "greedy", "T": 0.0}})
            for name, ops in grid:
                for T in temps:
                    reqs.append({"uid": f"{t}-{i}__{name}@T{T}__s0", "prompt": p, "max_tokens": mt, "force_len": False,
                                 "lcs": lcs_cfg(name, ops, T),
                                 "meta": {"pid": f"{t}-{i}", "task": t, "item": i, "sampler": name, "T": T}})
        eng = {"max_model_len": 24576 + 1024, "max_num_seqs": 512, "enable_prefix_caching": True}
        for k in ("language_model_only",):
            if k in src.get("engine", {}):
                eng[k] = src["engine"][k]
        job = {"job_id": f"longproc_hp/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/longproc_hp/{model}", "requests": reqs, "max_len": 24576}
        jf = f"{H}/jobs/longproc_hp/{model}.json"
        json.dump(job, open(jf, "w"))
        queue(jf, src["model"], src.get("tp", 1), prio, f"longproc_hp_{model}", env_for(model, ("core32k",)))
        print("longproc_hp", model, "configs", len(grid) * len(temps), "requests", len(reqs))


def build_mt(models, prio):
    """Natural multi-turn: configs not already covered by mt32k / mtk2_32k / mt32k_hp."""
    extra = ([(n, o, T) for n, o in (_g("topp", "top_p", [0.8, 0.9, 0.95, 0.98]) + _g("minp", "min_p", [0.02, 0.05, 0.1, 0.2]) +
                                     _g("nsigma", "top_nsigma", [0.5, 1.0, 1.5, 2.0]) + _g("toph", "top_h", [0.2, 0.4, 0.6, 0.8]) +
                                     _g("topk", "top_k", [10, 20, 50, 100]) + [("pless", [["p_less", 0.0]])])
              for T in (2.0, 3.0)] +
             [(n, o, T) for n, o in (_g("minp", "min_p", [0.3]) + _g("topk", "top_k", [5]) + _g("nsigma", "top_nsigma", [3.0]) +
                                     _g("toph", "top_h", [0.1]) + _g("typical", "typical", [0.5]) + [("plessnorm", [["p_less_norm", 0.0]])])
              for T in (1.0, 1.3, 1.6, 2.0)] +
             [(n, o, T) for n, o in TLAST if n in ("minp0.05~tl", "minp0.1~tl", "minp0.2~tl", "topp0.95~tl", "llamacpp~tl",
                                                     "pless~tl", "toph0.4~tl") for T in (1.0, 1.5, 2.0, 3.0)] +
             [(n, o, T) for n, o in CHAINS if n in ("llamacpp", "topk50>topp0.9", "topp0.9>topk50", "minp0.1>topp0.9",
                                                     "topp0.9>minp0.1") for T in (1.0, 1.5, 2.0)])
    os.makedirs(f"{H}/jobs/mt32k_hp2", exist_ok=True)
    for model in models:
        T0 = json.load(open(f"{H}/jobs/mt32k/{model}.json"))
        prompts = {}
        for r in T0["requests"]:
            if r["meta"].get("seed", 0) == 0:
                prompts.setdefault(r["meta"]["pid"], r)
        reqs = []
        for pid, base in sorted(prompts.items()):
            for name, ops, T in extra:
                r = copy.deepcopy(base)
                r["lcs"] = lcs_cfg(name, ops, T)
                r["meta"].update(sampler=name, T=T, seed=0)
                r["uid"] = f"{pid}__{name}@T{T}__s0"
                reqs.append(r)
        J = copy.deepcopy(T0)
        J.update(job_id=f"mt32k_hp2/{model}", out_dir=f"{H}/runs/mt32k_hp2/{model}", requests=reqs)
        jf = f"{H}/jobs/mt32k_hp2/{model}.json"
        json.dump(J, open(jf, "w"))
        queue(jf, J["model"], J.get("tp", 1), prio, f"mt32k_hp2_{model}", env_for(model, ("mt32k",)))
        print("mt32k_hp2", model, "configs", len(extra), "requests", len(reqs))


def build_reason(models, prio, temps=(0.6, 1.0, 1.3, 1.6), budget=32768):
    """AIME/HMMT 2025-26 (123 problems, 1 sample per config): every rule's parameter at four temperatures."""
    from .exp_reason import SUFFIX, THINK
    from .exp_reason2 import problems
    grid = (_g("minp", "min_p", [0.02, 0.05, 0.1, 0.2, 0.3]) + _g("topp", "top_p", [0.8, 0.9, 0.95, 0.98]) +
            _g("topk", "top_k", [10, 20, 50]) + _g("nsigma", "top_nsigma", [0.5, 1.0, 2.0]) +
            _g("toph", "top_h", [0.2, 0.4, 0.8]) + [("pless", [["p_less", 0.0]]), ("temp", [])] +
            _g("typical", "typical", [0.95]) + _g("eta", "eta", [9e-4]) +
            [c for c in TLAST if c[0] in ("minp0.1~tl", "minp0.2~tl", "llamacpp~tl")] +
            [c for c in CHAINS if c[0] in ("llamacpp",)] + [c for c in KEEP2 if c[0] in ("pless+k2", "nsigma1+k2")])
    probs = problems()
    os.makedirs(f"{H}/jobs/reason_hp", exist_ok=True)
    for model in models:
        srcf = next(f for f in (f"{H}/jobs/reason/{model}.json", f"{H}/jobs/core32k/{model}.json") if os.path.exists(f))
        src = json.load(open(srcf))
        fam = next((k for k in THINK if model.startswith(k)), None)
        ck = dict(THINK.get(fam, {}))
        reqs = []
        for pr in probs:
            base = {"prompt": {"type": "chat", "messages": [{"role": "user", "content": pr["problem"] + SUFFIX}], "chat_kwargs": ck},
                    "max_tokens": budget, "force_len": False}
            meta = {"pid": pr["pid"], "set": pr["set"], "src": pr["src"], "answer": pr["answer"]}
            for name, ops in grid:
                for T in temps:
                    reqs.append(dict(base, uid=f"{pr['pid']}__{name}@T{T}__s0", lcs=lcs_cfg(name, ops, T),
                                     meta=dict(meta, sampler=name, T=T, i=0)))
        eng = {"max_model_len": budget + 2048, "max_num_seqs": 512, "enable_prefix_caching": True}
        job = {"job_id": f"reason_hp/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/reason_hp/{model}", "requests": reqs}
        jf = f"{H}/jobs/reason_hp/{model}.json"
        json.dump(job, open(jf, "w"))
        queue(jf, src["model"], src.get("tp", 1), prio, f"reason_hp_{model}", env_for(model, ("core32k", "reason")))
        print("reason_hp", model, "configs", len(grid) * len(temps), "requests", len(reqs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["hpdiv", "longproc", "mt", "reason"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--prio", default="054")
    a = ap.parse_args()
    m = a.models.split(",")
    {"hpdiv": build_hpdiv, "longproc": build_longproc, "mt": build_mt, "reason": build_reason}[a.exp](m, a.prio)


if __name__ == "__main__":
    main()
