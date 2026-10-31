"""Create generation jobs and enqueue them for the scheduler.

python -m lcs.make_jobs core --models qwen3.5-0.8b,qwen3.5-2b --prio 10
"""
from __future__ import annotations

import argparse
import json
import os

from . import models as MR
from .prompts import chat_prompt, fiction_prompts, nonfiction_prompts

HOME = os.path.expanduser("~/lcg-cr")
JOBS = f"{HOME}/jobs"
RUNS = f"{HOME}/runs"
QP = f"{HOME}/queue/pending"

SAMPLERS_CORE = [
    ("temp", []),
    ("topk50", [["top_k", 50]]),
    ("topp0.9", [["top_p", 0.9]]),
    ("topp0.95", [["top_p", 0.95]]),
    ("minp0.05", [["min_p", 0.05]]),
    ("minp0.1", [["min_p", 0.1]]),
    ("nsigma1", [["top_nsigma", 1.0]]),
    ("pless", [["p_less", 0.0]]),
    ("toph0.4", [["top_h", 0.4]]),
]
SAMPLERS_EXT = [
    ("typical0.9", [["typical", 0.9]]),
    ("eta0.0009", [["eta", 0.0009]]),
    ("eps0.0003", [["epsilon", 0.0003]]),
    ("pless_norm", [["p_less_norm", 0.0]]),
    ("nsigma1.5", [["top_nsigma", 1.5]]),
    ("nsigma0.5", [["top_nsigma", 0.5]]),
    ("minp0.2", [["min_p", 0.2]]),
    ("toph0.6", [["top_h", 0.6]]),
]
TEMPS = [0.7, 1.0, 1.3, 1.6, 2.0]
DRY = {"mult": 0.8, "base": 1.75, "allowed": 2}
# penalties / DRY / vendor defaults / min-k  (T grid given per entry)
SAMPLERS_PEN = [
    ("mink3", [["min_k", 3.0]], None, None, [0.7, 1.0, 1.3, 1.6, 2.0]),
    ("dry+topp0.9", [["top_p", 0.9]], None, DRY, [0.7, 1.0, 1.3]),
    ("dry+minp0.1", [["min_p", 0.1]], None, DRY, [0.7, 1.0, 1.3, 1.6]),
    ("dry+nsigma1", [["top_nsigma", 1.0]], None, DRY, [0.7, 1.0, 1.3, 1.6]),
    ("dry+pless", [["p_less", 0.0]], None, DRY, [0.7, 1.0, 1.3, 1.6]),
    ("dry+toph0.4", [["top_h", 0.4]], None, DRY, [0.7, 1.0, 1.3, 1.6]),
    ("pres1.5+topp0.9", [["top_p", 0.9]], {"presence": 1.5}, None, [0.7, 1.0]),
    ("rep1.1+topp0.9", [["top_p", 0.9]], {"repetition": 1.1}, None, [0.7, 1.0]),
    ("vendor_qwen", [["top_k", 20], ["top_p", 0.8]], {"presence": 1.5}, None, [0.7]),
]


def pen_cfgs():
    out = []
    for name, ops, pen, dry, temps in SAMPLERS_PEN:
        for T in temps:
            l = {"T": T, "ops": ops}
            if pen:
                l["pen"] = pen
            if dry:
                l["dry"] = dry
            out.append((f"{name}@T{T}", l, name, T))
    return out


SAMPLERS_SLIM = [("temp", []), ("topp0.9", [["top_p", 0.9]]), ("minp0.1", [["min_p", 0.1]]),
                 ("nsigma1", [["top_nsigma", 1.0]]), ("pless", [["p_less", 0.0]]), ("toph0.4", [["top_h", 0.4]]),
                 ("topk50", [["top_k", 50]])]

# "absorption-free" variants: the candidate set always keeps the top-2 tokens (never a singleton)
SAMPLERS_K2 = [
    ("nsigma1+k2", [["top_nsigma", 1.0], ["keep_top", 2]]),
    ("pless+k2", [["p_less", 0.0], ["keep_top", 2]]),
    ("topp0.9+k2", [["top_p", 0.9], ["keep_top", 2]]),
    ("minp0.1+k2", [["min_p", 0.1], ["keep_top", 2]]),
    ("toph0.4+k2", [["top_h", 0.4], ["keep_top", 2]]),
]


def sampler_cfgs(samplers, temps, greedy=True):
    out = []
    for name, ops in samplers:
        for T in temps:
            out.append((f"{name}@T{T}", {"T": T, "ops": ops}, name, T))
    if greedy:
        out.append(("greedy", {"T": 1.0, "ops": [["greedy", 0]]}, "greedy", 0.0))
    return out


def enqueue(name, job, gpus, prio, model_dir, env=None):
    os.makedirs(os.path.dirname(f"{JOBS}/{name}.json"), exist_ok=True)
    jp = f"{JOBS}/{name}.json"
    json.dump(job, open(jp, "w"))
    q = {"job": jp, "gpus": gpus, "model_dir": model_dir, "kind": "gen", "env": env or {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}}
    json.dump(q, open(f"{QP}/{prio:03d}_{name.replace('/', '_')}.json", "w"))
    print(f"queued {name}: {len(job['requests'])} requests on {gpus} GPU(s), prio {prio}")


def core_job(model, exp, prompts, cfgs, max_tokens, engine_extra=None, seeds=(0,)):
    m = MR.get(model)
    reqs = []
    for pr in prompts:
        for cname, lcs, sname, T in cfgs:
            for s in seeds:
                uid = f"{pr['pid']}__{cname}__s{s}".replace("/", "_")
                prompt = chat_prompt(pr["text"], m["chat_kwargs"]) if m["kind"] != "base" else {"type": "text", "text": pr["text"]}
                reqs.append({"uid": uid, "prompt": prompt, "max_tokens": max_tokens, "force_len": True,
                             "lcs": lcs, "meta": {"pid": pr["pid"], "domain": pr["domain"], "genre": pr["genre"],
                                                  "sampler": sname, "T": T, "seed": s}})
    eng = {"max_model_len": max_tokens + 1024, "max_num_seqs": 512}
    eng.update(engine_extra or {})
    return {"job_id": f"{exp}/{model}", "model": m["path"], "tp": m["tp"], "engine": eng,
            "out_dir": f"{RUNS}/{exp}/{model}", "requests": reqs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--models", required=True)
    ap.add_argument("--prio", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--per_genre", type=int, default=2)
    ap.add_argument("--multiturn", action="store_true", help="run pen/k2 grids under the multi-turn protocol")
    a = ap.parse_args()
    for model in a.models.split(","):
        m = MR.get(model)
        if a.exp == "core":
            cfgs = sampler_cfgs(SAMPLERS_CORE, TEMPS)
            job = core_job(model, "core32k", fiction_prompts(a.per_genre), cfgs, a.max_tokens)
            enqueue(f"core32k/{model}", job, m["tp"], a.prio, m["path"])
        elif a.exp == "ext":
            cfgs = sampler_cfgs(SAMPLERS_EXT, TEMPS, greedy=False)
            job = core_job(model, "ext32k", fiction_prompts(a.per_genre), cfgs, a.max_tokens)
            enqueue(f"ext32k/{model}", job, m["tp"], a.prio, m["path"])
        elif a.exp == "slim":
            cfgs = sampler_cfgs(SAMPLERS_SLIM, [1.0, 1.6])
            job = core_job(model, "core32k", fiction_prompts(a.per_genre), cfgs, a.max_tokens)
            enqueue(f"core32k/{model}", job, m["tp"], a.prio, m["path"])
        elif a.exp in ("pen", "k2"):
            cfgs = pen_cfgs() if a.exp == "pen" else sampler_cfgs(SAMPLERS_K2, TEMPS, greedy=False)
            exp = ("mt" if a.multiturn else "") + ("pen32k" if a.exp == "pen" else "k2_32k")
            job = core_job(model, exp, fiction_prompts(a.per_genre), cfgs, a.max_tokens)
            if a.multiturn:
                for r in job["requests"]:
                    r["force_len"] = False
                    r["multiturn"] = {"turn_tokens": 8192, "total": a.max_tokens, "continue_msg": "Continue.", "max_turns": 48}
                job["max_len"] = a.max_tokens + 8192 + 1024
                job["engine"]["max_model_len"] = a.max_tokens + 8192 + 2048
            enqueue(f"{exp}/{model}", job, m["tp"], a.prio, m["path"])
        elif a.exp == "mt":
            cfgs = sampler_cfgs(SAMPLERS_CORE, [0.7, 1.0, 1.3, 1.6])
            job = core_job(model, "mt32k", fiction_prompts(a.per_genre), cfgs, a.max_tokens)
            for r in job["requests"]:
                r["force_len"] = False
                r["multiturn"] = {"turn_tokens": 8192, "total": a.max_tokens, "continue_msg": "Continue.", "max_turns": 48}
            job["max_len"] = a.max_tokens + 8192 + 1024
            job["engine"]["max_model_len"] = a.max_tokens + 8192 + 2048
            enqueue(f"mt32k/{model}", job, m["tp"], a.prio, m["path"])
        elif a.exp == "nonfic":
            cfgs = sampler_cfgs(SAMPLERS_CORE, [1.0, 1.3, 1.6])
            job = core_job(model, "nonfic32k", nonfiction_prompts(), cfgs, a.max_tokens)
            enqueue(f"nonfic32k/{model}", job, m["tp"], a.prio, m["path"])
        else:
            raise SystemExit(f"unknown exp {a.exp}")


if __name__ == "__main__":
    main()
