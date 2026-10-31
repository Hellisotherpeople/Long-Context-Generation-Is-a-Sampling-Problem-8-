"""Long-context code continuation at depth (agentic-coding proxy).

Context = concatenated Python source files of the `transformers` package (Apache-2.0), each preceded by
"# ===== File: <path> =====".  At depth D we cut the corpus at a file boundary, add the header of the next
file, and let the model write that file (8K tokens, forced length).  Metrics: repetition, and the fraction
of generated top-level blocks that parse (ast) -- see ana_code.py.

~/lcg-cr/ana/bin/python -m lcs.exp_code --models qwen3.5-9b-base --depths 8192,65536,250000 --prio 32
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import models as MR
from .exp_niah import YARN_QWEN35
from .make_jobs import JOBS, QP, RUNS, SAMPLERS_CORE, sampler_cfgs
from .tok import load_tokenizer

DATA = os.path.expanduser("~/lcg-cr/data")
SRC = "/home/ubuntu/glm53-vllm/.venv/lib/python3.12/site-packages/transformers"


def corpus_files():
    files = sorted(glob.glob(f"{SRC}/**/*.py", recursive=True))
    return [f for f in files if os.path.getsize(f) > 2000 and "/tests/" not in f]


def build(model, tok):
    d = f"{DATA}/code_tok/{model}"
    os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/corpus.npy"):
        return np.load(f"{d}/corpus.npy"), json.load(open(f"{d}/bounds.json"))
    ids, bounds = [], []
    for f in corpus_files():
        rel = os.path.relpath(f, os.path.dirname(SRC))
        head = tok(f"\n# ===== File: {rel} =====\n", add_special_tokens=False)["input_ids"]
        body = tok(open(f, errors="ignore").read(), add_special_tokens=False)["input_ids"]
        bounds.append([len(ids), rel])
        ids += head + body
        if len(ids) > 1_100_000:
            break
    arr = np.asarray(ids, np.int32)
    np.save(f"{d}/corpus.npy", arr)
    json.dump(bounds, open(f"{d}/bounds.json", "w"))
    return arr, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--depths", default="8192,65536,250000")
    ap.add_argument("--gen", type=int, default=8192)
    ap.add_argument("--prio", type=int, default=32)
    ap.add_argument("--yarn", action="store_true")
    a = ap.parse_args()
    cfgs = sampler_cfgs(SAMPLERS_CORE, [0.4, 0.7, 1.0, 1.3])
    for model in a.models.split(","):
        m = MR.get(model)
        tok = load_tokenizer(m["path"])
        corpus, bounds = build(model, tok)
        for D in [int(x) for x in a.depths.split(",")]:
            # cut at the first file boundary at or after D; the prompt ends with that file's header
            starts = [b for b in bounds if b[0] >= D]
            if not starts:
                continue
            b0 = starts[0][0]
            head = tok(f"\n# ===== File: {starts[0][1]} =====\n", add_special_tokens=False)["input_ids"]
            pf = f"{DATA}/code_tok/{model}/D{D}.npy"
            np.save(pf, np.concatenate([corpus[:b0], np.asarray(head, np.int32)]))
            reqs = []
            for cname, lcs, sname, T in cfgs:
                for s in range(2):
                    reqs.append({"uid": f"D{D}__{cname}__s{s}", "prompt": {"type": "tokens", "file": pf},
                                 "max_tokens": a.gen, "force_len": True, "lcs": lcs,
                                 "meta": {"D": D, "sampler": sname, "T": T, "seed": s, "pid": f"D{D}",
                                          "domain": "code", "file": starts[0][1]}})
            eng = {"max_model_len": b0 + len(head) + a.gen + 256, "max_num_seqs": 256,
                   "enable_prefix_caching": True, "max_num_batched_tokens": 16384}
            if a.yarn and m["family"] in ("qwen3.5", "qwen3.6", "qwen3.8"):
                eng["hf_overrides"] = YARN_QWEN35
            tag = "yarn" if a.yarn else "native"
            name = f"code-{tag}/{model}/D{D}"
            max_active = 128 if D <= 65536 else (48 if D <= 262144 else 16)
            job = {"job_id": name, "model": m["path"], "tp": m["tp"], "engine": eng,
                   "out_dir": f"{RUNS}/code-{tag}/{model}", "requests": reqs, "max_active": max_active,
                   "max_len": b0 + len(head) + a.gen + 8}
            jp = f"{JOBS}/{name}.json"
            os.makedirs(os.path.dirname(jp), exist_ok=True)
            json.dump(job, open(jp, "w"))
            json.dump({"job": jp, "gpus": m["tp"], "model_dir": m["path"], "kind": "gen",
                       "env": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}},
                      open(f"{QP}/{a.prio:03d}_{name.replace('/', '_')}.json", "w"))
            print(f"queued {name}: {len(reqs)} requests, prompt {b0 + len(head)} tokens")


if __name__ == "__main__":
    main()
