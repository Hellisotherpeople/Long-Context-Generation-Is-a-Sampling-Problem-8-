"""Loop-escape intervention: does the sampler let the model leave a repetition loop?

Context = natural book passage (PG-19 validation) that ends with sentence s, followed by
k extra verbatim repetitions of s.  For each sampler config we draw N short continuations
and measure whether the model repeats s again (stays in the loop) or escapes.
Raw-text completion (no chat template) so base and instruct models are comparable.

python -m lcs.exp_loop --models qwen3.5-4b,... --prio 30
"""
from __future__ import annotations

import argparse
import json
import os
import re

from . import models as MR
from .make_jobs import SAMPLERS_CORE, TEMPS, enqueue, sampler_cfgs, RUNS

KS = [0, 1, 2, 4, 8, 16]
N_SAMPLES = 8
N_CTX = 24
CONT_TOKENS = 48
DATA = os.path.expanduser("~/lcg-cr/data")
SENT = re.compile(r"[^.!?]*[.!?]")


def build_contexts(n_ctx=N_CTX, approx_chars=6000, seed=0):
    import pandas as pd
    import random
    out_path = f"{DATA}/loop_contexts.json"
    if os.path.exists(out_path):
        return json.load(open(out_path))
    rng = random.Random(seed)
    df = pd.read_parquet(f"{DATA}/pg19_val.parquet")
    ctxs = []
    rows = list(range(len(df)))
    rng.shuffle(rows)
    for i in rows:
        text = df.iloc[i]["text"]
        if len(text) < 60000:
            continue
        start = rng.randint(len(text) // 4, len(text) // 2)
        chunk = text[start:start + approx_chars]
        chunk = chunk[chunk.find("\n") + 1:] if "\n" in chunk[:500] else chunk
        chunk = re.sub(r"[ \t]+", " ", chunk.replace("\r", ""))
        chunk = re.sub(r"(?<!\n)\n(?!\n)", " ", chunk)   # unwrap hard-wrapped lines
        sents = [m.group(0) for m in SENT.finditer(chunk)]
        # choose the last sentence with 8..30 words
        for j in range(len(sents) - 1, len(sents) // 2, -1):
            s = sents[j].strip()
            nw = len(s.split())
            if 8 <= nw <= 30 and "\n" not in s and '"' not in s:
                end = chunk.find(sents[j]) + len(sents[j])
                ctxs.append({"cid": f"c{len(ctxs)}", "book": df.iloc[i]["short_book_title"],
                             "prefix": chunk[:end - len(sents[j])].rstrip() + " ", "s": s})
                break
        if len(ctxs) >= n_ctx:
            break
    json.dump(ctxs, open(out_path, "w"), indent=1)
    return ctxs


def loop_job(model, ctxs, cfgs, exp="loop"):
    m = MR.get(model)
    reqs = []
    for c in ctxs:
        for k in KS:
            text = c["prefix"] + " ".join([c["s"]] * (k + 1))
            for cname, lcs, sname, T in cfgs:
                ns = 1 if sname == "greedy" else N_SAMPLES
                for i in range(ns):
                    uid = f"{c['cid']}__k{k}__{cname}__n{i}"
                    reqs.append({"uid": uid, "prompt": {"type": "text", "text": text}, "max_tokens": CONT_TOKENS,
                                 "force_len": True, "lcs": lcs,
                                 "meta": {"cid": c["cid"], "k": k, "sampler": sname, "T": T, "i": i}})
    eng = {"max_model_len": 8192, "max_num_seqs": 512, "enable_prefix_caching": True}
    return {"job_id": f"{exp}/{model}", "model": m["path"], "tp": m["tp"], "engine": eng,
            "out_dir": f"{RUNS}/{exp}/{model}", "requests": reqs, "max_active": 512}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prio", type=int, default=30)
    a = ap.parse_args()
    ctxs = build_contexts()
    print(f"{len(ctxs)} contexts")
    cfgs = sampler_cfgs(SAMPLERS_CORE, TEMPS)
    for model in a.models.split(","):
        m = MR.get(model)
        job = loop_job(model, ctxs, cfgs)
        enqueue(f"loop/{model}", job, m["tp"], a.prio, m["path"])


if __name__ == "__main__":
    main()
