"""Multi-needle key-value retrieval under stochastic decoding, 8K -> ~1M tokens.

For each (model, context length L, instance) we build one haystack of PG-19 text with
NK needles "The special magic number for <key> is <7 digits>." at evenly spaced depths,
pre-tokenise the shared prefix once (prefix caching makes every query share a single
prefill), and ask for each key under every sampler config.

Run with the analysis venv (needs transformers tokenizers + pandas):
  ~/lcg-cr/ana/bin/python -m lcs.exp_niah --models qwen3.5-9b --lengths 8192,32768,131072 --prio 25
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from . import models as MR
from .make_jobs import SAMPLERS_CORE, enqueue, sampler_cfgs, RUNS

DATA = os.path.expanduser("~/lcg-cr/data")
NK = 16
WORDS = ("amber falcon cobalt willow ember harbor quartz meadow onyx thistle cedar lantern saffron "
         "glacier copper orchid basalt raven juniper marble sparrow indigo canyon velvet tundra "
         "maple garnet heron flint coral ivory nimbus prairie slate topaz wren birch cinder dune").split()
QUESTION = "What is the special magic number for {key} mentioned in the document? Answer with just the number."
INSTR = ("Below is a long document. Somewhere in it are several special magic numbers. "
         "Read it carefully; afterwards you will be asked about one of them.\n\n<document>\n")
YARN_QWEN35 = {"text_config": {"rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
               "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0,
               "original_max_position_embeddings": 262144}}}


def book_text():
    import pandas as pd
    p = f"{DATA}/haystack.txt"
    if not os.path.exists(p):
        df = pd.read_parquet(f"{DATA}/pg19_test.parquet")
        open(p, "w").write("\n\n".join(t.replace("\r", "") for t in df.text.tolist()))
    return open(p).read()


def build_instance(tok, chat_kwargs, L, inst, kind):
    rng = random.Random(1000 * inst + L % 997)
    text = book_text()
    # pick an offset so different instances use different text
    off = rng.randint(0, len(text) // 3)
    keys = rng.sample([f"{a}-{b}" for a in WORDS for b in WORDS if a != b], NK)
    vals = [str(rng.randint(1000000, 9999999)) for _ in keys]
    q_placeholder = "<<<QUESTION>>>"
    if kind == "base":
        pre, post = INSTR, "\n</document>\n\n" + q_placeholder + "\nAnswer:"
    else:
        msgs = [{"role": "user", "content": INSTR + "@@DOC@@" + "\n</document>\n\n" + q_placeholder}]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        pre, post = full.split("@@DOC@@")
        post = "\n</document>\n\n" + post.split("\n</document>\n\n", 1)[1]
    pre_ids = tok(pre, add_special_tokens=(kind == "base"))["input_ids"]
    q_len = len(tok(post.replace(q_placeholder, QUESTION.format(key="amber-falcon")), add_special_tokens=False)["input_ids"])
    budget = L - len(pre_ids) - q_len - 64
    # needles
    needles = [f" The special magic number for {k} is {v}. " for k, v in zip(keys, vals)]
    needle_ids = [tok(n, add_special_tokens=False)["input_ids"] for n in needles]
    hay_budget = budget - sum(len(x) for x in needle_ids)
    # tokenise enough haystack text (approx 3.5 chars/token, then trim)
    chunk = text[off: off + int(hay_budget * 4.5) + 10000]
    hay = tok(chunk, add_special_tokens=False)["input_ids"][:hay_budget]
    depths = [(i + 0.5) / NK for i in range(NK)]
    order = list(range(NK))
    rng.shuffle(order)
    ids = list(pre_ids)
    last = 0
    placed = []
    def snap(cut):
        for c in range(cut, min(cut + 96, len(hay))):
            piece = tok.decode(hay[c - 1:c])
            if piece.endswith((".", "\n", "!", "?")):
                return c
        return cut
    for j, d in enumerate(depths):
        cut = max(last, snap(int(d * len(hay))))
        ids += hay[last:cut]
        k = order[j]
        placed.append({"key": keys[k], "val": vals[k], "depth": d})
        ids += needle_ids[k]
        last = cut
    ids += hay[last:]
    return np.asarray(ids, np.int32), post, placed


def niah_jobs(model, lengths, n_inst, cfgs, n_samples, prio, yarn=False, exp="niah", tp=None):
    from .tok import load_tokenizer
    from .make_jobs import JOBS, QP
    m = MR.get(model)
    tok = load_tokenizer(m["path"])
    tag = "yarn" if yarn else "native"
    tokdir = f"{DATA}/niah_tok/{model}"
    os.makedirs(tokdir, exist_ok=True)
    for L in lengths:
        reqs = []
        for inst in range(n_inst):
            f = f"{tokdir}/L{L}_i{inst}.npy"
            meta_f = f + ".json"
            if not os.path.exists(meta_f):
                ids, post, placed = build_instance(tok, m["chat_kwargs"], L, inst, m["kind"])
                np.save(f, ids)
                json.dump({"post": post, "placed": placed, "n_prefix": int(len(ids))}, open(meta_f, "w"))
            info = json.load(open(meta_f))
            for nd in info["placed"]:
                suffix = info["post"].replace("<<<QUESTION>>>", QUESTION.format(key=nd["key"]))
                for cname, lcs, sname, T in cfgs:
                    ns = 1 if sname == "greedy" else n_samples
                    for i in range(ns):
                        uid = f"L{L}_i{inst}__{nd['key']}__{cname}__n{i}"
                        reqs.append({"uid": uid, "prompt": {"type": "tokens", "file": f, "suffix_text": suffix},
                                     "max_tokens": 24, "force_len": False, "lcs": lcs,
                                     "meta": {"L": L, "inst": inst, "key": nd["key"], "val": nd["val"],
                                              "depth": nd["depth"], "sampler": sname, "T": T, "i": i}})
        reqs.sort(key=lambda r: r["meta"]["inst"])
        max_active = 256 if L <= 140000 else (64 if L <= 270000 else (24 if L <= 530000 else 12))
        eng = {"max_model_len": L + 512, "max_num_seqs": 256, "enable_prefix_caching": True,
               "enable_chunked_prefill": True, "max_num_batched_tokens": 16384}
        if yarn and m["family"] in ("qwen3.5", "qwen3.6", "qwen3.8"):
            eng["hf_overrides"] = YARN_QWEN35
        name = f"{exp}-{tag}/{model}/L{L}"
        tpx = tp or m["tp"]
        job = {"job_id": name, "model": m["path"], "tp": tpx, "engine": eng,
               "out_dir": f"{RUNS}/{exp}-{tag}/{model}", "requests": reqs, "max_active": max_active,
               "max_len": L + 128}
        os.makedirs(os.path.dirname(f"{JOBS}/{name}.json"), exist_ok=True)
        jp = f"{JOBS}/{name}.json"
        json.dump(job, open(jp, "w"))
        q = {"job": jp, "gpus": tpx, "model_dir": m["path"], "kind": "gen",
             "env": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}}
        json.dump(q, open(f"{QP}/{prio:03d}_{name.replace('/', '_')}.json", "w"))
        print(f"queued {name}: {len(reqs)} requests (max_active {max_active})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--lengths", default="8192,32768,131072,262000")
    ap.add_argument("--n_inst", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--yarn", action="store_true")
    ap.add_argument("--prio", type=int, default=25)
    ap.add_argument("--tp", type=int, default=None)
    a = ap.parse_args()
    cfgs = sampler_cfgs(SAMPLERS_CORE, [0.7, 1.0, 1.5])
    lengths = [int(x) for x in a.lengths.split(",")]
    for model in a.models.split(","):
        niah_jobs(model, lengths, a.n_inst, cfgs, a.n_samples, a.prio, a.yarn, tp=a.tp)


if __name__ == "__main__":
    main()
