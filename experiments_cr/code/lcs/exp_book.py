"""Book continuation (PG-19) experiments.

book32k : continue the opening of a PG-19 test book for 32K tokens (raw text, base and
          instruct models), with the true continuation kept as a human reference.
human   : score the human continuation under each model (prompt logprobs -> rank and
          surprisal of every human token) to get the empirical 'human band'.
depth   : continue a long book from depth D in {2K..512K} (+ a ~1M multi-book corpus) for
          8K tokens: does the context depth change degeneration / entropy?

Run with the analysis venv:  ~/lcg-cr/ana/bin/python -m lcs.exp_book book32k --models ...
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from . import models as MR
from .make_jobs import JOBS, QP, RUNS, SAMPLERS_CORE, TEMPS, enqueue, sampler_cfgs
from .exp_niah import YARN_QWEN35
from .tok import load_tokenizer

DATA = os.path.expanduser("~/lcg-cr/data")
N_BOOKS = 16
PREFIX_TOK = 2048


def clean_book(t):
    t = t.replace("\r", "")
    # drop leading front matter: start at first paragraph after 3% of the book
    s = int(len(t) * 0.03)
    s = t.find("\n\n", s) + 2
    t = t[s:]
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)     # unwrap hard-wrapped lines
    t = re.sub(r"[ \t]+", " ", t)
    return t


def books(n=N_BOOKS, min_chars=400000):
    import pandas as pd
    df = pd.read_parquet(f"{DATA}/pg19_test.parquet")
    df["L"] = df.text.str.len()
    df = df[df.L >= min_chars].sort_values("short_book_title").head(n)
    return [(f"b{i}", r.short_book_title, clean_book(r.text)) for i, r in enumerate(df.itertuples())]


def tok_books(model, tok):
    d = f"{DATA}/book_tok/{model}"
    os.makedirs(d, exist_ok=True)
    out = []
    for bid, title, text in books():
        f = f"{d}/{bid}.npy"
        if not os.path.exists(f):
            ids = tok(text[:600000], add_special_tokens=False)["input_ids"]
            np.save(f, np.asarray(ids, np.int32))
        out.append((bid, title, f))
    return out


def book32k(model, prio, max_tokens=32000, cfgset="core", exp="book32k"):
    from .tok import load_tokenizer
    m = MR.get(model)
    tok = load_tokenizer(m["path"])
    reqs = []
    from .make_jobs import SAMPLERS_K2, pen_cfgs
    cfgs = (sampler_cfgs(SAMPLERS_CORE, TEMPS) if cfgset == "core" else
            sampler_cfgs(SAMPLERS_K2, TEMPS, greedy=False) if cfgset == "k2" else pen_cfgs())
    pdir = f"{DATA}/book_tok/{model}/prefix"
    os.makedirs(pdir, exist_ok=True)
    for bid, title, f in tok_books(model, tok):
        ids = np.load(f)
        pf = f"{pdir}/{bid}.npy"
        pre = ids[:PREFIX_TOK]
        if tok.bos_token_id is not None and m["family"] not in ("qwen3.5", "qwen3.6", "qwen3.8", "qwen2.5"):
            pre = np.concatenate([[tok.bos_token_id], pre]).astype(np.int32)
        np.save(pf, pre)
        for cname, lcs, sname, T in cfgs:
            uid = f"{bid}__{cname}__s0"
            reqs.append({"uid": uid, "prompt": {"type": "tokens", "file": pf}, "max_tokens": max_tokens,
                         "force_len": True, "lcs": lcs,
                         "meta": {"pid": bid, "domain": "book", "genre": title[:40], "sampler": sname, "T": T, "seed": 0}})
    job = {"job_id": f"{exp}/{model}", "model": m["path"], "tp": m["tp"],
           "engine": {"max_model_len": max_tokens + PREFIX_TOK + 512, "max_num_seqs": 512},
           "out_dir": f"{RUNS}/{exp}/{model}", "requests": reqs, "max_len": max_tokens + PREFIX_TOK + 8}
    enqueue(f"{exp}/{model}", job, m["tp"], prio, m["path"])


def depth(model, prio, depths=(2048, 16384, 65536, 131072, 262144, 524288), gen_tokens=8192, yarn=False):
    """Continue 'Carmen Ariza' (longest single novel, ~600K tokens) from several depths."""
    import pandas as pd
    m = MR.get(model)
    tok = load_tokenizer(m["path"])
    d = f"{DATA}/depth_tok/{model}"
    os.makedirs(d, exist_ok=True)
    full = f"{d}/corpus.npy"
    if not os.path.exists(full):
        df = pd.read_parquet(f"{DATA}/pg19_test.parquet")
        titles = ["Carmen Ariza", "The Ragged Trousered Philanthropists", "Old Mortality", "The Fair Maid of Perth"]
        parts = []
        for t in titles:
            row = df[df.short_book_title.str.startswith(t)].iloc[0]
            parts.append(clean_book(row.text))
        ids = tok("\n\n".join(parts), add_special_tokens=False)["input_ids"]
        np.save(full, np.asarray(ids, np.int32))
    ids = np.load(full)
    cfgs = [c for c in sampler_cfgs(SAMPLERS_CORE, [0.7, 1.0, 1.3, 1.6])]
    for D in depths:
        if D + gen_tokens > len(ids):
            continue
        pf = f"{d}/D{D}.npy"
        if not os.path.exists(pf):
            np.save(pf, ids[:D])
        reqs = []
        for cname, lcs, sname, T in cfgs:
            for s in range(2):
                uid = f"D{D}__{cname}__s{s}"
                reqs.append({"uid": uid, "prompt": {"type": "tokens", "file": pf}, "max_tokens": gen_tokens,
                             "force_len": True, "lcs": lcs,
                             "meta": {"D": D, "sampler": sname, "T": T, "seed": s, "pid": f"D{D}", "domain": "depth"}})
        eng = {"max_model_len": D + gen_tokens + 256, "max_num_seqs": 256, "enable_prefix_caching": True,
               "max_num_batched_tokens": 16384}
        if yarn and m["family"] in ("qwen3.5", "qwen3.6", "qwen3.8"):
            eng["hf_overrides"] = YARN_QWEN35
        tag = "yarn" if yarn else "native"
        name = f"depth-{tag}/{model}/D{D}"
        max_active = 128 if D <= 65536 else (48 if D <= 262144 else 16)
        job = {"job_id": name, "model": m["path"], "tp": m["tp"], "engine": eng,
               "out_dir": f"{RUNS}/depth-{tag}/{model}", "requests": reqs, "max_active": max_active,
               "max_len": D + gen_tokens + 8}
        jp = f"{JOBS}/{name}.json"
        os.makedirs(os.path.dirname(jp), exist_ok=True)
        json.dump(job, open(jp, "w"))
        json.dump({"job": jp, "gpus": m["tp"], "model_dir": m["path"], "kind": "gen",
                   "env": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}},
                  open(f"{QP}/{prio:03d}_{name.replace('/', '_')}.json", "w"))
        print(f"queued {name}: {len(reqs)} requests")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--models", required=True)
    ap.add_argument("--prio", type=int, default=40)
    ap.add_argument("--yarn", action="store_true")
    ap.add_argument("--depths", default="2048,16384,65536,131072,262144")
    a = ap.parse_args()
    for model in a.models.split(","):
        if a.exp == "book32k":
            book32k(model, a.prio)
        elif a.exp == "bookk2":
            book32k(model, a.prio, cfgset="k2", exp="bookk2")
        elif a.exp == "bookpen":
            book32k(model, a.prio, cfgset="pen", exp="bookpen")
        elif a.exp == "human":
            human_score_job(model, a.prio)
        elif a.exp == "depth":
            depth(model, a.prio, tuple(int(x) for x in a.depths.split(",")), yarn=a.yarn)
        else:
            raise SystemExit(a.exp)



def human_score_job(model, prio=12, cont=32000):
    """Score the true PG-19 continuation under `model` (rank / log-prob / entropy of human tokens)."""
    from .make_jobs import JOBS, QP, RUNS
    m = MR.get(model)
    tok = load_tokenizer(m["path"])
    d = f"{DATA}/book_tok/{model}/human"
    os.makedirs(d, exist_ok=True)
    items = []
    for bid, title, f in tok_books(model, tok):
        ids = np.load(f)
        pre = ids[:PREFIX_TOK]
        if tok.bos_token_id is not None and m["family"] not in ("qwen3.5", "qwen3.6", "qwen3.8", "qwen2.5"):
            pre = np.concatenate([[tok.bos_token_id], pre])
        seq = np.concatenate([pre, ids[PREFIX_TOK:PREFIX_TOK + cont]]).astype(np.int32)
        hf = f"{d}/{bid}.npy"
        np.save(hf, seq)
        items.append({"id": bid, "file": hf, "start": int(len(pre))})
    spec = {"model": m["path"], "tp": m["tp"], "max_model_len": PREFIX_TOK + cont + 64,
            "out": f"{RUNS}/human/{model}", "items": items, "topk": 20, "batch": 8}
    sp = f"{JOBS}/human/{model}.json"
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    json.dump(spec, open(sp, "w"))
    json.dump({"job": sp, "gpus": m["tp"], "model_dir": m["path"], "kind": "score", "args": [sp]},
              open(f"{QP}/{prio:03d}_human_{model}.json", "w"))
    print(f"queued human scoring for {model}: {len(items)} books")


if __name__ == "__main__":
    main()
