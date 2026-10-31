"""Analyse the loop-escape intervention.

stay = the continuation starts by repeating the looped sentence s again.
Also first-step diagnostics: p_max of the raw distribution, candidate-set size, retained mass.
python -m lcs.ana_loop  -> ~/lcg-cr/results/loop_cells.parquet
"""
from __future__ import annotations

import glob
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

RUNS = os.path.expanduser("~/lcg-cr/runs/loop")
OUT = os.path.expanduser("~/lcg-cr/results")
CTX = json.load(open(os.path.expanduser("~/lcg-cr/data/loop_contexts.json")))
SENT = {c["cid"]: c["s"] for c in CTX}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _chunk(args):
    model_dir, metas, statfiles = args
    from .tok import load_tokenizer
    tok = load_tokenizer(metas[0]["model"])
    model = os.path.basename(model_dir)
    rows = []
    for m in metas:
        u = m["uid"]
        mm = m["meta"]
        toks = np.load(f"{model_dir}/gen/{u}.npz")["tokens"]
        text = norm(tok.decode(toks, skip_special_tokens=True))
        s = norm(SENT[mm["cid"]])
        stay = text.startswith(s[: min(len(s), 40)])
        st = None
        f = statfiles.get(u)
        if f:
            d = np.load(f"{model_dir}/stats/{f}")
            pos = d["pos"]
            i0 = np.where(pos == 0)[0]
            if len(i0):
                st = d["stats"][i0[-1]]
        rows.append({"model": model, "cid": mm["cid"], "k": mm["k"], "sampler": mm["sampler"], "T": mm["T"],
                     "i": mm["i"], "stay": bool(stay),
                     "pmax0": float(st[2]) if st is not None else np.nan,
                     "kept0": float(st[3]) if st is not None else np.nan,
                     "mass0": float(st[4]) if st is not None else np.nan,
                     "Hq0": float(st[5]) if st is not None else np.nan})
    return rows


def tasks_for(model_dir, nchunk=24):
    metas = [json.loads(l) for l in open(f"{model_dir}/meta.jsonl")]
    statfiles = {}
    for f in os.listdir(f"{model_dir}/stats"):
        statfiles.setdefault(f.split(".p")[0], f)
    return [(model_dir, metas[i::nchunk], statfiles) for i in range(nchunk) if metas[i::nchunk]]


def main():
    os.makedirs(OUT, exist_ok=True)
    dirs = [d for d in glob.glob(f"{RUNS}/*") if os.path.exists(f"{d}/meta.jsonl")]
    tasks = [t for d in dirs for t in tasks_for(d)]
    with Pool(min(64, len(tasks))) as p:
        rows = [r for rs in p.map(_chunk, tasks) for r in rs]
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT}/loop_gen.parquet")
    cells = df.groupby(["model", "sampler", "T", "k"]).agg(
        n=("stay", "size"), stay=("stay", "mean"), pmax0=("pmax0", "mean"),
        singleton0=("kept0", lambda x: float(np.mean(np.asarray(x) <= 1))), mass0=("mass0", "mean")).reset_index()
    cells.to_parquet(f"{OUT}/loop_cells.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 2000)
    piv = cells.pivot_table(index=["model", "sampler", "T"], columns="k", values="stay").round(2)
    print(piv.to_string())
    print(cells.groupby(["model", "k"]).pmax0.mean().unstack().round(3))


if __name__ == "__main__":
    main()
