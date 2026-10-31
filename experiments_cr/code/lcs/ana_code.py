"""Analyse long-context code continuation: syntactic validity + repetition per sampler and depth.

valid_frac : for each generated file chunk (split at '# ===== File:' headers the model writes itself), the
             fraction of its lines covered by the longest prefix that parses with `ast`; averaged by line count.
"""
from __future__ import annotations

import ast
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .metrics import ngram_rep

RUNS = os.path.expanduser("~/lcg-cr/runs")
OUT = os.path.expanduser("~/lcg-cr/results")


def longest_valid_prefix(lines):
    lo, hi = 0, len(lines)
    def ok(n):
        try:
            ast.parse("\n".join(lines[:n]))
            return True
        except Exception:
            return False
    # largest n such that prefix parses; check at block boundaries (lines without indentation)
    cands = [i for i in range(1, len(lines) + 1) if i == len(lines) or (lines[i] and not lines[i][0].isspace())]
    best = 0
    l, r = 0, len(cands) - 1
    while l <= r:  # parsing validity is not monotone, but this gives a good, cheap estimate
        mid = (l + r) // 2
        if ok(cands[mid]):
            best = cands[mid]; l = mid + 1
        else:
            r = mid - 1
    return best


def one(args):
    run, m, tokpath = args
    from .tok import load_tokenizer
    global _TOK
    if "_TOK" not in globals() or _TOK[0] != tokpath:
        _TOK = (tokpath, load_tokenizer(tokpath))
    tok = _TOK[1]
    toks = np.load(f"{run}/gen/{m['uid']}.npz")["tokens"]
    text = tok.decode(toks, skip_special_tokens=True)
    chunks = text.split("\n# ===== File:")
    tot, val = 0, 0
    for ci, ch in enumerate(chunks):
        lines = ch.split("\n")[1:] if ci > 0 else ch.split("\n")
        if len(lines) < 3:
            continue
        tot += len(lines)
        val += longest_valid_prefix(lines)
    mm = m["meta"]
    return {"exp": run.split("/")[-2], "model": run.split("/")[-1], "D": mm["D"], "sampler": mm["sampler"],
            "T": mm["T"], "seed": mm["seed"], "valid_frac": val / max(1, tot), "n_files": len(chunks),
            "rep32": float(ngram_rep(toks.astype(np.int64), 32).mean()),
            "rep32_q4": float(ngram_rep(toks.astype(np.int64), 32)[3 * len(toks) // 4:].mean())}


def main():
    tasks = []
    for run in sorted(glob.glob(f"{RUNS}/code-*/*")):
        if not os.path.exists(f"{run}/meta.jsonl"):
            continue
        metas = [json.loads(l) for l in open(f"{run}/meta.jsonl")]
        tasks += [(run, m, metas[0]["model"]) for m in metas]
    with Pool(48) as p:
        rows = p.map(one, tasks, chunksize=8)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f"{OUT}/code_gen.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 1000)
    for model, g in df.groupby("model"):
        print("==", model)
        print(g.pivot_table(index=["sampler", "T"], columns="D", values=["valid_frac", "rep32_q4"]).round(2).to_string())


if __name__ == "__main__":
    main()
