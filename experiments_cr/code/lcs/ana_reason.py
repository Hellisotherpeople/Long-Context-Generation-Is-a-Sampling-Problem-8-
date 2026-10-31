"""Grade long-CoT reasoning runs.

python -m lcs.ana_reason  -> ~/lcg-cr/results/reason_{gen,cells}.parquet
correct    : last \\boxed{} answer matches (math-verify when available, else normalized string / integer)
exhausted  : generation hit the token budget (no natural end)
looped     : terminal loop (32-gram repetition >= 0.5 over the final 4K tokens)
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from .metrics import ngram_rep

EXP = os.environ.get("LCS_REASON_EXP", "reason")   # LCS_REASON_EXP=reason2 for the second wave
RUNS = os.path.expanduser(f"~/lcg-cr/runs/{EXP}")
OUT = os.path.expanduser("~/lcg-cr/results")


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def norm(s):
    s = s.strip().replace(" ", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "").rstrip(".")
    return s


def grade(pred, gold):
    if pred is None:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        pass
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return False


def main():
    from .tok import load_tokenizer
    rows = []
    for run in sorted(glob.glob(f"{RUNS}/*")):
        mf = f"{run}/meta.jsonl"
        if not os.path.exists(mf):
            continue
        metas = [json.loads(l) for l in open(mf)]
        tok = load_tokenizer(metas[0]["model"])
        for m in metas:
            mm = m["meta"]
            toks = np.load(f"{run}/gen/{m['uid']}.npz")["tokens"]
            text = tok.decode(toks, skip_special_tokens=True)
            tail = toks[-4096:]
            looped = len(tail) >= 2048 and float(ngram_rep(toks, 32)[-4096:].mean()) >= 0.5
            rows.append({"model": run.split("/")[-1].split("__")[0], "pid": mm["pid"], "set": mm["set"], "sampler": mm["sampler"],
                         "T": mm["T"], "i": mm["i"], "n_tokens": len(toks), "finish": m["finish"],
                         "exhausted": m["finish"] == "length", "looped": looped,
                         "correct": grade(last_boxed(text), mm["answer"])})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f"{OUT}/{EXP}_gen.parquet")
    cells = df.groupby(["model", "set", "sampler", "T"]).agg(
        n=("correct", "size"), acc=("correct", "mean"), exhausted=("exhausted", "mean"),
        looped=("looped", "mean"), len=("n_tokens", "mean")).reset_index()
    cells.to_parquet(f"{OUT}/{EXP}_cells.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 3000)
    print(cells.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
