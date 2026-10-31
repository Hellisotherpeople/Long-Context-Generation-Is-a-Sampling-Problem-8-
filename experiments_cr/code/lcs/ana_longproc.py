"""Score LongProc runs with the official evaluators (princeton-pli/LongProc).

python -m lcs.ana_longproc -> ~/lcg-cr/results/longproc_{gen,cells}.parquet
Main metric per task: html_to_tsv -> F1; travel_planning -> accuracy; others -> accuracy (strict) and partial_accuracy.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

H = os.path.expanduser("~/lcg-cr")
REPO = f"{H}/data/bench/LongProc"
EXP = os.environ.get("LCS_LONGPROC_EXP", "longproc")   # longproc | longproc_hp (ablation)


def loaders():
    sys.path.insert(0, REPO)
    from longproc.longproc_data import load_longproc_data
    out = {}
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        from .exp_longproc import TASKS
        for t in TASKS:
            out[t] = load_longproc_data(t, "./data")
    finally:
        os.chdir(cwd)
    return out


_L = None


def _init():
    global _L
    import contextlib
    sys.stdout = open(os.devnull, "w")  # the official evaluators print a lot
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        _L = loaders()


def _score(job):
    task, item, uid, pred = job
    data, ev = _L[task]
    try:
        met, _ = ev(pred, data[item])
    except Exception as e:  # evaluator failure counts as a failed extraction
        met = {"error": repr(e)[:100]}
    return uid, {"score": met.get("f1", met.get("accuracy", np.nan)),
                 "partial": met.get("partial_accuracy", met.get("recall", np.nan)),
                 "extracted": met.get("extraction_rate", np.nan)}


def main():
    from multiprocessing import Pool
    from .tok import load_tokenizer
    rows = []
    for run in sorted(glob.glob(f"{H}/runs/{EXP}/*")):
        model = run.split("/")[-1]
        if not os.path.exists(f"{run}/meta.jsonl"):
            continue
        metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
        cache_f = f"{run}/longproc_scores.jsonl"
        cache = {}
        if os.path.exists(cache_f):
            for l in open(cache_f):
                r = json.loads(l)
                cache[r["uid"]] = r
        todo = [m for m in metas if m["uid"] not in cache]
        if todo:
            tok = load_tokenizer(metas[0]["model"])
            jobs = [(m["meta"]["task"], m["meta"]["item"], m["uid"],
                     tok.decode(np.load(f"{run}/gen/{m['uid']}.npz")["tokens"], skip_special_tokens=True)) for m in todo]
            with Pool(int(os.environ.get("NPROC", 48)), initializer=_init) as pool, open(cache_f, "a") as f:
                for uid, sc in pool.imap_unordered(_score, jobs, chunksize=8):
                    rec = {"uid": uid, **{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in sc.items()}}
                    f.write(json.dumps(rec) + "\n")
                    cache[uid] = rec
        for m in metas:
            mm, sc = m["meta"], cache[m["uid"]]
            rows.append({"model": model, "task": mm["task"], "item": mm["item"], "sampler": mm["sampler"], "T": mm["T"],
                         "n_tokens": m["n_tokens"], "finish": m.get("finish"),
                         **{k: (np.nan if sc.get(k) is None else sc[k]) for k in ("score", "partial", "extracted")},
                         "eval_failed": sc.get("score") is None})
            if rows[-1]["eval_failed"]:  # evaluator could not score the output (e.g. generated code timed out): counts as 0
                rows[-1]["score"] = 0.0
        print(model, "scored", len(metas), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(f"{H}/results", exist_ok=True)
    df.to_parquet(f"{H}/results/{EXP}_gen.parquet")
    cells = df.groupby(["model", "task", "sampler", "T"]).agg(n=("item", "size"), score=("score", "mean"),
                                                              partial=("partial", "mean"),
                                                              extracted=("extracted", "mean"),
                                                              length=("n_tokens", "mean")).reset_index()
    cells.to_parquet(f"{H}/results/{EXP}_cells.parquet")
    pd.set_option("display.width", 220, "display.max_rows", 600)
    print(df.groupby(["model", "sampler", "T"]).score.mean().unstack("T").round(3).to_string())


if __name__ == "__main__":
    main()
