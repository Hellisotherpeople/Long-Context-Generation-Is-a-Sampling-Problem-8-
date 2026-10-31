"""Loop-rescue intervention: does switching ONLY the sampler free a generation that is already stuck in a loop?

For base-model book continuation (exact token-level prefixes) we take generations that entered a terminal loop, cut them
1K tokens after loop onset (inside the loop), and continue the identical token prefix for 4K tokens with a panel of
samplers.  Same model, same context, different sampler: if the loop is a property of the sampler, the escape rate depends
on the sampler; if it is an unrecoverable model state, nothing escapes.

python -m lcs.exp_rescue --models qwen3.5-4b-base,qwen3.5-9b-base --per-model 24 --prio 043
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

H = os.path.expanduser("~/lcg-cr")
DRY = {"mult": 0.8, "base": 1.75, "allowed": 2}
PANEL = [
    ("same", None, None),                       # the sampler that produced the loop, same temperature (control)
    ("greedy", 1.0, [["greedy", 0.0]]),
    ("temp@T1.0", 1.0, []),
    ("topk50@T1.0", 1.0, [["top_k", 50]]),
    ("topp0.9@T1.0", 1.0, [["top_p", 0.9]]),
    ("topp0.9@T1.3", 1.3, [["top_p", 0.9]]),
    ("minp0.1@T1.3", 1.3, [["min_p", 0.1]]),
    ("minp0.1@T1.6", 1.6, [["min_p", 0.1]]),
    ("nsigma1@T1.3", 1.3, [["top_nsigma", 1.0]]),
    ("nsigma1+k2@T1.3", 1.3, [["top_nsigma", 1.0], ["keep_top", 2]]),
    ("pless+k2@T1.3", 1.3, [["p_less", 0.0], ["keep_top", 2]]),
    ("toph0.4@T1.6", 1.6, [["top_h", 0.4]]),
    ("dry+topp0.9@T1.0", 1.0, [["top_p", 0.9]]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3.5-4b-base,qwen3.5-9b-base,gemma4-12b-base")
    ap.add_argument("--per-model", type=int, default=24)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--cont", type=int, default=4096)
    ap.add_argument("--prio", default="043")
    a = ap.parse_args()
    rng = random.Random(0)
    for model in a.models.split(","):
        src = json.load(open(f"{H}/jobs/book32k/{model}.json"))
        prompt_of = {r["uid"]: r["prompt"]["file"] for r in src["requests"]}
        rows = [json.loads(l) for l in open(f"{H}/runs/book32k/{model}/metrics.jsonl")]
        looped = [r for r in rows if r.get("loop_onset_win") is not None and r["T"] in (0.7, 1.0)
                  and r["sampler"] != "greedy" and (r["loop_onset_win"] + 2) * 1024 < 28000]
        rng.shuffle(looped)
        # balance over source samplers
        by_s = {}
        for r in looped:
            by_s.setdefault(r["sampler"], []).append(r)
        chosen, k = [], 0
        while len(chosen) < a.per_model and any(by_s.values()):
            s = sorted(by_s)[k % len(by_s)]
            if by_s[s]:
                chosen.append(by_s[s].pop())
            k += 1
        os.makedirs(f"{H}/data/rescue/{model}", exist_ok=True)
        reqs = []
        for r in chosen:
            cut = (r["loop_onset_win"] + 1) * 1024 + 1024
            gen = np.load(f"{H}/runs/book32k/{model}/gen/{r['uid']}.npz")["tokens"].astype(np.int64)
            pre = np.load(prompt_of[r["uid"]]).astype(np.int64)
            f = f"{H}/data/rescue/{model}/{r['uid']}.npy"
            np.save(f, np.concatenate([pre, gen[:cut]]))
            for name, T, ops in PANEL:
                if name == "same":
                    lcs = dict(r["lcs"])
                else:
                    lcs = {"T": T, "ops": ops}
                    if name.startswith("dry"):
                        lcs["dry"] = DRY
                for sd in range(a.samples if name != "greedy" else 1):
                    reqs.append({"uid": f"{r['uid']}__{name}__s{sd}", "prompt": {"type": "tokens", "file": f},
                                 "max_tokens": a.cont, "force_len": True, "lcs": lcs,
                                 "meta": {"pid": r["pid"], "src_uid": r["uid"], "src_sampler": r["sampler"],
                                          "src_T": r["T"], "cut": cut, "sampler": name, "T": lcs["T"], "seed": sd}})
        eng = dict(src["engine"])
        job = {"job_id": f"rescue/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/rescue/{model}", "requests": reqs}
        os.makedirs(f"{H}/jobs/rescue", exist_ok=True)
        json.dump(job, open(f"{H}/jobs/rescue/{model}.json", "w"))
        env = {}
        for d in ("done",):
            for fn in os.listdir(f"{H}/queue/{d}"):
                if fn.endswith(f"_book32k_{model}.json"):
                    env = json.load(open(f"{H}/queue/{d}/{fn}")).get("env", {}) or env
        q = {"job": f"{H}/jobs/rescue/{model}.json", "gpus": int(src.get("tp", 1)), "model_dir": src["model"], "kind": "gen"}
        if env:
            q["env"] = env
        json.dump(q, open(f"{H}/queue/pending/{a.prio}_rescue_{model}.json", "w"))
        print(model, "sources", len(chosen), "requests", len(reqs), sorted({r['sampler'] for r in chosen}))


if __name__ == "__main__":
    main()
