"""Loop rescue in natural multi-turn generation: resume a conversation from inside a loop with a different sampler.

The conversation is rebuilt up to the looping turn (earlier turns decoded, "Continue." between them) and the looping turn
is pre-filled up to 1K tokens after loop onset; the model then continues for up to 4K tokens and may end its turn.

python -m lcs.exp_rescue_mt --models qwen3.5-4b,qwen3.5-9b --per-model 24 --prio 044
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from .exp_rescue import DRY, PANEL

H = os.path.expanduser("~/lcg-cr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3.5-4b,qwen3.5-9b,qwen3.5-27b,qwen3.5-35b-a3b,nemotron3-30b-a3b")
    ap.add_argument("--per-model", type=int, default=24)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--cont", type=int, default=4096)
    ap.add_argument("--prio", default="044")
    a = ap.parse_args()
    from .tok import load_tokenizer
    rng = random.Random(0)
    for model in a.models.split(","):
        src = json.load(open(f"{H}/jobs/mt32k/{model}.json"))
        req_of = {r["uid"]: r for r in src["requests"]}
        tok = load_tokenizer(src["model"])
        rows = [json.loads(l) for l in open(f"{H}/runs/mt32k/{model}/metrics.jsonl")]
        looped = [r for r in rows if r.get("loop_onset_win") is not None and r["T"] in (0.7, 1.0)
                  and r["sampler"] != "greedy" and r["uid"] in req_of and r.get("part_lens")]
        rng.shuffle(looped)
        by_s = {}
        for r in looped:
            by_s.setdefault(r["sampler"], []).append(r)
        chosen, k = [], 0
        while len(chosen) < a.per_model and any(by_s.values()):
            s = sorted(by_s)[k % len(by_s)]
            if by_s[s]:
                chosen.append(by_s[s].pop())
            k += 1
        reqs = []
        for r in chosen:
            base = req_of[r["uid"]]
            cut = (r["loop_onset_win"] + 1) * 1024 + 1024
            pl = r["part_lens"]
            cum, t = 0, 0
            while t < len(pl) and cum + pl[t] <= cut:
                cum += pl[t]
                t += 1
            if t >= len(pl):
                continue
            msgs = list(base["prompt"]["messages"])
            for j in range(t):
                prev = np.load(f"{H}/runs/mt32k/{model}/gen/{r['uid']}_t{j}.npz")["tokens"]
                msgs.append({"role": "assistant", "content": tok.decode(prev, skip_special_tokens=True)})
                msgs.append({"role": "user", "content": base["multiturn"].get("continue_msg", "Continue.")})
            cur = np.load(f"{H}/runs/mt32k/{model}/gen/{r['uid']}_t{t}.npz")["tokens"]
            prefill = tok.decode(cur[:max(1, cut - cum)], skip_special_tokens=True)
            prompt = {"type": "chat", "messages": msgs, "chat_kwargs": base["prompt"].get("chat_kwargs") or {},
                      "prefill": prefill}
            for name, T, ops in PANEL:
                if name == "same":
                    lcs = dict(r["lcs"])
                else:
                    lcs = {"T": T, "ops": ops}
                    if name.startswith("dry"):
                        lcs["dry"] = DRY
                for sd in range(a.samples if name != "greedy" else 1):
                    reqs.append({"uid": f"{r['uid']}__{name}__s{sd}", "prompt": prompt, "max_tokens": a.cont,
                                 "force_len": False, "lcs": lcs,
                                 "meta": {"pid": r["pid"], "src_uid": r["uid"], "src_sampler": r["sampler"],
                                          "src_T": r["T"], "cut": cut, "turn": t, "sampler": name, "T": lcs["T"],
                                          "seed": sd}})
        job = {"job_id": f"rescue_mt/{model}", "model": src["model"], "tp": src.get("tp", 1),
               "engine": dict(src["engine"]), "out_dir": f"{H}/runs/rescue_mt/{model}", "requests": reqs}
        os.makedirs(f"{H}/jobs/rescue_mt", exist_ok=True)
        json.dump(job, open(f"{H}/jobs/rescue_mt/{model}.json", "w"))
        q = {"job": f"{H}/jobs/rescue_mt/{model}.json", "gpus": int(src.get("tp", 1)), "model_dir": src["model"], "kind": "gen"}
        json.dump(q, open(f"{H}/queue/pending/{a.prio}_rescue_mt_{model}.json", "w"))
        print(model, "sources", len(chosen), "requests", len(reqs), sorted({x['sampler'] for x in chosen}))


if __name__ == "__main__":
    main()
