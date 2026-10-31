"""Score token sequences with prompt logprobs (no generation).

python -m lcs.score SPEC.json
SPEC = {"model": path, "tp": 1, "max_model_len": N, "out": path.npz-dir,
        "items": [{"id": str, "file": ids.npy, "start": int}] | "texts": [{"id","text"}],
        "topk": 20}
For each item saves <out>/<id>.npz: lp [n], rank [n], ent20 [n] (entropy estimate from the
top-20 logprobs + lumped tail), for positions >= start.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np


def main(spec_path):
    spec = json.load(open(spec_path))
    out = spec["out"]
    os.makedirs(out, exist_ok=True)
    from .tok import load_tokenizer
    from vllm import LLM, SamplingParams
    tok = load_tokenizer(spec["model"])
    items = []
    for it in spec.get("items", []):
        if not os.path.exists(f"{out}/{it['id']}.npz"):
            items.append((it["id"], np.load(it["file"]).astype(np.int64).tolist(), int(it.get("start", 1))))
    for it in spec.get("texts", []):
        if not os.path.exists(f"{out}/{it['id']}.npz"):
            ids = tok(it["text"], add_special_tokens=True)["input_ids"][: spec["max_model_len"] - 2]
            items.append((it["id"], ids, int(it.get("start", 1))))
    print(f"[score] {len(items)} items", flush=True)
    if not items:
        return
    k = int(spec.get("topk", 20))
    llm = LLM(model=spec["model"], tensor_parallel_size=int(spec.get("tp", 1)), trust_remote_code=True,
              max_model_len=int(spec["max_model_len"]), gpu_memory_utilization=float(spec.get("gpu_mem", 0.85)), enable_prefix_caching=False,
              max_num_seqs=64, **spec.get("engine", {}))
    sp = SamplingParams(max_tokens=1, prompt_logprobs=k, temperature=1.0, detokenize=False)
    B = int(spec.get("batch", 16))
    for b in range(0, len(items), B):
        chunk = items[b:b + B]
        outs = llm.generate([{"prompt_token_ids": ids} for _, ids, _ in chunk], sp, use_tqdm=False)
        for (iid, ids, start), o in zip(chunk, outs):
            pl = o.prompt_logprobs
            n = len(ids)
            lp = np.full(n, np.nan, np.float32)
            rk = np.full(n, -1, np.int64)
            ent = np.full(n, np.nan, np.float32)
            for i in range(max(1, start), n):
                d = pl[i]
                if d is None:
                    continue
                t = ids[i]
                if t in d:
                    lp[i] = d[t].logprob
                    rk[i] = d[t].rank
                tops = sorted((v.logprob for v in d.values()), reverse=True)[:k]
                ps = np.exp(np.array(tops, np.float64))
                tail = max(0.0, 1.0 - ps.sum())
                ent[i] = float(-(ps * np.log(np.maximum(ps, 1e-300))).sum() + (0 if tail <= 0 else -tail * math.log(tail)))
            np.savez(f"{out}/{iid}.npz", lp=lp[start:], rank=rk[start:], ent20=ent[start:], start=start)
        print(f"[score] {min(b + B, len(items))}/{len(items)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
