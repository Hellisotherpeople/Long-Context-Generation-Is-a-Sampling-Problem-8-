"""Evaluator-model perplexity of generated texts, per 1K-token window of the generator.

Loops -> very low NLL; word salad / incoherence -> very high NLL.  Using one fixed external
base model as evaluator makes values comparable across generator models/tokenizers.

python -m lcs.evalppl SPEC.json
SPEC = {"evaluator": path, "name": "q35a3b", "tp": 1, "runs": {run_dir: generator_tokenizer_path}}
Writes <run>/evalppl_<name>.jsonl : {"uid", "win_nll": [...], "nll": float, "n_eval_tokens": int}
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

W = 1024


def main(spec_path):
    spec = json.load(open(spec_path))
    from .tok import load_tokenizer
    from vllm import LLM, SamplingParams
    etok = load_tokenizer(spec["evaluator"])
    llm = None
    maxlen = int(spec.get("max_model_len", 40960))
    for run, gtok_path in spec["runs"].items():
        gtok = load_tokenizer(gtok_path)
        outp = f"{run}/evalppl_{spec['name']}.jsonl"
        done = set()
        if os.path.exists(outp):
            done = {json.loads(l)["uid"] for l in open(outp)}
        uids = sorted(os.path.basename(f)[:-4] for f in os.listdir(f"{run}/gen") if f.endswith(".npz"))
        uids = [os.path.basename(u) for u in uids if u not in done]
        print(f"[evalppl] {run}: {len(uids)} to score", flush=True)
        if not uids:
            continue
        if llm is None:
            llm = LLM(model=spec["evaluator"], tensor_parallel_size=int(spec.get("tp", 1)), trust_remote_code=True,
                      max_model_len=maxlen, gpu_memory_utilization=0.88, enable_prefix_caching=False,
                      max_num_seqs=32, max_num_batched_tokens=32768)
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False)
        B = 64
        with open(outp, "a") as f:
            for b in range(0, len(uids), B):
                batch = []
                for u in uids[b:b + B]:
                    toks = np.load(f"{run}/gen/{u}.npz")["tokens"]
                    nw = (len(toks) + W - 1) // W
                    # char boundaries of generator windows
                    pieces = [gtok.decode(toks[i * W:(i + 1) * W], skip_special_tokens=True) for i in range(nw)]
                    text = "".join(pieces)
                    bounds = np.cumsum([len(p) for p in pieces])
                    enc = etok(text, add_special_tokens=True, return_offsets_mapping=True)
                    ids = enc["input_ids"][: maxlen - 2]
                    offs = enc["offset_mapping"][: len(ids)]
                    batch.append((u, ids, offs, bounds, nw))
                outs = llm.generate([{"prompt_token_ids": ids} for _, ids, _, _, _ in batch], sp, use_tqdm=False)
                for (u, ids, offs, bounds, nw), o in zip(batch, outs):
                    pl = o.prompt_logprobs
                    nll = np.full(len(ids), np.nan)
                    for i in range(1, len(ids)):
                        d = pl[i]
                        if d is not None and ids[i] in d:
                            nll[i] = -d[ids[i]].logprob
                    starts = np.array([s for s, _ in offs])
                    widx = np.searchsorted(bounds, starts, side="right")
                    win = []
                    for w in range(nw):
                        v = nll[(widx == w)]
                        v = v[np.isfinite(v)]
                        win.append(float(v.mean()) if len(v) else None)
                    tot = nll[np.isfinite(nll)]
                    f.write(json.dumps({"uid": u, "win_nll": win, "nll": float(tot.mean()) if len(tot) else None,
                                        "n_eval_tokens": int(len(ids))}) + "\n")
                f.flush()
                print(f"[evalppl] {run}: {min(b + B, len(uids))}/{len(uids)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
