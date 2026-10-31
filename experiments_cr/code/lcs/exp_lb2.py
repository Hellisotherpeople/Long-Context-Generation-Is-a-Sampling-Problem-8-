"""LongBench v2 (Bai et al., 2025) under stochastic decoding: 503 long-context multiple-choice questions, official 2-stage
CoT protocol.  Stage 1 (this file): chain-of-thought (<=1024 tokens) with the sampler under test, contexts middle-truncated
to `max_ctx` tokens exactly as in THUDM/LongBench pred.py.  Stage 2 (lcs.ana_lb2 --stage2): greedy answer extraction
from the CoT with the official 0shot_cot_ans template (context omitted), so differences come only from the sampled
reasoning.

python -m lcs.exp_lb2 --models qwen3.5-9b,nemotron3-30b-a3b --max-ctx 120000 --prio 046
"""
from __future__ import annotations

import argparse
import json
import os

H = os.path.expanduser("~/lcg-cr")
B = f"{H}/data/bench"
CONFIGS = [("greedy", 1.0, [["greedy", 0.0]])] + [
    (f"{n}@T{T}", T, ops) for T in (0.7, 1.0, 1.5) for n, ops in
    [("temp", []), ("topk50", [["top_k", 50]]), ("topp0.9", [["top_p", 0.9]]), ("topp0.95", [["top_p", 0.95]]),
     ("minp0.1", [["min_p", 0.1]]), ("toph0.4", [["top_h", 0.4]]), ("pless", [["p_less", 0.0]]),
     ("nsigma1", [["top_nsigma", 1.0]]), ("pless+k2", [["p_less", 0.0], ["keep_top", 2]]),
     ("nsigma1+k2", [["top_nsigma", 1.0], ["keep_top", 2]])]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--max-ctx", type=int, default=120000)
    ap.add_argument("--prio", default="046")
    ap.add_argument("--tag", default="lb2")
    a = ap.parse_args()
    from .tok import load_tokenizer
    data = json.load(open(f"{B}/zai-org__LongBench-v2/data.json"))
    tmpl = open(f"{B}/LongBench/prompts/0shot_cot.txt").read()
    for model in a.models.split(","):
        src = json.load(open(f"{H}/jobs/core32k/{model}.json"))
        ck = src["requests"][0]["prompt"].get("chat_kwargs") or {}
        tok = load_tokenizer(src["model"])
        os.makedirs(f"{H}/data/{a.tag}/{model}", exist_ok=True)
        reqs = []
        import numpy as np
        for d in data:
            text = tmpl.replace("$DOC$", d["context"].strip()).replace("$Q$", d["question"].strip()) \
                .replace("$C_A$", d["choice_A"].strip()).replace("$C_B$", d["choice_B"].strip()) \
                .replace("$C_C$", d["choice_C"].strip()).replace("$C_D$", d["choice_D"].strip())
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > a.max_ctx:
                ids = ids[:a.max_ctx // 2] + ids[-a.max_ctx // 2:]
                text = tok.decode(ids, skip_special_tokens=True)
            # render the chat prompt once and store token ids; requests reference the file (no per-request copies)
            rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                               add_generation_prompt=True, **ck)
            pids = tok(rendered, add_special_tokens=False)["input_ids"]
            f = f"{H}/data/{a.tag}/{model}/{d['_id']}.npy"
            np.save(f, np.asarray(pids, dtype=np.int32))
            prompt = {"type": "tokens", "file": f}
            for name, T, ops in CONFIGS:
                reqs.append({"uid": f"{d['_id']}__{name}__s0", "prompt": prompt, "max_tokens": 1024, "force_len": False,
                             "lcs": {"T": T, "ops": ops},
                             "meta": {"pid": d["_id"], "difficulty": d["difficulty"], "length": d["length"],
                                      "domain": d["domain"], "answer": d["answer"], "sampler": name.split("@")[0],
                                      "T": 0.0 if name == "greedy" else T, "n_ctx": len(pids)}})
        eng = {"max_model_len": a.max_ctx + 4096, "max_num_seqs": 256, "enable_prefix_caching": True,
               "max_num_batched_tokens": 16384}
        for k in ("language_model_only", "tokenizer_mode", "config_format", "load_format"):  # model-specific engine flags
            if k in src.get("engine", {}):
                eng[k] = src["engine"][k]
        job = {"job_id": f"{a.tag}/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/{a.tag}/{model}", "requests": reqs, "max_len": a.max_ctx + 2048}
        os.makedirs(f"{H}/jobs/{a.tag}", exist_ok=True)
        jf = f"{H}/jobs/{a.tag}/{model}.json"
        json.dump(job, open(jf, "w"))
        env = {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
        for fn in os.listdir(f"{H}/queue/done"):
            if fn.endswith(f"_core32k_{model}.json"):
                env.update(json.load(open(f"{H}/queue/done/{fn}")).get("env", {}) or {})
        q = {"job": jf, "gpus": int(src.get("tp", 1)), "model_dir": src["model"], "kind": "gen", "env": env}
        json.dump(q, open(f"{H}/queue/pending/{a.prio}_{a.tag}_{model}.json", "w"))
        print(model, "requests", len(reqs))


if __name__ == "__main__":
    main()
