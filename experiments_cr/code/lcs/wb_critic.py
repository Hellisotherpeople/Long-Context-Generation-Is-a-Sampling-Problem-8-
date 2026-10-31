"""Score WritingBench runs with the released critic model (AQuarterMile/WritingBench-Critic-Model-Qwen-7B).

Runs in the vLLM environment (stdlib + vllm + transformers).  Official prompts (WritingBench/prompt.py: evaluate_system,
evaluate_prompt) with each of the 5 query-specific criteria; the critic is decoded greedily (the official script samples
at T=1, which would add judge noise to a sampler comparison).
python -m lcs.wb_critic <run_dir>   -> <run>/critic.jsonl  {uid, criterion, score}
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

H = os.path.expanduser("~/lcg-cr")
B = f"{H}/data/bench/WritingBench"
CRITIC = f"{H}/models/AQuarterMile__WritingBench-Critic-Model-Qwen-7B"


def main(run):
    sys.path.insert(0, B)
    from prompt import evaluate_prompt, evaluate_system
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    bench = {f"wb-{r['index']}": r for r in (json.loads(l) for l in open(f"{B}/benchmark_query/benchmark_all.jsonl"))}
    metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
    tok = AutoTokenizer.from_pretrained(metas[0]["model"], trust_remote_code=True)
    out_f = f"{run}/critic.jsonl"
    done = set()
    if os.path.exists(out_f):
        done = {(json.loads(l)["uid"], json.loads(l)["criterion"]) for l in open(out_f)}
    convs, keys = [], []
    for m in metas:
        q = bench[m["meta"]["pid"]]
        resp = tok.decode(np.load(f"{run}/gen/{m['uid']}.npz")["tokens"], skip_special_tokens=True)
        for c in q["checklist"]:
            k = (m["uid"], c.get("name", str(c)))
            if k in done:
                continue
            convs.append([{"role": "system", "content": evaluate_system},
                          {"role": "user", "content": evaluate_prompt.format(criteria=c, query=q["query"], response=resp)}])
            keys.append(k)
    print("critic calls", len(convs), flush=True)
    if not convs:
        return
    llm = LLM(model=CRITIC, max_model_len=32768, gpu_memory_utilization=0.88)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)
    with open(out_f, "a") as f:
        for i in range(0, len(convs), 2000):
            outs = llm.chat(convs[i:i + 2000], sp, use_tqdm=False)
            for (uid, cname), o in zip(keys[i:i + 2000], outs):
                txt = o.outputs[0].text
                mm = re.search(r'"score"\s*:\s*(\d+)', txt)
                f.write(json.dumps({"uid": uid, "criterion": cname, "score": int(mm.group(1)) if mm else None}) + "\n")
            f.flush()
            print("done", i + len(outs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
