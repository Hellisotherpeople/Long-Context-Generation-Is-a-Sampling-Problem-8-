"""Long chain-of-thought code generation: LiveCodeBench v6 (175 problems, Jan-Apr 2025; official generic prompt), thinking
mode, 32K budget, natural EOS, same sampler grid as exp_reason2.  Scored by lcs.ana_lcb (official LCB tests, sandboxed).

python -m lcs.exp_lcb --models qwen3.5-9b,nemotron3-30b-a3b --samples 2 --prio 052
"""
from __future__ import annotations

import argparse
import json
import os

from .exp_reason import THINK
from .exp_reason2 import SAMPLERS

H = os.path.expanduser("~/lcg-cr")
B = f"{H}/data/bench"
SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
          "correct Python program that matches the specification and passes all tests.")
WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
NO_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample "
              "inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
              "inputs, runs the algorithm and writes output to STDOUT.")


def prompt(q):
    p = f"### Question:\n{q['question_content']}\n\n"
    if q["starter_code"]:
        p += f"### Format: {WITH_STARTER}\n```python\n{q['starter_code']}\n```\n\n"
    else:
        p += f"### Format: {NO_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    return p + "### Answer: (use the provided format with backticks)\n\n"


def problems(release="test6"):
    return [json.loads(l) for l in open(f"{B}/livecodebench__code_generation_lite/{release}.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--prio", default="052")
    a = ap.parse_args()
    probs = problems()
    for model in a.models.split(","):
        srcf = next(f for f in (f"{H}/jobs/reason/{model}.json", f"{H}/jobs/core32k/{model}.json") if os.path.exists(f))
        src = json.load(open(srcf))
        fam = next((k for k in THINK if model.startswith(k)), None)
        ck = dict(THINK.get(fam, {}))
        reqs = []
        for q in probs:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(q)}]
            if model.startswith("r1-distill"):  # DeepSeek: no system prompt
                msgs = [{"role": "user", "content": SYSTEM + "\n\n" + prompt(q)}]
            base = {"prompt": {"type": "chat", "messages": msgs, "chat_kwargs": ck}, "max_tokens": a.budget, "force_len": False}
            meta = {"pid": q["question_id"], "difficulty": q["difficulty"], "platform": q["platform"]}
            reqs.append(dict(base, uid=f"{q['question_id']}__greedy__s0", lcs={"T": 1.0, "ops": [["greedy", 0.0]]},
                             meta=dict(meta, sampler="greedy", T=0.0, i=0)))
            for name, ops, temps in SAMPLERS:
                for T in temps:
                    for i in range(a.samples):
                        reqs.append(dict(base, uid=f"{q['question_id']}__{name}@T{T}__s{i}", lcs={"T": T, "ops": ops},
                                         meta=dict(meta, sampler=name, T=T, i=i)))
        eng = {"max_model_len": a.budget + 4096, "max_num_seqs": 512, "enable_prefix_caching": True}
        for k in ("language_model_only", "tokenizer_mode", "config_format", "load_format"):  # model-specific engine flags
            if k in src.get("engine", {}):
                eng[k] = src["engine"][k]
        job = {"job_id": f"lcb/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/lcb/{model}", "requests": reqs}
        os.makedirs(f"{H}/jobs/lcb", exist_ok=True)
        jf = f"{H}/jobs/lcb/{model}.json"
        json.dump(job, open(jf, "w"))
        env = {}
        for fn in os.listdir(f"{H}/queue/done"):
            if fn.endswith(f"_core32k_{model}.json") or fn.endswith(f"_reason_{model}.json"):
                env.update(json.load(open(f"{H}/queue/done/{fn}")).get("env", {}) or {})
        q = {"job": jf, "gpus": int(src.get("tp", 1)), "model_dir": src["model"], "kind": "gen"}
        if env:
            q["env"] = env
        json.dump(q, open(f"{H}/queue/pending/{a.prio}_lcb_{model}.json", "w"))
        print(model, "problems", len(probs), "requests", len(reqs), "chat_kwargs", ck)


if __name__ == "__main__":
    main()
