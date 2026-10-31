"""Quality-diversity frontier: many samples per open-ended prompt, every rule across temperature.

Prompts: NoveltyBench (curated 100 + 100 WildChat) and Infinity-Chat100 (Jiang et al., NeurIPS'25).  For each prompt we draw
`samples` responses (natural end of turn, <= max_tokens) per configuration.  Diversity is measured across the samples of a
prompt (lcs.diversity); quality by a reward model and LLM judges.

python -m lcs.exp_divx --models qwen3.5-4b,llama3.1-8b --prio 045
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

H = os.path.expanduser("~/lcg-cr")
B = f"{H}/data/bench"
RULES = [("temp", []), ("topk50", [["top_k", 50]]), ("topp0.9", [["top_p", 0.9]]), ("topp0.95", [["top_p", 0.95]]),
         ("minp0.05", [["min_p", 0.05]]), ("minp0.1", [["min_p", 0.1]]), ("toph0.4", [["top_h", 0.4]]),
         ("pless", [["p_less", 0.0]]), ("nsigma1", [["top_nsigma", 1.0]]),
         ("toph0.4+k2", [["top_h", 0.4], ["keep_top", 2]]), ("pless+k2", [["p_less", 0.0], ["keep_top", 2]]),
         ("nsigma1+k2", [["top_nsigma", 1.0], ["keep_top", 2]])]
TEMPS = [0.7, 1.0, 1.3, 1.6, 2.0]


def load_prompts(n_wild=100):
    out = []
    nb = f"{B}/yimingzhang__novelty-bench"
    for split, cap in (("curated", 10 ** 9), ("wildchat", n_wild)):
        fs = sorted(glob.glob(f"{nb}/**/*{split}*.parquet", recursive=True)) + sorted(glob.glob(f"{nb}/**/*{split}*.jsonl", recursive=True))
        if not fs:
            raise SystemExit(f"NoveltyBench split {split} not found under {nb}")
        df = pd.concat([pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f, lines=True) for f in fs])
        for _, r in df.head(cap).iterrows():
            out.append({"pid": f"nb-{split}-{r['id']}", "src": f"novelty-{split}", "prompt": r["prompt"]})
    ic = sorted(glob.glob(f"{B}/liweijiang__infinite-chats-eval/**/*.parquet", recursive=True)) + \
        sorted(glob.glob(f"{B}/liweijiang__infinite-chats-eval/**/*.jsonl", recursive=True))
    df = pd.concat([pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f, lines=True) for f in ic])
    for i, r in enumerate(df.itertuples()):
        out.append({"pid": f"ic-{i}", "src": "infinity-chat", "prompt": getattr(r, "query")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--prio", default="045")
    a = ap.parse_args()
    prompts = load_prompts()
    os.makedirs(f"{H}/jobs/divx", exist_ok=True)
    for model in a.models.split(","):
        tmpl = json.load(open(f"{H}/jobs/mt32k/{model}.json"))
        ck = tmpl["requests"][0]["prompt"].get("chat_kwargs") or {}
        sysmsgs = [m for m in tmpl["requests"][0]["prompt"]["messages"] if m["role"] == "system"]
        reqs = []
        for p in prompts:
            msgs = sysmsgs[:0] + [{"role": "user", "content": p["prompt"]}]   # no novelist system prompt here
            base = {"prompt": {"type": "chat", "messages": msgs, "chat_kwargs": ck}, "max_tokens": a.max_tokens,
                    "force_len": False}
            reqs.append(dict(base, uid=f"{p['pid']}__greedy__s0", lcs={"T": 1.0, "ops": [["greedy", 0.0]]},
                             meta={"pid": p["pid"], "src": p["src"], "sampler": "greedy", "T": 0.0, "seed": 0}))
            for name, ops in RULES:
                for T in TEMPS:
                    for sd in range(a.samples):
                        reqs.append(dict(base, uid=f"{p['pid']}__{name}@T{T}__s{sd}", lcs={"T": T, "ops": ops},
                                         meta={"pid": p["pid"], "src": p["src"], "sampler": name, "T": T, "seed": sd}))
        eng = dict(tmpl["engine"])
        eng["max_model_len"] = 8192
        job = {"job_id": f"divx/{model}", "model": tmpl["model"], "tp": tmpl.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/divx/{model}", "requests": reqs, "max_len": 8192}
        jf = f"{H}/jobs/divx/{model}.json"
        json.dump(job, open(jf, "w"))
        env = {}
        for d in ("done",):
            for fn in os.listdir(f"{H}/queue/{d}"):
                if fn.endswith(f"_mt32k_{model}.json"):
                    env = json.load(open(f"{H}/queue/{d}/{fn}")).get("env", {}) or env
        q = {"job": jf, "gpus": int(tmpl.get("tp", 1)), "model_dir": tmpl["model"], "kind": "gen"}
        if env:
            q["env"] = env
        json.dump(q, open(f"{H}/queue/pending/{a.prio}_divx_{model}.json", "w"))
        print(model, "prompts", len(prompts), "requests", len(reqs))


if __name__ == "__main__":
    main()
