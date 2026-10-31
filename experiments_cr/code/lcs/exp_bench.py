"""Generic sampler-grid jobs for prompt benchmarks with natural stopping.

  lbwrite : LongBench-Write (Bai et al., 2024; THUDM/LongWriter evaluation/longbench_write_en.jsonl), 60 English prompts with
            required lengths of 100-20,000 words; scored by length adherence (S_l) and judged quality (S_q).
  wbench  : WritingBench (Wu et al., 2025; X-PLUG/WritingBench), 555 English queries with 5 query-specific criteria each;
            scored by the released WritingBench critic model.

python -m lcs.exp_bench lbwrite --models qwen3.5-9b --prio 048
"""
from __future__ import annotations

import argparse
import json
import os

H = os.path.expanduser("~/lcg-cr")
B = f"{H}/data/bench"
RULES = [("temp", []), ("topk50", [["top_k", 50]]), ("topp0.9", [["top_p", 0.9]]), ("topp0.95", [["top_p", 0.95]]),
         ("minp0.05", [["min_p", 0.05]]), ("minp0.1", [["min_p", 0.1]]), ("toph0.4", [["top_h", 0.4]]),
         ("pless", [["p_less", 0.0]]), ("nsigma1", [["top_nsigma", 1.0]]),
         ("toph0.4+k2", [["top_h", 0.4], ["keep_top", 2]]), ("pless+k2", [["p_less", 0.0], ["keep_top", 2]]),
         ("nsigma1+k2", [["top_nsigma", 1.0], ["keep_top", 2]])]
SPEC = {
    "lbwrite": {"temps": [0.7, 1.0, 1.3], "rules": [r[0] for r in RULES], "max_tokens": 32000, "max_len": 34816},
    "wbench": {"temps": [0.7, 1.0, 1.3], "rules": ["topp0.9", "minp0.1", "toph0.4", "pless", "nsigma1", "pless+k2",
                                                    "nsigma1+k2", "temp"], "max_tokens": 12000, "max_len": 20480},
}


def load(name):
    if name == "lbwrite":
        rows = [json.loads(l) for l in open(f"{B}/LongWriter/evaluation/longbench_write_en.jsonl")]
        return [{"pid": f"lbw-{i}", "prompt": r["prompt"], "meta": {"type": r.get("type"), "length": r.get("length")}}
                for i, r in enumerate(rows)]
    if name == "wbench":
        rows = [json.loads(l) for l in open(f"{B}/WritingBench/benchmark_query/benchmark_all.jsonl")]
        return [{"pid": f"wb-{r['index']}", "prompt": r["query"],
                 "meta": {"domain1": r.get("domain1"), "domain2": r.get("domain2"), "checklist": r.get("checklist")}}
                for r in rows if r.get("lang") == "en"]
    raise SystemExit(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--models", required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--prio", default="048")
    a = ap.parse_args()
    sp = SPEC[a.bench]
    items = load(a.bench)
    rules = [r for r in RULES if r[0] in sp["rules"]]
    for model in a.models.split(","):
        src = json.load(open(f"{H}/jobs/core32k/{model}.json"))
        ck = src["requests"][0]["prompt"].get("chat_kwargs") or {}
        reqs = []
        for it in items:
            p = {"type": "chat", "messages": [{"role": "user", "content": it["prompt"]}], "chat_kwargs": ck}
            meta = {k: v for k, v in it["meta"].items() if k != "checklist"}
            reqs.append({"uid": f"{it['pid']}__greedy__s0", "prompt": p, "max_tokens": sp["max_tokens"], "force_len": False,
                         "lcs": {"T": 1.0, "ops": [["greedy", 0.0]]},
                         "meta": dict(meta, pid=it["pid"], sampler="greedy", T=0.0, seed=0)})
            for name, ops in rules:
                for T in sp["temps"]:
                    for sd in range(a.samples):
                        reqs.append({"uid": f"{it['pid']}__{name}@T{T}__s{sd}", "prompt": p, "max_tokens": sp["max_tokens"],
                                     "force_len": False, "lcs": {"T": T, "ops": ops},
                                     "meta": dict(meta, pid=it["pid"], sampler=name, T=T, seed=sd)})
        eng = {"max_model_len": sp["max_len"], "max_num_seqs": 512, "enable_prefix_caching": True}
        job = {"job_id": f"{a.bench}/{model}", "model": src["model"], "tp": src.get("tp", 1), "engine": eng,
               "out_dir": f"{H}/runs/{a.bench}/{model}", "requests": reqs, "max_len": sp["max_len"]}
        os.makedirs(f"{H}/jobs/{a.bench}", exist_ok=True)
        jf = f"{H}/jobs/{a.bench}/{model}.json"
        json.dump(job, open(jf, "w"))
        env = {}
        for fn in os.listdir(f"{H}/queue/done"):
            if fn.endswith(f"_core32k_{model}.json"):
                env.update(json.load(open(f"{H}/queue/done/{fn}")).get("env", {}) or {})
        q = {"job": jf, "gpus": int(src.get("tp", 1)), "model_dir": src["model"], "kind": "gen"}
        if env:
            q["env"] = env
        json.dump(q, open(f"{H}/queue/pending/{a.prio}_{a.bench}_{model}.json", "w"))
        print(a.bench, model, "items", len(items), "requests", len(reqs))


if __name__ == "__main__":
    main()
