"""Queue SWE-bench Verified agent sweeps (lcs.swe_job), one scheduler job per model x config-chunk.

python3 tools/make_swe.py <prio> <model> [<model> ...] [--think] [--ids file] [--chunks 3] [--set main]
"""
import argparse
import json
import os

H = os.path.expanduser("~/lcg-cr")
M = f"{H}/models"
VPY = "/home/ubuntu/glm53-vllm/.venv/bin/python"
MODELS = {  # key: (dir, tp, non-thinking chat kwargs, thinking chat kwargs, vendor (T, ops, extra) non-thinking/thinking)
    "qwen3.5-9b": ("Qwen__Qwen3.5-9B", 1),
    "qwen3.5-4b": ("Qwen__Qwen3.5-4B", 1),
    "qwen3.5-27b": ("Qwen__Qwen3.5-27B", 1),
    "qwen3.5-35b-a3b": ("Qwen__Qwen3.5-35B-A3B", 1),
    "qwen3.6-27b": ("Qwen__Qwen3.6-27B", 1),
    "qwen3.8-27b": ("Qwen__Qwen3.8-27B", 1),
    "qwen3.5-122b-a10b": ("Qwen__Qwen3.5-122B-A10B", 4),
    "nemotron3-30b-a3b": ("nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 1),
    "gemma4-26b-a4b": ("google__gemma-4-26B-A4B-it", 1),
    "gemma4-31b": ("google__gemma-4-31B-it", 1),
}
QWEN_VENDOR = {False: (0.7, [["top_k", 20], ["top_p", 0.8]], {"pen": {"presence": 1.5}}),
               True: (0.6, [["top_k", 20], ["top_p", 0.95]], {})}
VENDOR = {"nemotron3-30b-a3b": {False: (0.6, [["top_p", 0.95]], {}), True: (1.0, [], {})},
          "gemma4-26b-a4b": {False: (1.0, [["top_k", 64], ["top_p", 0.95]], {}), True: (1.0, [["top_k", 64], ["top_p", 0.95]], {})},
          "gemma4-31b": {False: (1.0, [["top_k", 64], ["top_p", 0.95]], {}), True: (1.0, [["top_k", 64], ["top_p", 0.95]], {})}}

SETS = {
    "main": [("greedy", 1.0, [["greedy", 0.0]]),
             ("temp@T0.7", 0.7, []), ("temp@T1.0", 1.0, []),
             ("topp0.95@T1.0", 1.0, [["top_p", 0.95]]),
             ("minp0.1@T1.0", 1.0, [["min_p", 0.1]]), ("nsigma1@T1.0", 1.0, [["top_nsigma", 1.0]]),
             ("pless@T1.0", 1.0, [["p_less", 0.0]]), ("toph0.4@T1.0", 1.0, [["top_h", 0.4]]),
             ("topp0.95@T1.5", 1.5, [["top_p", 0.95]]),
             ("minp0.1@T1.5", 1.5, [["min_p", 0.1]]), ("nsigma1@T1.5", 1.5, [["top_nsigma", 1.0]])],
    "think": [("topp0.95@T1.0", 1.0, [["top_p", 0.95]]), ("temp@T0.6", 0.6, []),
              ("minp0.1@T1.0", 1.0, [["min_p", 0.1]]), ("nsigma1@T1.0", 1.0, [["top_nsigma", 1.0]]),
              ("pless@T1.0", 1.0, [["p_less", 0.0]]), ("topp0.95@T1.5", 1.5, [["top_p", 0.95]]),
              ("minp0.1@T1.5", 1.5, [["min_p", 0.1]])],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prio")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--set", default="")
    ap.add_argument("--workers", type=int, default=12)  # 48 agents/GPU keeps the KV working set in cache
    ap.add_argument("--step-limit", type=int, default=250)
    ap.add_argument("--max-model-len", type=int, default=131072)
    a = ap.parse_args()
    os.makedirs(f"{H}/jobs/swe", exist_ok=True)
    for model in a.models:
        d, tp = MODELS[model]
        think = a.think
        T, ops, extra = (VENDOR.get(model) or QWEN_VENDOR)[think]
        configs = [list(c) for c in SETS[a.set or ("think" if think else "main")]] + [["vendor", T, ops, extra]]
        ck = {"enable_thinking": think}
        if model.startswith("gemma"):
            ck = {"enable_thinking": think} if think else {}
        tag = model + ("_think" if think else "")
        out = f"{H}/runs/swe/{tag}"
        n = a.chunks
        for c in range(n):
            part = configs[c::n]
            J = {"model": f"{M}/{d}", "served": model, "tp": tp, "max_model_len": a.max_model_len,
                 "chat_kwargs": ck, "ids": a.ids, "configs": part, "parallel_configs": len(part),
                 "workers": a.workers, "step_limit": a.step_limit,
                 "max_tokens": 32768 if think else 8192, "out": out}
            if think:
                J["engine_args"] = ["--reasoning-parser", "qwen3"] if model.startswith("qwen") else []
            jf = f"{H}/jobs/swe/{tag}__c{c}.json"
            json.dump(J, open(jf, "w"), indent=1)
            # gated on the image pull (tools/pull_swe_images.sh touches data/swe/images/.done)
            q = {"job": jf, "gpus": tp, "model_dir": f"{H}/data/swe/images", "kind": "py",
                 "args": ["-c", f"import os; os.execv('{VPY}', ['{VPY}', '-m', 'lcs.swe_job', '{jf}'])"]}
            json.dump(q, open(f"{H}/queue/pending/{a.prio}_swe_{tag}__c{c}.json", "w"))
            print(tag, c, [p[0] for p in part])


if __name__ == "__main__":
    main()
