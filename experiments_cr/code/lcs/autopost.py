"""Watch the scheduler's done/ folder and queue LLM-judge jobs for finished generation runs.

nohup ~/lcg-cr/ana/bin/python -m lcs.autopost > ~/lcg-cr/logs/autopost.log 2>&1 &
Writing-type experiments get window+global judging (window_dense for 256K/1M generations); retrieval, reasoning,
code and probe runs are graded by their own scripts and are skipped.  State: ~/lcg-cr/queue/autopost.json.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

H = os.path.expanduser("~/lcg-cr")
STATE = f"{H}/queue/autopost.json"
WINDOW = ("core32k", "mt32k", "mtk2_32k", "mt32k_hp", "mt32k_hp2", "book32k", "bookk2", "bookpen", "capx", "k2_32k", "pen32k", "mtpen32k",
          "ext32k", "depth-native", "depth-yarn")
DENSE = ("long256k", "long1000k")
TAIL = ("rescue", "rescue_mt")       # 4K continuations: judge the final 1.5K tokens
SKIP_JUDGE = ("divx", "lb2", "longproc", "lbwrite", "wbench", "reason2")   # scored by their own pipelines


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"handled": []}
    first = not os.path.exists(STATE)
    while True:
        for f in sorted(glob.glob(f"{H}/queue/done/*.json")):
            name = os.path.basename(f)
            if name in state["handled"]:
                continue
            state["handled"].append(name)
            if first:  # on the very first start only record what is already done
                continue
            q = json.load(open(f))
            if q.get("kind") != "gen":
                continue
            job = q.get("job", "")
            try:   # the job file knows where it wrote (seed / split jobs write into the base model's run dir)
                jd = json.load(open(job))
                run = jd["out_dir"].rstrip("/")
                exp, model = run.split("/")[-2], run.split("/")[-1]
            except Exception:
                exp, model = job.split("/")[-2], os.path.basename(job)[:-5]
                run = f"{H}/runs/{exp}/{model}"
            if not os.path.exists(f"{run}/meta.jsonl"):
                continue
            vpy = "/home/ubuntu/glm53-vllm/.venv/bin/python"
            if exp == "wbench":   # WritingBench critic scoring in the vLLM environment
                spec = {"job": run, "gpus": 1, "model_dir": f"{H}/models/AQuarterMile__WritingBench-Critic-Model-Qwen-7B",
                        "kind": "py", "args": ["-c", f"import os; os.execv({vpy!r}, [{vpy!r}, '-m', 'lcs.wb_critic', {run!r}])"]}
                json.dump(spec, open(f"{H}/queue/pending/016_wbcritic_{model}.json", "w"))
                print(time.strftime("%m-%d %H:%M:%S"), name, "-> queued wb_critic", flush=True)
                continue
            if exp == "lbwrite":  # LongBench-Write official quality rubric, full responses -> 40K judge context
                cmd = [sys.executable, "-m", "lcs.make_post", "judge", "--exps", exp, "--models", model, "--tasks", "lbw",
                       "--max-model-len", "40960", "--prio", "016"]
                out = subprocess.run(cmd, cwd=f"{H}/code", capture_output=True, text=True)
                print(time.strftime("%m-%d %H:%M:%S"), name, "->", out.stdout.strip() or out.stderr.strip()[-300:], flush=True)
                continue
            if exp == "lb2":      # LongBench v2 stage 2: greedy answer extraction
                cmd = [sys.executable, "-m", "lcs.ana_lb2", "stage2", "--models", model, "--prio", "016"]
                out = subprocess.run(cmd, cwd=f"{H}/code", capture_output=True, text=True)
                print(time.strftime("%m-%d %H:%M:%S"), name, "-> lb2 stage2", out.stdout.strip()[-200:] or out.stderr.strip()[-300:], flush=True)
                continue
            if exp in ("divx", "hpdiv"):   # quality-diversity scoring (GPU half) in the vLLM environment
                spec = {"job": run, "gpus": 1, "model_dir": f"{H}/models/Skywork__Skywork-Reward-V2-Llama-3.1-8B",
                        "kind": "py", "args": ["-c", f"import os; os.execv({vpy!r}, [{vpy!r}, '-m', 'lcs.divscore_gpu', {run!r}])"]}
                json.dump(spec, open(f"{H}/queue/pending/016_divscore_{exp}_{model}.json", "w"))
                print(time.strftime("%m-%d %H:%M:%S"), name, "-> queued divscore", flush=True)
                continue
            tasks = ("window_dense" if exp in DENSE else "tail" if exp in TAIL
                     else "window,global" if exp in WINDOW else None)
            if tasks is None:
                continue
            cmd = [sys.executable, "-m", "lcs.make_post", "judge", "--exps", exp, "--models", model, "--tasks", tasks,
                   "--prio", "016"]
            out = subprocess.run(cmd, cwd=f"{H}/code", capture_output=True, text=True)
            print(time.strftime("%m-%d %H:%M:%S"), name, "->", out.stdout.strip() or out.stderr.strip()[-300:], flush=True)
        first = False
        json.dump(state, open(STATE, "w"))
        time.sleep(120)


if __name__ == "__main__":
    main()
