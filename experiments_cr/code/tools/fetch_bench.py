"""Download benchmark datasets and helper models into ~/lcg-cr/data/bench and ~/lcg-cr/models."""
import os, subprocess, sys
H = os.path.expanduser("~/lcg-cr")
D = f"{H}/data/bench"
os.makedirs(D, exist_ok=True)
from huggingface_hub import snapshot_download
ds = ["yimingzhang/novelty-bench", "liweijiang/infinite-chats-eval", "zai-org/LongBench-v2", "PrincetonPLI/LongProc",
      "mozhu/LongGenBench", "MathArena/aime_2026", "MathArena/aime_2025", "MathArena/hmmt_feb_2025", "MathArena/hmmt_feb_2026",
      "LIFEBench/LIFEBench", "THU-KEG/Arena-Write"]
for d in ds:
    try:
        p = snapshot_download(d, repo_type="dataset", local_dir=f"{D}/{d.replace('/', '__')}")
        print("OK", d, p, flush=True)
    except Exception as e:
        print("FAIL", d, repr(e)[:200], flush=True)
# LiveCodeBench: jsonl files only (loading script unsupported by datasets>=4)
try:
    snapshot_download("livecodebench/code_generation_lite", repo_type="dataset",
                      local_dir=f"{D}/livecodebench__code_generation_lite", allow_patterns=["*.jsonl"])
    print("OK livecodebench", flush=True)
except Exception as e:
    print("FAIL livecodebench", repr(e)[:200], flush=True)
# BABILong: only the long configs we use
try:
    snapshot_download("RMT-team/babilong", repo_type="dataset", local_dir=f"{D}/RMT-team__babilong",
                      allow_patterns=["*128k*", "*256k*", "*512k*", "*1M*", "*32k*", "*README*"])
    print("OK babilong", flush=True)
except Exception as e:
    print("FAIL babilong", repr(e)[:200], flush=True)
for repo, dst in [("https://github.com/THUDM/LongWriter", "LongWriter"), ("https://github.com/X-PLUG/WritingBench", "WritingBench"),
                  ("https://github.com/NVIDIA/RULER", "RULER"), ("https://github.com/novelty-bench/novelty-bench", "novelty-bench"),
                  ("https://github.com/LiveCodeBench/LiveCodeBench", "LiveCodeBench")]:
    if not os.path.exists(f"{D}/{dst}"):
        r = subprocess.run(["git", "clone", "--depth", "1", repo, f"{D}/{dst}"], capture_output=True, text=True)
        print("git", dst, r.returncode, r.stderr[-200:], flush=True)
for m in ["Qwen/Qwen3-Embedding-0.6B", "yimingzhang/deberta-v3-large-generation-similarity",
          "AQuarterMile/WritingBench-Critic-Model-Qwen-7B", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"]:
    try:
        p = snapshot_download(m, local_dir=f"{H}/models/{m.replace('/', '__')}")
        print("OK model", m, flush=True)
    except Exception as e:
        print("FAIL model", m, repr(e)[:200], flush=True)
