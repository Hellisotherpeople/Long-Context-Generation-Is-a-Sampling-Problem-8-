"""Model registry: path, tensor parallelism, chat kwargs, context window."""
from __future__ import annotations

ROOT = "/home/ubuntu/lcg-cr/models"

NOTHINK = {"enable_thinking": False}

# name -> dict(repo, tp, chat_kwargs, ctx (native), family, params_b, kind)
M = {}


def add(name, repo, tp=1, chat_kwargs=None, ctx=131072, family="", params=0.0, kind="instruct", **kw):
    M[name] = dict(name=name, repo=repo, path=f"{ROOT}/{repo.replace('/', '__')}", tp=tp,
                   chat_kwargs=chat_kwargs if chat_kwargs is not None else {}, ctx=ctx,
                   family=family, params=params, kind=kind, **kw)


# Qwen3.5 (hybrid Gated-DeltaNet / attention): the main scale ladder
for s, p, tp in [("0.8B", 0.9, 1), ("2B", 2.3, 1), ("4B", 4.7, 1), ("9B", 9.7, 1), ("27B", 27.8, 2)]:
    add(f"qwen3.5-{s.lower()}", f"Qwen/Qwen3.5-{s}", tp, NOTHINK, 262144, "qwen3.5", p)
add("qwen3.5-35b-a3b", "Qwen/Qwen3.5-35B-A3B", 1, NOTHINK, 262144, "qwen3.5", 36.0, active=3.0)
for s, p in [("0.8B", 0.9), ("2B", 2.3), ("4B", 4.7), ("9B", 9.7)]:
    add(f"qwen3.5-{s.lower()}-base", f"Qwen/Qwen3.5-{s}-Base", 1, NOTHINK, 262144, "qwen3.5", p, kind="base")
add("qwen3.5-35b-a3b-base", "Qwen/Qwen3.5-35B-A3B-Base", 1, NOTHINK, 262144, "qwen3.5", 36.0, kind="base")
add("qwen3.6-27b", "Qwen/Qwen3.6-27B", 2, NOTHINK, 262144, "qwen3.6", 27.8)
add("qwen3.8-27b", "Qwen/Qwen3.8-27B", 2, NOTHINK, 262144, "qwen3.8", 27.8)
add("qwen3.5-122b-a10b", "Qwen/Qwen3.5-122B-A10B", 4, NOTHINK, 262144, "qwen3.5", 125.0, active=10.0)

# Gemma 4
add("gemma4-e2b", "google/gemma-4-E2B-it", 1, NOTHINK, 131072, "gemma4", 5.1)
add("gemma4-e4b", "google/gemma-4-E4B-it", 1, NOTHINK, 131072, "gemma4", 8.0)
add("gemma4-12b", "google/gemma-4-12B-it", 1, NOTHINK, 262144, "gemma4", 12.0)
add("gemma4-26b-a4b", "google/gemma-4-26B-A4B-it", 1, NOTHINK, 262144, "gemma4", 25.8, active=4.0)
add("gemma4-31b", "google/gemma-4-31B-it", 2, NOTHINK, 262144, "gemma4", 31.3)

# Ministral 3
add("ministral3-3b", "mistralai/Ministral-3-3B-Instruct-2512-BF16", 1, {}, 262144, "ministral3", 4.3)
add("ministral3-8b", "mistralai/Ministral-3-8B-Instruct-2512-BF16", 1, {}, 262144, "ministral3", 8.9)
add("ministral3-14b", "mistralai/Ministral-3-14B-Instruct-2512-BF16", 1, {}, 262144, "ministral3", 13.9)

# NVIDIA Nemotron (hybrid Mamba)
add("nemotron3-4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", 1, NOTHINK, 262144, "nemotron3", 4.0)
add("nemotron3-30b-a3b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 1, NOTHINK, 1048576, "nemotron3", 31.6, active=3.0)
add("llama3.1-8b-ultralong-1m", "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct", 1, {}, 1048576, "llama3", 8.0)

# Fully open (open data)
add("olmo3-7b", "allenai/Olmo-3-7B-Instruct", 1, {}, 65536, "olmo3", 7.3)
add("olmo3.1-32b", "allenai/Olmo-3.1-32B-Instruct", 2, {}, 65536, "olmo3", 32.2)

# Legacy set (close to the original submission's models)
add("llama3.2-1b", "unsloth/Llama-3.2-1B-Instruct", 1, {}, 131072, "llama3", 1.2)
add("llama3.1-8b", "unsloth/Llama-3.1-8B-Instruct", 1, {}, 131072, "llama3", 8.0)
add("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 1, {}, 32768, "qwen2.5", 1.5)
add("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", 1, {}, 32768, "qwen2.5", 3.1)
add("mistral-7b-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", 1, {}, 32768, "mistral", 7.2)
add("mistral-nemo-12b", "mistralai/Mistral-Nemo-Instruct-2407", 1, {}, 131072, "mistral", 12.2)
add("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", 1, {}, 32768, "qwen2.5", 14.7)
add("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", 2, {}, 32768, "qwen2.5", 32.5)
add("llama3.3-70b", "unsloth/Llama-3.3-70B-Instruct", 4, {}, 131072, "llama3", 70.6)
add("gpt-oss-20b", "openai/gpt-oss-20b", 1, {"reasoning_effort": "low"}, 131072, "gpt-oss", 20.9, active=3.6)
add("gpt-oss-120b", "openai/gpt-oss-120b", 2, {"reasoning_effort": "low"}, 131072, "gpt-oss", 116.8, active=5.1)
add("glm4.5-air", "zai-org/GLM-4.5-Air", 4, NOTHINK, 131072, "glm4.5", 110.5, active=12.0)


def get(name):
    return M[name]

# Base models of other families (book continuation)
add("gemma4-e2b-base", "google/gemma-4-E2B", 1, {}, 131072, "gemma4", 5.1, kind="base")
add("gemma4-e4b-base", "google/gemma-4-E4B", 1, {}, 131072, "gemma4", 8.0, kind="base")
add("gemma4-12b-base", "google/gemma-4-12B", 1, {}, 262144, "gemma4", 12.0, kind="base")
add("olmo3-7b-base", "allenai/Olmo-3-1025-7B", 1, {}, 65536, "olmo3", 7.3, kind="base")
add("ministral3-3b-base", "mistralai/Ministral-3-3B-Base-2512", 1, {}, 262144, "ministral3", 4.3, kind="base")
# Reasoning distills
add("r1-distill-qwen-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1, {}, 131072, "r1distill", 1.8)
add("r1-distill-qwen-7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 1, {}, 131072, "r1distill", 7.6)
add("r1-distill-qwen-14b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 1, {}, 131072, "r1distill", 14.8)
