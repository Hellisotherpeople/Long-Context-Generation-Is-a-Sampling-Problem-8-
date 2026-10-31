"""LLM-as-judge for long generations (vLLM offline).

python -m lcs.judge SPEC.json
SPEC = {"judge": path, "judge_name": str, "tp": 1, "runs": [run_dir, ...], "tasks": ["window", "global", "pair"],
        "tok": {run_dir: tokenizer_path}, "pair_baseline": "topp0.9@T1.0", "max_items": null,
        "chat_kwargs": {...}}
Writes <run>/judge_<judge_name>/<task>.jsonl (resumable).
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys

import numpy as np

WIN_STARTS = (0, 8192, 16384, 24576, 30464)
WIN_LEN = 1536
EXC_LEN = 1200

RUBRIC_WINDOW = """You are an expert literary editor. Below is an excerpt from a long machine-generated text. The excerpt starts at an arbitrary point, so it may begin or end mid-sentence; do not penalize that.

Rate the excerpt on each criterion from 1 (worst) to 10 (best):
- fluency: grammaticality and readability of the prose.
- coherence: whether the excerpt makes sense locally (sentences follow logically; no contradictions, gibberish, or derailment).
- non_repetition: 10 = no repetition at all; 1 = dominated by repeated words, phrases, sentences, or loops.
- interest: how engaging, vivid, and original the writing is.
- overall: overall quality of this excerpt as part of a published long-form work.

<excerpt>
{text}
</excerpt>

Respond with only a JSON object of the form {{"fluency": n, "coherence": n, "non_repetition": n, "interest": n, "overall": n}}."""

RUBRIC_GLOBAL = """You are an expert literary editor. A writer was given this task:

<task>
{prompt}
</task>

Below are three excerpts from the resulting long text (about 32,000 tokens): one from the beginning, one from the middle, and one from the end. Excerpts may start or end mid-sentence.

<beginning>
{b}
</beginning>

<middle>
{m}
</middle>

<end>
{e}
</end>

Rate the whole text on each criterion from 1 (worst) to 10 (best):
- consistency: do characters, setting, facts and style stay consistent across the three excerpts?
- progression: does the work keep developing (new events, ideas, scenes) rather than stalling, resetting, or looping?
- sustained_quality: 10 = the middle and end are as good as the beginning; 1 = severe degradation (repetition, gibberish, collapse).
- task_fidelity: does the text stay on the assigned task?
- overall: overall quality as a long-form work.

Respond with only a JSON object of the form {{"consistency": n, "progression": n, "sustained_quality": n, "task_fidelity": n, "overall": n}}."""

RUBRIC_PAIR = """You are an expert literary editor comparing two long texts (A and B) written for the same task:

<task>
{prompt}
</task>

For each text you see excerpts from its beginning, middle, and end (each text is about 32,000 tokens long). Excerpts may start or end mid-sentence.

<text_A>
[beginning] {ab}
[middle] {am}
[end] {ae}
</text_A>

<text_B>
[beginning] {bb}
[middle] {bm}
[end] {be}
</text_B>

Which text is better overall as a long-form work, considering coherence, freedom from repetition and degeneration, sustained quality from beginning to end, and writing quality? Respond with only a JSON object of the form {{"winner": "A" | "B" | "tie"}}."""


def parse_json(text):
    # take the last {...} block (reasoning models may think first)
    ms = list(re.finditer(r"\{[^{}]*\}", text, re.S))
    for m in reversed(ms):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def load_run(run, tokdir, max_items=None):
    from .tok import load_tokenizer
    tok = load_tokenizer(tokdir)
    metas = {}
    for line in open(f"{run}/meta.jsonl"):
        m = json.loads(line)
        metas[m["uid"]] = m
    prompts = {}
    job_files = glob.glob(os.path.expanduser("~/lcg-cr/jobs/") + metas[next(iter(metas))]["job_id"] + ".json")
    if job_files:
        for r in json.load(open(job_files[0]))["requests"]:
            p = r["prompt"]
            prompts[r["uid"]] = p["messages"][-1]["content"] if p.get("type") == "chat" else "Continue the book."
    # merge multi-part generations (multi-turn "_t<k>", segmented "_seg<k>") into one sequence
    groups = {}
    for u in metas:
        m_ = re.match(r"^(.*)_(t|seg)(\d+)$", u)
        base, k = (m_.group(1), int(m_.group(3))) if m_ else (u, 0)
        groups.setdefault(base, []).append((k, u))
    items = []
    bases = sorted(groups)
    if max_items:
        bases = bases[:max_items]
    for b in bases:
        parts = [u for _, u in sorted(groups[b])]
        arrs = [np.load(f"{run}/gen/{u}.npz")["tokens"] for u in parts if os.path.exists(f"{run}/gen/{u}.npz")]
        if not arrs:
            continue
        m0 = dict(metas[parts[0]])
        items.append((b, m0, np.concatenate(arrs), prompts.get(b, prompts.get(parts[0], "Continue the book."))))
    return tok, items


def dec(tok, toks, a, n):
    return tok.decode(toks[a:a + n], skip_special_tokens=True).strip()


def main(spec_path):
    spec = json.load(open(spec_path))
    from vllm import LLM, SamplingParams
    llm = LLM(model=spec["judge"], tensor_parallel_size=int(spec.get("tp", 1)), trust_remote_code=True,
              max_model_len=int(spec.get("max_model_len", 16384)), gpu_memory_utilization=0.88,
              enable_prefix_caching=True, max_num_seqs=256)
    sp = SamplingParams(temperature=float(spec.get("temperature", 0.0)), max_tokens=int(spec.get("max_tokens", 2048)))
    ck = spec.get("chat_kwargs", {})
    jn = spec.get("judge_name", "judge")
    rng = random.Random(0)
    for run in spec["runs"]:
        tok, items = load_run(run, spec["tok"][run], spec.get("max_items"))
        od = f"{run}/judge_{jn}"
        os.makedirs(od, exist_ok=True)
        for task in spec["tasks"]:
            outp = f"{od}/{task}.jsonl"
            done = set()
            if os.path.exists(outp):
                done = {json.loads(l)["key"] for l in open(outp)}
            jobs = []  # (key, record, prompt)
            if task in ("window", "window_dense", "tail"):
                for u, m, toks, pr in items:
                    if task == "tail":   # the last WIN_LEN tokens of each text (short continuations, e.g. loop rescue)
                        starts = (max(0, len(toks) - WIN_LEN),)
                    else:
                        starts = WIN_STARTS if task == "window" else tuple(range(0, max(1, len(toks) - 512), 4096))
                    for s in starts:
                        if s >= len(toks):
                            continue
                        key = f"{u}@{s}"
                        if key in done:
                            continue
                        jobs.append((key, {"uid": u, "start": s, "meta": m.get("meta", {})},
                                     RUBRIC_WINDOW.format(text=dec(tok, toks, s, WIN_LEN))))
            elif task == "lbw":   # LongBench-Write official quality rubric (6 dimensions, 1-5), full response
                tmpl = open(os.path.expanduser("~/lcg-cr/data/bench/LongWriter/evaluation/judge.txt")).read()
                for u, m, toks, pr in items:
                    if u in done:
                        continue
                    t = toks if len(toks) <= 30000 else np.concatenate([toks[:15000], toks[-15000:]])
                    resp = tok.decode(t, skip_special_tokens=True)
                    jobs.append((u, {"uid": u, "meta": m.get("meta", {})},
                                 tmpl.replace("$INST$", pr).replace("$RESPONSE$", resp)))
            elif task == "global":
                for u, m, toks, pr in items:
                    if u in done:
                        continue
                    L = len(toks)
                    jobs.append((u, {"uid": u, "meta": m.get("meta", {})}, RUBRIC_GLOBAL.format(
                        prompt=pr, b=dec(tok, toks, 0, EXC_LEN), m=dec(tok, toks, L // 2, EXC_LEN),
                        e=dec(tok, toks, max(0, L - EXC_LEN), EXC_LEN))))
            elif task == "pair":
                base = spec.get("pair_baseline", "topp0.9@T1.0")
                byp = {}
                for u, m, toks, pr in items:
                    pid, cname = u.split("__")[0], u.split("__")[1]
                    byp.setdefault(pid, {})[cname] = (u, m, toks, pr)
                for pid, d in byp.items():
                    if base not in d:
                        continue
                    bu, bm_, btoks, pr = d[base]
                    for cname, (u, m, toks, _) in d.items():
                        if cname == base:
                            continue
                        for order in (0, 1):
                            key = f"{u}|{bu}|{order}"
                            if key in done:
                                continue
                            X, Y = (toks, btoks) if order == 0 else (btoks, toks)
                            LX, LY = len(X), len(Y)
                            prompt = RUBRIC_PAIR.format(
                                prompt=pr, ab=dec(tok, X, 0, EXC_LEN), am=dec(tok, X, LX // 2, EXC_LEN),
                                ae=dec(tok, X, max(0, LX - EXC_LEN), EXC_LEN), bb=dec(tok, Y, 0, EXC_LEN),
                                bm=dec(tok, Y, LY // 2, EXC_LEN), be=dec(tok, Y, max(0, LY - EXC_LEN), EXC_LEN))
                            jobs.append((key, {"uid": u, "base": bu, "order": order, "meta": m.get("meta", {})}, prompt))
            print(f"[judge] {run} {task}: {len(jobs)} prompts", flush=True)
            B = 2048
            with open(outp, "a") as f:
                for b in range(0, len(jobs), B):
                    chunk = jobs[b:b + B]
                    msgs = [[{"role": "user", "content": p}] for _, _, p in chunk]
                    outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs=ck)
                    for (key, rec, _), o in zip(chunk, outs):
                        txt = o.outputs[0].text
                        rec = dict(rec)
                        rec["key"] = key
                        rec["scores"] = parse_json(txt)
                        if rec["scores"] is None:
                            rec["raw"] = txt[-500:]
                        f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"[judge] {task} {min(b + B, len(jobs))}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
