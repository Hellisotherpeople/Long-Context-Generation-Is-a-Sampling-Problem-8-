"""Frontier-model judge (Anthropic or OpenAI API) on a stratified subset of long generations, with the same rubrics and
output format as lcs.judge, so lcs.ana_judges can measure agreement with the local judges.

ANTHROPIC_API_KEY=... python -m lcs.judge_api --provider anthropic --model claude-sonnet-5-5 --name claude \
    --runs ~/lcg-cr/runs/mt32k/qwen3.5-9b,... --tasks window --per-cell 6 --starts last [--dry-run]
The key is read from the environment only (never written to disk).  --dry-run prints the number of calls and an
input-token estimate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .judge import EXC_LEN, RUBRIC_GLOBAL, RUBRIC_WINDOW, WIN_LEN, WIN_STARTS, dec, load_run, parse_json


def call(provider, model, prompt, max_tokens=300, retries=6):
    if provider == "anthropic":
        url, body = "https://api.anthropic.com/v1/messages", {
            "model": model, "max_tokens": max_tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]}
        hdr = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    else:
        url, body = "https://api.openai.com/v1/chat/completions", {
            "model": model, "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
        hdr = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"}
    for k in range(retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr)
            r = json.load(urllib.request.urlopen(req, timeout=120))
            if provider == "anthropic":
                return "".join(c.get("text", "") for c in r["content"]), r.get("usage", {})
            return r["choices"][0]["message"]["content"], r.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and k < retries - 1:
                time.sleep(2 ** k * 5)
                continue
            raise
        except Exception:
            if k < retries - 1:
                time.sleep(2 ** k * 5)
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--name", default="claude")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tasks", default="window")
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--starts", default="last", choices=["last", "all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rng = random.Random(0)
    total_calls, total_chars = 0, 0
    for run in a.runs.split(","):
        run = os.path.expanduser(run)
        meta0 = json.loads(open(f"{run}/meta.jsonl").readline())
        tok, items = load_run(run, meta0["model"])
        cells = {}
        for it in items:
            m = it[1].get("meta", {})
            cells.setdefault((m.get("sampler"), m.get("T")), []).append(it)
        chosen = []
        for k in sorted(cells, key=str):
            v = cells[k]
            rng.shuffle(v)
            chosen += v[:a.per_cell]
        od = f"{run}/judge_{a.name}"
        os.makedirs(od, exist_ok=True)
        for task in a.tasks.split(","):
            outp = f"{od}/{task}.jsonl"
            done = {json.loads(l)["key"] for l in open(outp)} if os.path.exists(outp) else set()
            jobs = []
            for u, m, toks, pr in chosen:
                if task == "window":
                    starts = [s for s in WIN_STARTS if s < len(toks)]
                    if a.starts == "last":
                        starts = starts[-1:]
                    for s in starts:
                        key = f"{u}@{s}"
                        if key not in done:
                            jobs.append((key, {"uid": u, "start": s, "meta": m.get("meta", {})},
                                         RUBRIC_WINDOW.format(text=dec(tok, toks, s, WIN_LEN))))
                elif task == "global" and u not in done:
                    L = len(toks)
                    jobs.append((u, {"uid": u, "meta": m.get("meta", {})}, RUBRIC_GLOBAL.format(
                        prompt=pr, b=dec(tok, toks, 0, EXC_LEN), m=dec(tok, toks, L // 2, EXC_LEN),
                        e=dec(tok, toks, max(0, L - EXC_LEN), EXC_LEN))))
            total_calls += len(jobs)
            total_chars += sum(len(p) for _, _, p in jobs)
            print(f"{run} {task}: {len(jobs)} calls", flush=True)
            if a.dry_run or not jobs:
                continue

            def work(j):
                key, rec, prompt = j
                txt, usage = call(a.provider, a.model, prompt)
                rec = dict(rec, key=key, scores=parse_json(txt), usage=usage)
                if rec["scores"] is None:
                    rec["raw"] = txt[-500:]
                return rec

            with open(outp, "a") as f, ThreadPoolExecutor(a.workers) as ex:
                for rec in ex.map(work, jobs):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
    print(f"total calls {total_calls}, ~{total_chars / 4 / 1e6:.2f}M input tokens", flush=True)


if __name__ == "__main__":
    main()
