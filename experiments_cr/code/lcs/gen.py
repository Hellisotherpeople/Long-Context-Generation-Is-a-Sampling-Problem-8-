"""Resumable long-generation driver (vLLM offline engine, step loop).

usage: python -m lcs.gen JOB.json

JOB = {
  "job_id": str, "model": path, "tp": 1,
  "engine": {max_model_len, max_num_seqs, gpu_memory_utilization, hf_overrides, ...},
  "out_dir": path,                      # gen/<uid>.npz + stats/<uid>.p*.npz + meta.jsonl
  "requests": [{"uid", "prompt": {"type": "chat"|"text"|"tokens", ...},
                "max_tokens", "force_len": bool, "lcs": {T, ops, pen}, "meta": {...}}]
}
Prompt types:
  chat:   {"messages": [...], "chat_kwargs": {"enable_thinking": false}, "prefill": str?}
  text:   {"text": str}          (raw completion, e.g. base-model book continuation)
  tokens: {"file": path.npy}     (pre-tokenised ids, e.g. 1M-token contexts)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def stop_ids(tok, model_dir):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    try:
        gc = json.load(open(os.path.join(model_dir, "generation_config.json")))
        e = gc.get("eos_token_id")
        if isinstance(e, int):
            ids.add(e)
        elif isinstance(e, list):
            ids.update(int(x) for x in e)
    except FileNotFoundError:
        pass
    # NB: "<|end|>" is NOT banned: in gpt-oss (harmony) it closes the analysis channel, and models that use it
    # as end-of-sequence list it in generation_config anyway.
    for s in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "</s>",
              "<|return|>", "<|call|>", "<turn|>", "<|end_of_text|>", "<|im_start|>", "<|user|>",
              "<|eom_id|>", "<|start_header_id|>"):
        try:
            i = tok.convert_tokens_to_ids(s)
        except Exception:
            continue
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(ids)


def gen_tokens(llm):
    try:
        tot = None
        for m in llm.get_metrics():
            if "generation_tokens" in m.name and hasattr(m, "value"):
                tot = (tot or 0.0) + float(m.value)
        return tot
    except Exception:
        return None


_ID_CACHE = {}


def _load_ids(path):
    if path not in _ID_CACHE:
        if len(_ID_CACHE) > 4:
            _ID_CACHE.clear()
        _ID_CACHE[path] = np.load(path).astype(np.int64)
    return _ID_CACHE[path]


def build_prompt(tok, p, model_dir):
    t = p["type"]
    if t == "text":
        ids = tok(p["text"], add_special_tokens=p.get("add_bos", True))["input_ids"]
    elif t == "tokens":
        ids = _load_ids(p["file"]).tolist()
        if p.get("suffix_text"):
            ids += tok(p["suffix_text"], add_special_tokens=False)["input_ids"]
    elif t == "chat":
        kw = dict(p.get("chat_kwargs") or {})
        text = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True, **kw)
        if p.get("prefill"):
            text += p["prefill"]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        bos = tok.bos_token_id
        if bos is not None and (not ids or ids[0] != bos) and p.get("force_bos", False):
            ids = [bos] + ids
    else:
        raise ValueError(t)
    return ids


def main(job_path):
    job = json.load(open(job_path))
    out_dir = job["out_dir"]
    os.makedirs(os.path.join(out_dir, "gen"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "stats"), exist_ok=True)
    os.environ["LCS_STATS_DIR"] = os.path.join(out_dir, "stats")
    def seg_done(uid, k):
        return os.path.exists(os.path.join(out_dir, "gen", f"{uid}_seg{k}.npz"))

    def turn_files(uid):
        k = 0
        while os.path.exists(os.path.join(out_dir, "gen", f"{uid}_t{k}.npz")):
            k += 1
        return k

    reqs = []
    for r in job["requests"]:
        if "multiturn" in r:
            k = turn_files(r["uid"])
            got = sum(len(np.load(os.path.join(out_dir, "gen", f"{r['uid']}_t{j}.npz"))["tokens"]) for j in range(k))
            if got < int(r["multiturn"]["total"]) and k < int(r["multiturn"].get("max_turns", 64)):
                r = dict(r)
                r["_turn"], r["_got"] = k, got
                reqs.append(r)
            continue
        if "chain" in r:
            n_seg = int(r["chain"]["n_seg"])
            k = 0
            while k < n_seg and seg_done(r["uid"], k):
                k += 1
            if k < n_seg:
                r = dict(r)
                r["_seg"] = k
                reqs.append(r)
        elif not os.path.exists(os.path.join(out_dir, "gen", f"{r['uid']}.npz")):
            reqs.append(r)
    print(f"[gen] {job['job_id']}: {len(reqs)}/{len(job['requests'])} requests to run", flush=True)
    if not reqs:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    model = job["model"]
    from .tok import load_tokenizer
    tok = load_tokenizer(job.get("tokenizer", model))
    bans = stop_ids(tok, model)
    eng = dict(job.get("engine") or {})
    eng.setdefault("gpu_memory_utilization", 0.85)
    eng.setdefault("enable_prefix_caching", True)
    eng.setdefault("max_num_seqs", 256)
    llm = LLM(model=model, tensor_parallel_size=int(job.get("tp", 1)), seed=int(job.get("seed", 0)),
              logits_processors=["lcs.processor:LCSProcessor"], trust_remote_code=True,
              async_scheduling=False, **eng)
    engine = llm.llm_engine
    meta_f = open(os.path.join(out_dir, "meta.jsonl"), "a")

    # ---- admission control: never admit more sequences than fit in the KV cache at full
    # length, so long generations are never preempted / recomputed.
    cc = engine.vllm_config.cache_config
    kv_tokens = (cc.num_gpu_blocks or 0) * (cc.block_size or 16)
    print(f"[gen] kv cache: {cc.num_gpu_blocks} blocks x {cc.block_size} = {kv_tokens} tokens", flush=True)
    if job.get("max_len"):
        max_len = int(job["max_len"])
        built = [(r, None) for r in reqs]
    else:
        # batch-tokenise distinct prompts (fast Rust path), then reuse for duplicates
        keyf = lambda p: json.dumps(p, sort_keys=True)
        uniq = {}
        for r in reqs:
            uniq.setdefault(keyf(r["prompt"]), r["prompt"])
        texts, keys = [], []
        cache = {}
        for k, p in uniq.items():
            if p["type"] == "text":
                texts.append(p["text"]); keys.append((k, p.get("add_bos", True)))
            else:
                cache[k] = build_prompt(tok, p, model)
        for flag in (True, False):
            sel = [(k, t) for (k, f), t in zip(keys, texts) if f == flag]
            if sel:
                enc = tok([t for _, t in sel], add_special_tokens=flag)["input_ids"]
                for (k, _), ids in zip(sel, enc):
                    cache[k] = ids
        built = [(r, cache[keyf(r["prompt"])]) for r in reqs]
        max_len = max(len(ids) + int(r["max_tokens"]) for r, ids in built)
    cap = int(job.get("max_active") or 0) or max(1, int(0.97 * kv_tokens / max_len)) if kv_tokens else eng["max_num_seqs"]
    cap = min(cap, eng["max_num_seqs"])
    print(f"[gen] admission cap {cap} (max_len {max_len})", flush=True)

    def make_sp(r, rid=None):
        if "multiturn" in r:
            mt = min(int(r["multiturn"]["turn_tokens"]), int(r["multiturn"]["total"]) - r["_got"])
        else:
            mt = int(r["chain"]["seg_tokens"]) if "chain" in r else int(r["max_tokens"])
        lcs = dict(r["lcs"])
        lcs["tag"] = rid or r["uid"]
        lcs["max_tokens"] = mt
        force = r.get("force_len", True)
        if force:
            lcs["ban"] = bans
        return SamplingParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=mt,
                              detokenize=False, skip_special_tokens=False,
                              extra_args={"lcs": lcs}, ignore_eos=bool(force),
                              output_kind=RequestOutputKind.FINAL_ONLY)

    from collections import deque
    queue = deque(built)
    pending = {}

    def admit():
        while queue and len(pending) < cap:
            r, ids = queue.popleft()
            if ids is None:
                ids = build_prompt(tok, r["prompt"], model)
            rid = r["uid"]
            if "multiturn" in r:
                k = r["_turn"]
                msgs = list(r["prompt"]["messages"])
                for j in range(k):
                    prev = np.load(os.path.join(out_dir, "gen", f"{r['uid']}_t{j}.npz"))["tokens"]
                    msgs.append({"role": "assistant", "content": tok.decode(prev, skip_special_tokens=True)})
                    msgs.append({"role": "user", "content": r["multiturn"].get("continue_msg", "Continue.")})
                p2 = dict(r["prompt"]); p2["messages"] = msgs
                ids = build_prompt(tok, p2, model)
                rid = f"{r['uid']}_t{k}"
            elif "chain" in r:
                k = r["_seg"]
                ids = list(ids)
                for j in range(k):
                    ids += np.load(os.path.join(out_dir, "gen", f"{r['uid']}_seg{j}.npz"))["tokens"].tolist()
                rid = f"{r['uid']}_seg{k}"
            sp = make_sp(r, rid)
            room = int(eng["max_model_len"]) - len(ids) - 1
            if sp.max_tokens > room:
                if room < 16:
                    print(f"[gen] skip {rid}: prompt {len(ids)} leaves no room", flush=True)
                    continue
                sp.max_tokens = room
                if "lcs" in (sp.extra_args or {}):
                    sp.extra_args["lcs"]["max_tokens"] = room
            try:
                engine.add_request(rid, {"prompt_token_ids": ids}, sp)
            except Exception as e:  # never let one request kill the job
                print(f"[gen] add_request failed for {rid}: {repr(e)[:200]}", flush=True)
                continue
            pending[rid] = (r, len(ids))

    admit()
    done = 0
    ntok = 0
    t1 = time.time()
    last = t1
    last_gtok = gen_tokens(llm)
    while engine.has_unfinished_requests() or queue:
        for o in engine.step():
            if not o.finished:
                continue
            r, plen = pending.pop(o.request_id)
            c = o.outputs[0]
            toks = np.asarray(c.token_ids, dtype=np.int32)
            np.savez(os.path.join(out_dir, "gen", f"{o.request_id}.npz"), tokens=toks,
                     prompt_len=np.int64(plen))
            meta = {"uid": o.request_id, "n_tokens": int(len(toks)), "prompt_len": plen,
                    "finish": c.finish_reason, "lcs": r["lcs"], "meta": r.get("meta", {}),
                    "model": model, "job_id": job["job_id"], "t": time.time() - t1}
            meta_f.write(json.dumps(meta) + "\n"); meta_f.flush()
            done += 1
            ntok += len(toks)
            if "multiturn" in r:
                got = r["_got"] + len(toks)
                if got < int(r["multiturn"]["total"]) and r["_turn"] + 1 < int(r["multiturn"].get("max_turns", 64)):
                    r2 = dict(r); r2["_turn"], r2["_got"] = r["_turn"] + 1, got
                    queue.appendleft((r2, None))
            if "chain" in r and r["_seg"] + 1 < int(r["chain"]["n_seg"]):
                r2 = dict(r)
                r2["_seg"] = r["_seg"] + 1
                queue.appendleft((r2, None))
        admit()
        now = time.time()
        if now - last > 120:
            gtok = gen_tokens(llm)
            rate = (gtok - last_gtok) / (now - last) if gtok is not None and last_gtok is not None else float("nan")
            print(f"[gen] {done} done, {len(pending)} active, {len(queue)} queued, "
                  f"gen {rate:.0f} tok/s, elapsed {now-t1:.0f}s", flush=True)
            last, last_gtok = now, gtok
    print(f"[gen] finished {done} in {time.time()-t1:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
