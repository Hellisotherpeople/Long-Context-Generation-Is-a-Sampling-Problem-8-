"""GPU half of quality-diversity scoring (runs in the vLLM environment; stdlib + numpy + torch + transformers + vllm only).

python -m lcs.divscore_gpu <run_dir>
writes <run>/divscore_samples.jsonl  (uid, reward)              Skywork-Reward-V2-Llama-3.1-8B via vLLM pooling
       <run>/divscore_emb.npy + divscore_emb_uids.json        Qwen3-Embedding-0.6B (last-token pooling, normalized)
       <run>/divscore_partition.jsonl (group, uids, labels)   NoveltyBench v1.0 classifier partitions
The CPU half (lcs.divscore) turns these into per-prompt diversity metrics.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from collections import defaultdict

import numpy as np

H = os.path.expanduser("~/lcg-cr")
RM = f"{H}/models/Skywork__Skywork-Reward-V2-Llama-3.1-8B"
EMBED = f"{H}/models/Qwen__Qwen3-Embedding-0.6B"
SIM = f"{H}/models/yimingzhang__deberta-v3-large-generation-similarity"
TOKD = f"{H}/models/microsoft__deberta-v3-large"


def load(run):
    from transformers import AutoTokenizer
    metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
    tok = AutoTokenizer.from_pretrained(metas[0]["model"], trust_remote_code=True)
    job = json.load(open(f"{H}/jobs/{metas[0]['job_id']}.json"))
    prompt_of = {r["uid"]: r["prompt"]["messages"][-1]["content"] for r in job["requests"]}
    rows = []
    for m in metas:
        toks = np.load(f"{run}/gen/{m['uid']}.npz")["tokens"]
        rows.append({"uid": m["uid"], "meta": m["meta"], "prompt": prompt_of.get(m["uid"], ""),
                     "text": tok.decode(toks, skip_special_tokens=True)})
    return rows


def free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def rewards(rows):
    from transformers import AutoTokenizer
    from vllm import LLM
    tk = AutoTokenizer.from_pretrained(RM)
    texts = [tk.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["text"]}],
                                    tokenize=False) for r in rows]
    from vllm.pooling_params import PoolingParams
    llm = LLM(model=RM, runner="pooling", max_model_len=8192, gpu_memory_utilization=0.85, enable_prefix_caching=False)
    outs = llm.classify(texts, pooling_params=PoolingParams(use_activation=False),
                        tokenization_kwargs={"truncation": True, "max_length": 8000})
    vals = [float(o.outputs.probs[0]) for o in outs]
    del llm
    free()
    return vals


def embeddings(rows):
    from vllm import LLM
    llm = LLM(model=EMBED, runner="pooling", max_model_len=4096, gpu_memory_utilization=0.5)
    outs = llm.embed([r["text"] or " " for r in rows], tokenization_kwargs={"truncation": True, "max_length": 4000})
    V = np.asarray([o.outputs.embedding for o in outs], np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    del llm
    free()
    return V


def partitions(rows, bs=1024):
    """NoveltyBench partition: within each (prompt, sampler, T) group, in seed order, a sample joins the first earlier
    class head that is identical or judged equivalent (P > 0.102 on the first 128 tokens of each); else it opens a new
    class.  Batched by rounds (round j = j-th sample of every group vs. all of that group's heads) -> same result as the
    sequential first-match rule."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(TOKD)
    m = AutoModelForSequenceClassification.from_pretrained(SIM, torch_dtype=torch.float32).to("cuda:0").eval()

    def probs(pairs):
        out = np.zeros(len(pairs), np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            seqs, tts = [], []
            for i in idx:
                a, c = pairs[i]
                seqs.append([tk.cls_token_id] + a + [tk.sep_token_id] + c + [tk.sep_token_id])
                tts.append([0] * (len(a) + 2) + [1] * (len(c) + 1))
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tk.pad_token_id, dtype=torch.long)
            tt = torch.zeros((len(seqs), L), dtype=torch.long)
            am = torch.zeros((len(seqs), L), dtype=torch.long)
            for k, (q, t) in enumerate(zip(seqs, tts)):
                ids[k, :len(q)] = torch.tensor(q); tt[k, :len(t)] = torch.tensor(t); am[k, :len(q)] = 1
            with torch.inference_mode():
                pr = m(input_ids=ids.cuda(), token_type_ids=tt.cuda(), attention_mask=am.cuda()).logits.softmax(-1)[:, 1]
            out[idx] = pr.float().cpu().numpy()
        return out

    groups = defaultdict(list)
    for r in rows:
        mm = r["meta"]
        if mm["sampler"] == "greedy":
            continue
        groups[(mm["pid"], mm["sampler"], mm["T"])].append(r)
    keys = list(groups)
    G = [sorted(groups[k], key=lambda r: r["meta"]["seed"]) for k in keys]
    enc = [[tk.encode(r["text"], truncation=True, max_length=128, add_special_tokens=False) for r in g] for g in G]
    heads = [[] for _ in G]
    labels = [[] for _ in G]
    for j in range(max(map(len, G))):
        pairs, owner = [], []
        for gi, g in enumerate(G):
            if j < len(g):
                for h in heads[gi]:
                    pairs.append((enc[gi][h], enc[gi][j]))
                    owner.append(gi)
        P = probs(pairs) if pairs else np.zeros(0)
        pos = 0
        for gi, g in enumerate(G):
            if j >= len(g):
                continue
            nh = len(heads[gi])
            pg = P[pos:pos + nh]
            pos += nh
            k = next((c for c, h in enumerate(heads[gi]) if g[h]["text"] == g[j]["text"] or pg[c] > 0.102), -1)
            if k < 0:
                heads[gi].append(j)
                k = nh
            labels[gi].append(k)
        print("partition round", j, "pairs", len(pairs), flush=True)
    out = [{"pid": key[0], "sampler": key[1], "T": key[2], "uids": [r["uid"] for r in g], "labels": lab}
           for key, g, lab in zip(keys, G, labels)]
    del m
    free()
    return out


def main(run):
    rows = load(run)
    print("samples", len(rows), flush=True)
    sf = f"{run}/divscore_samples.jsonl"
    if not os.path.exists(sf):
        vals = rewards(rows)
        with open(sf, "w") as f:
            for r, v in zip(rows, vals):
                f.write(json.dumps({"uid": r["uid"], "reward": v}) + "\n")
        print("rewards done", flush=True)
    if not os.path.exists(f"{run}/divscore_emb.npy"):
        V = embeddings(rows)
        np.save(f"{run}/divscore_emb.npy", V)
        json.dump([r["uid"] for r in rows], open(f"{run}/divscore_emb_uids.json", "w"))
        print("embeddings done", V.shape, flush=True)
    pf = f"{run}/divscore_partition.jsonl"
    if not os.path.exists(pf):
        parts = partitions(rows)
        with open(pf, "w") as f:
            for p in parts:
                f.write(json.dumps(p) + "\n")
        print("partitions done", len(parts), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
