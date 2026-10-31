"""CPU half of quality-diversity scoring: per-(prompt, sampler, T) diversity and quality from the GPU outputs of
lcs.divscore_gpu (rewards, embeddings, NoveltyBench partitions).

python -m lcs.divscore <run_dir>  -> <run>/divscore_groups.parquet

Per group (the samples of one prompt under one configuration):
  reward_mean  mean reward-model score (Skywork-Reward-V2-Llama-3.1-8B)
  nb_distinct  NoveltyBench v1.0 distinct count (DeBERTa similarity classifier, threshold 0.102, first 128 tokens)
  utility      NoveltyBench patience-weighted (p=0.8) reward of the first member of each new class
  emb_dist     mean pairwise cosine distance of Qwen3-Embedding-0.6B vectors (semantic spread)
  vendi        Vendi score of the cosine-similarity matrix (effective number of distinct outputs)
  distinct2/3  distinct n-gram ratio over the union of samples; self_bleu: mean BLEU-4 vs the other samples
  gzip_ratio   compressed/raw length of the concatenated samples
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd


def _ngrams(w):
    return [Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) for n in range(1, 5)]


def bleu4(hyp, refs, hc=None, rcs=None):
    """Sentence BLEU-4 with smoothing of zero counts (NLTK method1-style) and brevity penalty vs the closest reference.
    hc / rcs: precomputed n-gram counters of the hypothesis / references (same result, faster)."""
    if not hyp:
        return 0.0
    hc = hc or _ngrams(hyp)
    rcs = rcs or [_ngrams(r) for r in refs]
    logp = 0.0
    for n in range(4):
        h = hc[n]
        if not h:
            return 0.0
        match = 0
        for g, c in h.items():
            m = max(rc[n].get(g, 0) for rc in rcs) if rcs else 0
            match += min(c, m)
        tot = sum(h.values())
        p = match / tot if match > 0 else 0.1 / tot
        logp += math.log(p) / 4
    rl = min((len(r) for r in refs), key=lambda L: (abs(L - len(hyp)), L))
    bp = 1.0 if len(hyp) > rl else math.exp(1 - rl / max(1, len(hyp)))
    return bp * math.exp(logp)


def self_bleu(words):
    C = [_ngrams(w) for w in words]
    return float(np.mean([bleu4(words[i], [w for j, w in enumerate(words) if j != i], C[i],
                                [c for j, c in enumerate(C) if j != i]) for i in range(len(words))]))


def main(run):
    from .tok import load_tokenizer
    metas = [json.loads(l) for l in open(f"{run}/meta.jsonl") if l.strip()]
    tok = load_tokenizer(metas[0]["model"])
    rew = {}
    for l in open(f"{run}/divscore_samples.jsonl"):
        r = json.loads(l)
        rew[r["uid"]] = r["reward"]
    V = np.load(f"{run}/divscore_emb.npy")
    uidx = {u: i for i, u in enumerate(json.load(open(f"{run}/divscore_emb_uids.json")))}
    parts = [json.loads(l) for l in open(f"{run}/divscore_partition.jsonl")]
    text, ntok, src = {}, {}, {}
    for m in metas:
        t = np.load(f"{run}/gen/{m['uid']}.npz")["tokens"]
        text[m["uid"]] = tok.decode(t, skip_special_tokens=True)
        ntok[m["uid"]] = len(t)
        src[m["meta"]["pid"]] = m["meta"].get("src")
    rows = []
    for p in parts:
        uids, labels = p["uids"], p["labels"]
        if len(uids) < 2:
            continue
        E = V[[uidx[u] for u in uids]]
        S = E @ E.T
        n = len(uids)
        iu = np.triu_indices(n, 1)
        ev = np.linalg.eigvalsh(S / n)
        ev = ev[ev > 1e-12]
        texts = [text[u] for u in uids]
        words = [t.split() for t in texts]
        u2 = [tuple(w[i:i + 2]) for w in words for i in range(len(w) - 1)]
        u3 = [tuple(w[i:i + 3]) for w in words for i in range(len(w) - 2)]
        raw = "\n".join(texts).encode()
        rs = [rew.get(u, np.nan) for u in uids]
        seen, util, wsum = set(), 0.0, 0.0
        for k, (lab, r) in enumerate(zip(labels, rs)):
            wk = 0.8 ** k
            wsum += wk
            if lab not in seen:
                util += wk * r
                seen.add(lab)
        rows.append({"pid": p["pid"], "src": src.get(p["pid"]), "sampler": p["sampler"], "T": p["T"], "n": n,
                     "reward_mean": float(np.nanmean(rs)), "nb_distinct": len(set(labels)), "utility": util / wsum,
                     "emb_dist": float(np.mean(1 - S[iu])), "vendi": float(np.exp(-(ev * np.log(ev)).sum())),
                     "distinct2": len(set(u2)) / max(1, len(u2)), "distinct3": len(set(u3)) / max(1, len(u3)),
                     "self_bleu": None,
                     "gzip_ratio": len(gzip.compress(raw)) / max(1, len(raw)),
                     "len_mean": float(np.mean([ntok[u] for u in uids])),
                     "truncated": float(np.mean([ntok[u] >= 2048 for u in uids]))})
    from multiprocessing import Pool
    allwords = [[text[u].split() for u in p["uids"]] for p in parts if len(p["uids"]) >= 2]
    with Pool(int(os.environ.get("NPROC", 32))) as pool:
        sb = pool.map(self_bleu, allwords, chunksize=64)
    for r, v in zip(rows, sb):
        r["self_bleu"] = v
    gm = pd.DataFrame(rows)
    gm.to_parquet(f"{run}/divscore_groups.parquet")
    greedy = [rew[m["uid"]] for m in metas if m["meta"]["sampler"] == "greedy" and m["uid"] in rew]
    print("greedy reward", round(float(np.mean(greedy)), 3) if greedy else None)
    pd.set_option("display.width", 220)
    print(gm.groupby(["sampler", "T"])[["reward_mean", "utility", "nb_distinct", "vendi", "emb_dist", "distinct3",
                                        "self_bleu"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
