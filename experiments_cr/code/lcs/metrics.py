"""Per-generation, per-window metrics for long generations.

For each run dir (runs/<exp>/<model>) writes metrics.jsonl with one record per uid:
  meta fields + per-window arrays (window = W model tokens) + whole-sequence scalars.

Repetition / degeneration
  rep{n}_tok : fraction of positions whose n-gram (of model tokens, ending at the
               position) already occurred earlier in the generation  (n = 4, 10, 32)
  loop_onset : first window from which >= 80% of the remaining windows have rep32 >= 0.5
               (a "terminal loop"); None if never.
  zlib       : compressed/raw byte ratio of the window text (lower = more redundant)
Lexical diversity (word level, tokenizer independent)
  d2, d3     : distinct bigram / trigram ratio within the window
  mattr      : moving-average type-token ratio (50-word windows)
Sampler diagnostics (from the logits processor)
  H1, HT, pmax, kept(median), mass, Hq, surpr (=-lp1), rank1_mean, logrank_mean, argmax_frac
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import glob
import json
import re
import sys
import zlib
from multiprocessing import Pool

import numpy as np

W = 1024
WORD = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]")


def ngram_rep(toks: np.ndarray, n: int) -> np.ndarray:
    """rep[i] = 1 if the n-gram ending at i occurred earlier (ending at j < i)."""
    L = len(toks)
    out = np.zeros(L, dtype=bool)
    if L < n:
        return out
    t = toks.astype(np.uint64)
    h = np.zeros(L - n + 1, dtype=np.uint64)
    P = np.uint64(1099511628211)
    for k in range(n):
        h = h * P + (t[k:L - n + 1 + k] + np.uint64(1))
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    out[n - 1:] = first[inv] < np.arange(len(h))
    return out


def copy_distance(toks: np.ndarray, n: int = 32) -> np.ndarray:
    """For each position whose n-gram occurred before, distance to the PREVIOUS occurrence (else 0)."""
    L = len(toks)
    out = np.zeros(L, dtype=np.int64)
    if L < n:
        return out
    t = toks.astype(np.uint64)
    h = np.zeros(L - n + 1, dtype=np.uint64)
    P = np.uint64(1099511628211)
    for k in range(n):
        h = h * P + (t[k:L - n + 1 + k] + np.uint64(1))
    order = np.lexsort((np.arange(len(h)), h))
    hs = h[order]
    same = np.concatenate([[False], hs[1:] == hs[:-1]])
    prev = np.where(same, np.concatenate([[0], order[:-1]]), -1)
    dist = np.zeros(len(h), dtype=np.int64)
    dist[order] = np.where(prev >= 0, order - prev, 0)
    out[n - 1:] = dist
    return out


def distinct(words, n):
    if len(words) < n:
        return np.nan
    grams = list(zip(*[words[i:] for i in range(n)]))
    return len(set(grams)) / max(1, len(grams))


def mattr(words, w=50):
    if len(words) < w:
        return len(set(words)) / max(1, len(words))
    vals = []
    for i in range(0, len(words) - w + 1, 10):
        seg = words[i:i + w]
        vals.append(len(set(seg)) / w)
    return float(np.mean(vals))


def load_stats(run, uid):
    S, T = {}, {}
    for p in sorted(glob.glob(f"{run}/stats/{uid}.p*.npz")):
        try:
            d = np.load(p)
            pos, st = d["pos"], d["stats"]
            tpos, lp1, rk, lq = d["tpos"], d["lp1"], d["rank1"], d["lq"]
        except Exception:
            continue  # unreadable part (e.g. concurrent write); treated as missing
        for i, ps in enumerate(pos.tolist()):
            S[ps] = st[i]
        for i, tp in enumerate(tpos.tolist()):
            T[tp] = (lp1[i], rk[i], lq[i])
    return S, T


_tok = None


def _init(tokdir):
    global _tok
    from .tok import load_tokenizer
    _tok = load_tokenizer(tokdir)


def gen_metrics(args):
    run, uid, meta = args[:3]
    parts = args[3] if len(args) > 3 else [uid]
    toks_l, S, T, off, part_lens = [], {}, {}, 0, []
    for pu in parts:
        t = np.load(f"{run}/gen/{pu}.npz")["tokens"].astype(np.int64)
        s_, t_ = load_stats(run, pu)
        for p, v in s_.items():
            S[p + off] = v
        for p, v in t_.items():
            T[p + off] = v
        toks_l.append(t)
        part_lens.append(int(len(t)))
        off += len(t)
    toks = np.concatenate(toks_l) if toks_l else np.zeros(0, np.int64)
    L = len(toks)
    st = np.full((L, 7), np.nan, np.float32)
    for p, v in S.items():
        if p < L:
            st[p, :len(v)] = v
    lp = np.full(L, np.nan, np.float32)
    rk = np.full(L, np.nan, np.float64)
    lqbad = 0
    for p, (a, b, c) in T.items():
        if p < L:
            lp[p] = a
            rk[p] = b
            lqbad += int(not np.isfinite(c))
    reps = {n: ngram_rep(toks, n) for n in (4, 10, 32)}
    cdist = copy_distance(toks, 32)
    nw = (L + W - 1) // W
    rec = dict(meta)
    rec.update({"uid": uid, "n_tokens": int(L), "n_windows": int(nw), "lq_outside": int(lqbad), "part_lens": part_lens,
                "stats_cover": float(np.isfinite(st[:, 0]).mean()) if L else 0.0})
    win = {k: [] for k in ("rep4", "rep10", "rep32", "zlib", "d2", "d3", "mattr", "H1", "HT", "pmax", "kept",
                           "mass", "Hq", "pend", "pend_n", "surpr", "rank1_mean", "logrank_mean", "argmax_frac",
                           "single_frac", "copy_dist")}
    all_words = []
    for w in range(nw):
        a, b = w * W, min(L, (w + 1) * W)
        seg = toks[a:b]
        for n in (4, 10, 32):
            win[f"rep{n}"].append(float(reps[n][a:b].mean()))
        cd = cdist[a:b]
        win["copy_dist"].append(float(np.median(cd[cd > 0])) if (cd > 0).any() else None)
        text = _tok.decode(seg, skip_special_tokens=False)
        raw = text.encode("utf-8", "ignore")
        win["zlib"].append(len(zlib.compress(raw, 6)) / max(1, len(raw)))
        words = WORD.findall(text.lower())
        all_words.extend(words)
        win["d2"].append(distinct(words, 2))
        win["d3"].append(distinct(words, 3))
        win["mattr"].append(mattr(words))
        s = st[a:b]
        for j, k in enumerate(("H1", "HT", "pmax")):
            win[k].append(float(np.nanmean(s[:, j])) if np.isfinite(s[:, j]).any() else np.nan)
        win["kept"].append(float(np.nanmedian(s[:, 3])) if np.isfinite(s[:, 3]).any() else np.nan)
        win["single_frac"].append(float(np.nanmean(s[:, 3] <= 1)) if np.isfinite(s[:, 3]).any() else np.nan)
        win["mass"].append(float(np.nanmean(s[:, 4])) if np.isfinite(s[:, 4]).any() else np.nan)
        win["Hq"].append(float(np.nanmean(s[:, 5])) if np.isfinite(s[:, 5]).any() else np.nan)
        win["pend"].append(float(np.nanmax(s[:, 6])) if np.isfinite(s[:, 6]).any() else np.nan)
        win["pend_n"].append(int(np.nansum(s[:, 6] > 0.2)) if np.isfinite(s[:, 6]).any() else np.nan)
        l, r = lp[a:b], rk[a:b]
        ok = np.isfinite(r)
        win["surpr"].append(float(-np.nanmean(l)) if ok.any() else np.nan)
        win["rank1_mean"].append(float(r[ok].mean()) if ok.any() else np.nan)
        win["logrank_mean"].append(float(np.log(r[ok]).mean()) if ok.any() else np.nan)
        win["argmax_frac"].append(float((r[ok] == 1).mean()) if ok.any() else np.nan)
    rec["win"] = win
    deg = np.array(win["rep32"]) >= 0.5
    onset = None
    for w in range(nw):
        rest = deg[w:]
        if rest[0] and rest.mean() >= 0.8:
            onset = w
            break
    rec["loop_onset_win"] = onset
    if onset is not None:
        cd = cdist[onset * W:]
        cd = cd[cd > 0]
        rec["loop_copy_dist"] = float(np.median(cd)) if len(cd) else None
    else:
        rec["loop_copy_dist"] = None
    rec["loop_onset_tok"] = None if onset is None else int(onset * W)
    rec["d2_all"] = distinct(all_words, 2)
    rec["n_words"] = len(all_words)
    q = max(1, nw // 4)
    for qi in range(4):
        sl = slice(qi * q, (qi + 1) * q if qi < 3 else nw)
        for k in ("rep10", "rep32", "d2", "zlib", "H1", "Hq", "argmax_frac", "logrank_mean", "surpr", "mattr", "pend", "single_frac"):
            vals = np.array(win[k][sl], dtype=float)
            rec[f"Q{qi+1}_{k}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
    return rec


def run_metrics(run, tokdir, nproc=48, force=False):
    outp = f"{run}/metrics.jsonl"
    done = set()
    if os.path.exists(outp) and not force:
        for line in open(outp):
            done.add(json.loads(line)["uid"])
    metas = {}
    if os.path.exists(f"{run}/meta.jsonl"):
        for line in open(f"{run}/meta.jsonl"):
            m = json.loads(line)
            mm = dict(m.get("meta", {}))
            mm.update({"model_path": m["model"], "lcs": m["lcs"], "job_id": m["job_id"]})
            metas[m["uid"]] = mm
    import re as _re
    groups = {}
    for f in glob.glob(f"{run}/gen/*.npz"):
        u = os.path.basename(f)[:-4]
        m_ = _re.match(r"^(.*)_(t|seg)(\d+)$", u)
        if m_:
            groups.setdefault(m_.group(1), []).append((int(m_.group(3)), u, m_.group(2)))
        else:
            groups.setdefault(u, []).append((0, u, ""))
    job_reqs = {}
    try:
        jid = next(iter(metas.values()))["job_id"] if metas else None
        jf = os.path.expanduser(f"~/lcg-cr/jobs/{jid}.json") if jid else None
        if jf and os.path.exists(jf):
            job_reqs = {r["uid"]: r for r in json.load(open(jf))["requests"]}
    except Exception:
        job_reqs = {}
    todo = []
    for base, lst in groups.items():
        if base in done:
            continue
        lst.sort()
        parts = [u for _, u, _ in lst]
        kind = lst[0][2]
        r = job_reqs.get(base, {})
        if kind == "seg" and "chain" in r and len(parts) < int(r["chain"]["n_seg"]):
            continue  # incomplete chain
        if kind == "t" and "multiturn" in r:
            tot = sum(int(metas.get(p, {}).get("n_tokens_", 0)) for p in parts)
            got = sum(len(np.load(f"{run}/gen/{p}.npz")["tokens"]) for p in parts)
            if got < int(r["multiturn"]["total"]) and len(parts) < int(r["multiturn"].get("max_turns", 64)):
                continue  # incomplete multi-turn
        meta = dict(metas.get(parts[0], {}))
        todo.append((run, base, meta, parts))
    if not todo:
        return 0
    with Pool(nproc, initializer=_init, initargs=(tokdir,)) as pool, open(outp, "w" if force else "a") as f:
        for rec in pool.imap_unordered(gen_metrics, todo, chunksize=4):
            f.write(json.dumps(rec, default=lambda o: None if o is None else float(o)) + "\n")
    return len(todo)


if __name__ == "__main__":
    run, tokdir = sys.argv[1], sys.argv[2]
    n = run_metrics(run, tokdir, force="--force" in sys.argv)
    print(f"{run}: {n} new records")
