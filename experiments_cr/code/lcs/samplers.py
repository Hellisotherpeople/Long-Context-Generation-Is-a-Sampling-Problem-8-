"""Batched truncation samplers + per-step diagnostics (pure torch).

Every row has a temperature T and an ordered list of truncation ops
[(method, param), ...].  Temperature is applied first (standard HF/vLLM order),
then each op computes a keep-mask on the current (renormalised) post-temperature
distribution.  The raw argmax is always kept (min_keep = 1).

Methods
  temp        no truncation (pure ancestral sampling at temperature T)
  greedy      argmax only
  top_k       k highest logits                               (Fan et al. 2018)
  top_p       smallest prefix with cumulative mass >= p      (Holtzman et al. 2020)
  min_p       p(v) >= p * p_max                              (Nguyen et al. 2024)
  typical     locally typical set with mass >= tau           (Meister et al. 2023)
  eta         p(v) >= min(eta, sqrt(eta) * exp(-H))          (Hewitt et al. 2022)
  epsilon     p(v) >= eps                                    (Hewitt et al. 2022)
  top_nsigma  z(v) >= max(z) - n * std(z)                    (Tang et al. 2024)
  p_less      p(v) >= sum_u p(u)^2                           (Tan et al. 2025)
  p_less_norm p(v) >= (sum p^2 - 1/|V|) / (1 - 1/|V|)        (Tan et al. 2025)
  top_h       longest prefix whose renormalised entropy <= alpha * H(p)
                                                             (Baghaei et al. 2025)
  temp_last   pseudo-op ["temp_last", t]: compute every truncation mask at the row's T (use T=1 for the usual
              "temperature last" order of llama.cpp/koboldcpp), then sample from softmax(z / t) on the kept set

Diagnostics returned per row (STAT_NAMES):
  H1    entropy (nats) of the model distribution at T=1 (after bans/penalties)
  HT    entropy of the post-temperature distribution, before truncation
  pmax  max probability at T=1
  kept  size of the final candidate set
  mass  post-temperature probability mass retained by truncation
  Hq    entropy of the final (renormalised) sampling distribution
  pend  raw probability the model assigns to the (banned) end/stop tokens ("end pressure")
"""
from __future__ import annotations

import math
import os

import torch

STAT_NAMES = ("H1", "HT", "pmax", "kept", "mass", "Hq", "pend")
METHODS = ("temp", "greedy", "top_k", "top_p", "min_p", "typical", "eta", "epsilon",
           "top_nsigma", "p_less", "p_less_norm", "top_h", "keep_top", "min_k",
           "top_nsigma_softcap", "top_nsigma_uncap", "temp_last")
CAP = 30.0  # final-logit soft-cap of Gemma-2/3/4 (logits -> CAP * tanh(logits / CAP))
NEG_INF = float("-inf")


KSORT = 2048  # partial-sort width for prefix-based rules (exact fallback to a full sort)


def _prefix_keep(p: torch.Tensor, keep_fn, need_full_fn) -> torch.Tensor:
    """Evaluate a descending-probability prefix rule using topk(KSORT) with an exact
    full-sort fallback for rows whose prefix might extend beyond KSORT tokens."""
    n, V = p.shape
    K = min(KSORT, V)
    vals, order = torch.topk(p, K, dim=-1, sorted=True)
    ks = keep_fn(vals, slice(None))
    out = torch.zeros_like(p, dtype=torch.bool).scatter_(1, order, ks)
    if K < V:
        full = need_full_fn(vals, ks)
        if bool(full.any()):
            rows = full.nonzero().squeeze(1)
            ps, o2 = torch.sort(p.index_select(0, rows), dim=-1, descending=True)
            ks2 = keep_fn(ps, rows)
            out[rows] = torch.zeros_like(ps, dtype=torch.bool).scatter_(1, o2, ks2)
    return out


def op_mask(method: str, a: torch.Tensor, z: torch.Tensor, p: torch.Tensor, H: torch.Tensor,
            finite: bool) -> torch.Tensor:
    """Keep-mask for one op on the current distribution.

    z [n,V] current post-temperature logits (may contain -inf), p = softmax(z),
    H [n] entropy of p, a [n] per-row hyper-parameter, finite = z has no -inf.
    """
    if method == "greedy":
        return z == z.max(-1, keepdim=True).values
    if method == "keep_top":
        kmax = max(1, min(int(getattr(a, "_kmax", 0)) or int(a.max().item()), z.shape[-1]))
        vals = torch.topk(z, kmax, dim=-1).values
        kth = vals.gather(1, (a.long().clamp(1, kmax) - 1)[:, None])
        return z >= kth
    if method == "min_k":
        # Ding et al. (2026): keep the top-k where k maximises the position-weighted relative logit
        # decay w_i = (l_i - l_{i+1}) / R / i, with fallback floor(tau / R).  Computed on the given
        # logits; callers pass RAW (pre-temperature) logits so the rule is temperature invariant.
        zr = getattr(a, "_raw", None)
        zz = z if zr is None else zr
        fin = torch.isfinite(zz)
        K = min(4096, zz.shape[-1])
        vals, order = torch.topk(zz, K, dim=-1)
        low = torch.where(fin, zz, torch.full_like(zz, float("inf"))).min(-1, keepdim=True).values
        R = (vals[:, :1] - low).clamp_min(1e-8)
        nxt = vals[:, 1:]
        gaps = torch.where(torch.isfinite(nxt), vals[:, :-1] - nxt, torch.zeros_like(nxt))
        idx = torch.arange(1, K, device=zz.device, dtype=zz.dtype)
        w = gaps / R / idx
        cliff = w.argmax(-1, keepdim=True) + 1
        fallback = torch.floor(a[:, None] / R).clamp(0, K).long()
        k = torch.maximum(cliff, fallback).clamp(1, K)
        ks = torch.arange(K, device=zz.device)[None, :] < k
        return torch.zeros_like(zz, dtype=torch.bool).scatter_(1, order, ks)
    if method == "top_k":
        kmax = max(1, min(int(getattr(a, "_kmax", 0)) or int(a.max().item()), z.shape[-1]))
        vals = torch.topk(z, kmax, dim=-1).values
        kth = vals.gather(1, (a.long().clamp(1, kmax) - 1)[:, None])
        return z >= kth
    if method == "min_p":
        return p >= a[:, None] * p.max(-1, keepdim=True).values
    if method == "epsilon":
        return p >= a[:, None]
    if method == "eta":
        thr = torch.minimum(a, a.sqrt() * torch.exp(-H))
        return p >= thr[:, None]
    if method in ("top_nsigma_softcap", "top_nsigma_uncap"):
        # Mechanism probe: the top-nsigma set computed on soft-capped (or un-capped) RAW logits, so the set stays
        # temperature invariant exactly like the model's own soft-cap; sampling still uses the model's logits.
        zr = getattr(a, "_raw", None)
        zz = z if zr is None else zr
        fin = torch.isfinite(zz)
        x = torch.where(fin, zz, torch.zeros_like(zz)) / CAP
        y = CAP * (torch.tanh(x) if method == "top_nsigma_softcap" else torch.atanh(x.clamp(-1 + 1e-6, 1 - 1e-6)))
        y = torch.where(fin, y, torch.full_like(y, NEG_INF))
        return op_mask("top_nsigma", a, y, p, H, False)
    if method == "top_nsigma":
        if finite:
            sd = z.std(-1, unbiased=False, keepdim=True)
        else:
            fin = torch.isfinite(z)
            cnt = fin.sum(-1, keepdim=True).clamp_min(1)
            zz = torch.where(fin, z, 0.0)
            mean = zz.sum(-1, keepdim=True) / cnt
            sd = ((zz * zz).sum(-1, keepdim=True) / cnt - mean * mean).clamp_min(0).sqrt()
        return z >= z.max(-1, keepdim=True).values - a[:, None] * sd
    if method == "p_less":
        L = (p * p).sum(-1, keepdim=True)
        return p >= L
    if method == "p_less_norm":
        nV = torch.isfinite(z).sum(-1, keepdim=True).to(p.dtype)
        L = (p * p).sum(-1, keepdim=True)
        base = 1.0 / nV
        thr = torch.where(nV > 1, (L - base) / (1 - base).clamp_min(1e-30), L)
        return p >= thr
    if method == "top_p":
        def keep_fn(ps, rows):
            aa = a[rows][:, None] if not isinstance(rows, slice) else a[:, None]
            return (ps.cumsum(-1) - ps) < aa
        def need_full(vals, ks):
            return ks[:, -1]  # last of the K candidates still inside the nucleus
        return _prefix_keep(p, keep_fn, need_full)
    if method == "top_h":
        def keep_fn(ps, rows):
            aa = (a[rows] if not isinstance(rows, slice) else a)[:, None]
            HH = (H[rows] if not isinstance(rows, slice) else H)[:, None]
            S = ps.cumsum(-1).clamp_min(1e-30)
            C = torch.special.entr(ps).cumsum(-1)          # = -sum p log p
            Hk = C / S + torch.log(S)                      # entropy of renormalised prefix
            ok = Hk <= aa * HH * (1 + 1e-5) + 1e-12
            ok[:, 0] = True
            return ok.int().cumprod(-1).bool()
        def need_full(vals, ks):
            return ks[:, -1]
        return _prefix_keep(p, keep_fn, need_full)
    if method == "typical":
        logp = torch.log(p.clamp_min(1e-45))
        dev = (-logp - H[:, None]).abs()
        dev = torch.where(p > 0, dev, torch.full_like(dev, float("inf")))
        ds, order = torch.sort(dev, dim=-1)
        ps = p.gather(1, order)
        keep_sorted = (ps.cumsum(-1) - ps) < a[:, None]
        return torch.zeros_like(p, dtype=torch.bool).scatter_(1, order, keep_sorted)
    raise ValueError(f"unknown method {method}")


def apply_penalties(z: torch.Tensor, counts: torch.Tensor, presence: torch.Tensor,
                    frequency: torch.Tensor, repetition: torch.Tensor) -> torch.Tensor:
    """OpenAI-style presence/frequency + CTRL-style repetition penalty (vLLM semantics).

    counts [n,V] occurrences of each token in the generated output so far.
    """
    seen = counts > 0
    rep = repetition[:, None]
    z = torch.where(seen & (rep != 1.0), torch.where(z > 0, z / rep, z * rep), z)
    z = z - frequency[:, None] * counts.to(z.dtype) - presence[:, None] * seen.to(z.dtype)
    return z


def _ent(p: torch.Tensor) -> torch.Tensor:
    return torch.special.entr(p).sum(-1)


def _base(z: torch.Tensor, T: torch.Tensor, all_t1: bool):
    p1 = torch.softmax(z, -1)
    H1 = _ent(p1)
    pmax, amax = p1.max(-1)
    if all_t1:
        return p1, H1, pmax, amax, z, p1, H1
    zt = z / T[:, None]
    pT = torch.softmax(zt, -1)
    return p1, H1, pmax, amax, zt, pT, _ent(pT)


def _finalize(zt: torch.Tensor, pT: torch.Tensor, keep: torch.Tensor):
    final = zt.masked_fill(~keep, float("-inf"))
    kf = keep.to(pT.dtype)
    mass = (pT * kf).sum(-1)
    S = (torch.special.entr(pT) * kf).sum(-1)
    Hq = S / mass.clamp_min(1e-30) + torch.log(mass.clamp_min(1e-30))
    return final, mass, Hq, kf.sum(-1)


THRESH = {"min_p", "epsilon", "eta", "top_nsigma", "p_less", "p_less_norm", "greedy"}


def _thresh_final(method: str, a, zt, pT, HT, amax, finite: bool):
    keep = op_mask(method, a, zt, pT, HT, finite)
    keep = keep.scatter(1, amax[:, None], True)
    return (keep,) + _finalize(zt, pT, keep)


_COMPILED = {}


def _compiled(name, fn):
    if os.environ.get("LCS_NO_COMPILE"):
        return fn
    if name not in _COMPILED:
        try:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 64
            _COMPILED[name] = torch.compile(fn, dynamic=True)
        except Exception:
            _COMPILED[name] = fn
    return _COMPILED[name]


def _call(name, fn, *args):
    try:
        return _compiled(name, fn)(*args)
    except Exception as e:  # never let compilation problems break generation
        if os.environ.get("LCS_STRICT"):
            raise
        _COMPILED[name] = fn
        return fn(*args)


def process_rows(z: torch.Tensor, T: torch.Tensor, ops: list[tuple[str, torch.Tensor]],
                 all_t1: bool = False, has_ban: bool = True, pend: torch.Tensor | None = None):
    """Process a group of rows sharing the same op *methods* (params may differ per row).

    z [n,V] float32 raw logits (bans / penalties applied), T [n] temperatures.
    pend [n] optional end-pressure stat (computed by the caller before bans).
    Returns (final_logits [n,V], stats [n,7], p1 [n,V] raw probs, pT, keep|None).
    """
    tl = next((p for m, p in ops if m == "temp_last"), None)
    ops = [(m, p) for m, p in ops if m != "temp_last"]
    if tl is not None:  # truncation decided at T, sampling at tl ("temperature last" when T = 1)
        _, stats, p1, pT, keep = process_rows(z, T, ops, all_t1, has_ban, pend)
        if keep is None:
            keep = torch.isfinite(z)
        zl = z / tl[:, None]
        final, mass, Hq, kept = _call("final", _finalize, zl, torch.softmax(zl, -1), keep)
        stats = stats.clone()
        stats[:, 3], stats[:, 4], stats[:, 5] = kept, mass, Hq
        return final, stats, p1, pT, keep
    p1, H1, pmax, amax, zt, pT, HT = _call(f"base{int(all_t1)}", _base, z, T, all_t1)
    if pend is None:
        pend = torch.full_like(H1, float("nan"))
    if not ops:
        nfin = torch.isfinite(z).sum(-1).to(torch.float32)
        stats = torch.stack([H1, HT, pmax, nfin, torch.ones_like(H1), HT, pend], dim=-1)
        return zt, stats, p1, pT, None
    if len(ops) == 1 and ops[0][0] in THRESH:
        method, a = ops[0]
        keep, final, mass, Hq, kept = _call(f"th_{method}_{int(not has_ban)}", _thresh_final,
                                            method, a, zt, pT, HT, amax, not has_ban)
    else:
        keep = None
        cz, cp, cH, cfin = zt, pT, HT, not has_ban
        for k, (method, param) in enumerate(ops):
            if method == "keep_top":
                m = op_mask(method, param, zt, pT, HT, not has_ban)
                keep = m if keep is None else (keep | m)
            else:
                if method in ("min_k", "top_nsigma_softcap", "top_nsigma_uncap"):
                    param._raw = z
                m = op_mask(method, param, cz, cp, cH, cfin)
                keep = m if keep is None else (keep & m)
            keep.scatter_(1, amax[:, None], True)  # min_keep = 1 (raw argmax)
            if k + 1 < len(ops):
                cz = zt.masked_fill(~keep, float("-inf"))
                cp = torch.softmax(cz, -1)
                cH = _ent(cp)
                cfin = False
        final, mass, Hq, kept = _call("final", _finalize, zt, pT, keep)
    stats = torch.stack([H1, HT, pmax, kept, mass, Hq, pend], dim=-1)
    return final, stats, p1, pT, keep


def parse_ops(cfg: dict) -> tuple[float, tuple[tuple[str, float], ...]]:
    """cfg = {"T": 1.0, "ops": [["top_p", 0.9], ...]}  (or "method"/"param" shorthand)."""
    T = float(cfg.get("T", 1.0))
    if "ops" in cfg:
        ops = tuple((str(m), float(p)) for m, p in cfg["ops"])
    else:
        m = cfg.get("method", "temp")
        ops = ((str(m), float(cfg.get("param", 0.0))),)
    for m, p in ops:
        if m not in METHODS:
            raise ValueError(f"unknown sampler method {m!r}")
        if m == "top_k" and p < 1:
            raise ValueError("top_k needs k >= 1")
    if any(m == "greedy" for m, _ in ops):
        T = 1.0
    if not (T > 0 and math.isfinite(T)):
        raise ValueError("temperature must be positive and finite (use greedy op for T=0)")
    ops = tuple((m, p) for m, p in ops if m != "temp")
    return T, ops
