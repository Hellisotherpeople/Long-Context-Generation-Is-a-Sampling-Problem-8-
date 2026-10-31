"""Check the temp_last pseudo-op and chained-op order against naive per-row references.
python tools/test_temp_last.py"""
import sys
import torch
sys.path.insert(0, ".")
from lcs.samplers import process_rows

torch.manual_seed(0)
V = 5000
z = torch.randn(64, V) * 3
z[:, 0] += torch.linspace(0, 12, 64)  # from flat to very confident rows


def ref_mask(zz, method, a):
    p = torch.softmax(zz, -1)
    if method == "min_p":
        return p >= a * p.max(-1, keepdim=True).values
    if method == "top_p":
        ps, idx = p.sort(-1, descending=True)
        c = ps.cumsum(-1)
        keep_sorted = (c - ps) < a
        m = torch.zeros_like(p, dtype=torch.bool)
        return m.scatter(-1, idx, keep_sorted)
    if method == "top_k":
        th = zz.topk(int(a), -1).values[:, -1:]
        return zz >= th
    raise ValueError(method)


bad = 0
for method, a in (("min_p", 0.1), ("min_p", 0.3), ("top_p", 0.9), ("top_k", 50)):
    for t in (0.7, 1.5, 3.0):
        n = z.shape[0]
        final, *_ = process_rows(z.clone(), torch.ones(n), [(method, torch.full((n,), a)), ("temp_last", torch.full((n,), t))])
        m = ref_mask(z, method, a)
        m[torch.arange(n), z.argmax(-1)] = True
        ref = (z / t).masked_fill(~m, float("-inf"))
        ok = torch.equal(torch.isfinite(final), torch.isfinite(ref)) and torch.allclose(final[torch.isfinite(final)], ref[torch.isfinite(ref)])
        bad += not ok
        print(f"temp_last {method} {a} t={t}: {'OK' if ok else 'MISMATCH'} (mean kept {torch.isfinite(final).sum(-1).float().mean():.1f})")
# chain order: top_k 50 -> top_p 0.9 (sequential renormalisation) vs naive
n = z.shape[0]
for T in (1.0, 2.0):
    final, *_ = process_rows(z.clone(), torch.full((n,), T), [("top_k", torch.full((n,), 50.)), ("top_p", torch.full((n,), 0.9))])
    zt = z / T
    m1 = ref_mask(zt, "top_k", 50)
    m2 = ref_mask(zt.masked_fill(~m1, float("-inf")), "top_p", 0.9) & m1
    m2[torch.arange(n), z.argmax(-1)] = True
    ok = torch.equal(torch.isfinite(final), m2)
    bad += not ok
    final_r, *_ = process_rows(z.clone(), torch.full((n,), T), [("top_p", torch.full((n,), 0.9)), ("top_k", torch.full((n,), 50.))])
    print(f"chain top_k->top_p T={T}: {'OK' if ok else 'MISMATCH'}; kept {torch.isfinite(final).sum(-1).float().mean():.1f} vs reversed {torch.isfinite(final_r).sum(-1).float().mean():.1f}")
print("ALL OK" if not bad else f"{bad} MISMATCHES")
