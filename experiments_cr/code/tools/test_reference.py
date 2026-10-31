"""Check batched samplers against naive per-row reference implementations of the definitions."""
import os
os.environ["LCS_NO_COMPILE"] = "1"
import numpy as np, torch
import lcs.samplers as S
rng = np.random.default_rng(0)
n, V = 40, 600
z = (rng.standard_normal((n, V)) * rng.uniform(0.5, 4, (n, 1))).astype(np.float32)
z[:, :3] = -np.inf
T = rng.uniform(0.5, 2.0, n).astype(np.float32)

def softmax(x):
    m = np.max(x); e = np.exp(x - m); e[~np.isfinite(x)] = 0; return e / e.sum()
def H(p):
    q = p[p > 0]; return float(-(q * np.log(q)).sum())
def ref(method, a, zr, t):
    zt = zr / t; p = softmax(zt); keep = np.zeros(V, bool); fin = np.isfinite(zr)
    order = np.argsort(-p, kind="stable")
    if method == "top_p":
        c = 0
        for i in order:
            keep[i] = True; c += p[i]
            if c >= a: break
    elif method == "min_p": keep = p >= a * p.max()
    elif method == "p_less": keep = p >= (p ** 2).sum()
    elif method == "top_nsigma": keep = zt >= zt[fin].max() - a * zt[fin].std()
    elif method == "eta": keep = p >= min(a, np.sqrt(a) * np.exp(-H(p)))
    elif method == "epsilon": keep = p >= a
    elif method == "top_k": keep[order[:int(a)]] = True
    elif method == "top_h":
        Hp = H(p); sel = []
        for i in order:
            cand = sel + [i]; q = p[cand] / p[cand].sum()
            if len(sel) > 0 and H(q) > a * Hp + 1e-6: break
            sel = cand
        keep[sel] = True
    elif method == "typical":
        Hp = H(p); dev = np.where(p > 0, np.abs(-np.log(np.maximum(p, 1e-45)) - Hp), np.inf)
        c = 0
        for i in np.argsort(dev, kind="stable"):
            keep[i] = True; c += p[i]
            if c >= a: break
    keep[np.argmax(zr)] = True
    return keep & fin
tests = [("top_p", .9), ("min_p", .1), ("p_less", 0), ("top_nsigma", 1.0), ("eta", 9e-4), ("epsilon", 3e-4),
         ("top_k", 50), ("top_h", .4), ("typical", .9), ("top_p", .5), ("top_h", .7), ("top_nsigma", 2.0)]
allok = True
for m, a in tests:
    final, stats, *_ = S.process_rows(torch.tensor(z), torch.tensor(T), [(m, torch.full((n,), float(a)))], False, True)
    got = torch.isfinite(final).numpy()
    mism = sum(not np.array_equal(got[r], ref(m, a, z[r], T[r])) for r in range(n))
    sizes = got.sum(1)
    print(f"{m:11s} a={a:<7} rows_mismatch={mism}/{n}  kept mean={sizes.mean():.1f}")
    allok &= mism == 0
print("ALL OK" if allok else "MISMATCH")
# keep_top union check
final, *_ = S.process_rows(torch.tensor(z), torch.tensor(T), [("p_less", torch.zeros(n)), ("keep_top", torch.full((n,), 2.))], False, True)
k = torch.isfinite(final).sum(1)
print("pless+keep_top2 min kept =", int(k.min()), "(must be >= 2)")
# min-k vs exact full-sort reference on raw logits (temperature invariant)
def ref_mink(zr, tau=3.0):
    fin = np.isfinite(zr); l = np.sort(zr[fin])[::-1]; R = max(l[0] - l[-1], 1e-8)
    w = (l[:-1] - l[1:]) / R / np.arange(1, len(l)); k = max(int(np.argmax(w)) + 1, int(np.floor(tau / R)))
    keep = np.zeros(V, bool); keep[np.argsort(-np.where(fin, zr, -np.inf), kind="stable")[:max(1, k)]] = True
    return keep
mm = 0
for t_ in (0.5, 1.0, 2.0):
    final, *_ = S.process_rows(torch.tensor(z), torch.full((n,), t_), [("min_k", torch.full((n,), 3.0))], False, True)
    got = torch.isfinite(final).numpy()
    mm += sum(not np.array_equal(got[r], ref_mink(z[r])) for r in range(n))
print("min_k mismatches across T in {0.5,1,2}:", mm, "kept mean", got.sum(1).mean())
