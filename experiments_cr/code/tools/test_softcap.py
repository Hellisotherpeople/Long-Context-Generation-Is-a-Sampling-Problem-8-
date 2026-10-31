"""top_nsigma_softcap / top_nsigma_uncap: reference agreement and temperature invariance."""
import os
os.environ.setdefault("LCS_NO_COMPILE", "1")
import numpy as np
import torch
from lcs import samplers as S

rng = np.random.default_rng(0)
n, V = 64, 5000
z = rng.normal(0, 3, (n, V)).astype(np.float32)
z[np.arange(n), rng.integers(0, V, n)] += rng.uniform(5, 30, n).astype(np.float32)   # peaked rows
zt = torch.tensor(z)


def ref_ns(x, k=1.0):
    return x >= x.max(-1, keepdims=True) - k * x.std(-1, keepdims=True)


def keep_of(method, z_, T):
    out = S.process_rows(z_, torch.full((z_.shape[0],), T), [(method, torch.ones(z_.shape[0]))], False, True)
    return out[4].numpy()


capped = 30 * np.tanh(z / 30)
ok = True
for T in (0.5, 1.0, 2.0):
    k1 = keep_of("top_nsigma_softcap", zt, T)
    r1 = ref_ns(capped)
    r1[np.arange(n), z.argmax(-1)] = True
    ok &= (k1 == r1).all()
    # uncap applied to capped logits recovers plain top-nsigma on the original logits
    k2 = keep_of("top_nsigma_uncap", torch.tensor(capped.astype(np.float32)), T)
    r2 = ref_ns(z)
    r2[np.arange(n), z.argmax(-1)] = True
    agree = (k2 == r2).mean()
    ok &= agree > 0.9999
    print(f"T={T}: softcap exact={bool((k1 == r1).all())} kept(softcap)={k1.sum(1).mean():.1f} "
          f"kept(plain)={r2.sum(1).mean():.1f} uncap agreement={agree:.6f}")
print("ALL OK" if ok else "MISMATCH")
