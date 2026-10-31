import numpy as np, torch
from lcs.dry import DryBuffer
rng = np.random.default_rng(0)
V = 50
brk = torch.zeros(V, dtype=torch.bool); brk[[3, 7]] = True
def naive(c, mult=0.8, base=1.75, allowed=2):
    n = len(c); best = np.zeros(V)
    for j in range(1, n):
        L = 0
        while L < j and c[j - 1 - L] == c[n - 1 - L] and not brk[c[n - 1 - L]]:
            L += 1
        best[c[j]] = max(best[c[j]], L)
    pen = np.where(best >= allowed, mult * base ** (best - allowed), 0.0)
    pen[brk.numpy()] = 0
    return pen
ok = True
for trial in range(20):
    base_seq = rng.integers(0, V, 30).tolist()
    seq = (base_seq * 4)[: rng.integers(40, 120)]
    seq = [x if rng.random() > 0.1 else int(rng.integers(0, V)) for x in seq]
    buf = DryBuffer(4, 256, V, "cpu", brk)
    # path 1: init on prompt half, push the rest
    h = len(seq) // 2
    buf.init_row(1, seq[:h])
    for x in seq[h:]:
        buf.push([1], [x])
    # path 2: init all at once
    buf.init_row(2, seq)
    args = (torch.tensor([0.8]), torch.tensor([1.75]), torch.tensor([2.0]))
    p1 = buf.penalty([1], *args)[0].numpy(); p2 = buf.penalty([2], *args)[0].numpy(); ref = naive(seq)
    if not (np.allclose(p1, ref) and np.allclose(p2, ref)):
        ok = False; print("mismatch", trial, np.abs(p1 - ref).max(), np.abs(p2 - ref).max())
print("DRY OK" if ok else "DRY MISMATCH")
# long-sequence init path (Z-algorithm) vs push path
seq = (rng.integers(0, V, 400).tolist() * 30)[:10000]
seq = [x if rng.random() > 0.05 else int(rng.integers(0, V)) for x in seq]
buf = DryBuffer(4, 12000, V, "cpu", brk)
buf.init_row(1, seq)            # n > 8192 -> Z path
buf.init_row(2, seq[:5000])     # E-matrix path
for x in seq[5000:]:
    buf.push([2], [x])
args = (torch.tensor([0.8]), torch.tensor([1.75]), torch.tensor([2.0]))
print("Z-path == push-path:", torch.equal(buf.M[1, :10000], buf.M[2, :10000]),
      np.allclose(buf.penalty([1], *args)[0].numpy(), buf.penalty([2], *args)[0].numpy()))
