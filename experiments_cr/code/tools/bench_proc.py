import time, torch
from lcs.samplers import process_rows
torch.manual_seed(0)
n, V = 300, 248320
z = torch.randn(n, V, device="cuda") * 3
z[:, :5] = float("-inf")
methods = [("top_k", 50.), ("top_p", .9), ("top_p", .95), ("min_p", .05), ("min_p", .1), ("top_nsigma", 1.), ("p_less", 0.), ("top_h", .4), ("temp", 0)]
def run(groups, T=1.3):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        for (m, p), idx in groups:
            zz = z[idx]
            Tt = torch.full((len(idx),), T, device="cuda")
            process_rows(zz, Tt, [(m, torch.full((len(idx),), p, device="cuda"))] if m != "temp" else [], False, True)
    torch.cuda.synchronize(); return (time.time() - t) / 10 * 1000
idx = torch.arange(n, device="cuda")
g9 = [(methods[i], c) for i, c in enumerate(torch.chunk(idx, 9))]
run(g9)
print(f"9 method groups (300 rows): {run(g9):.1f} ms/step")
for m in methods:
    print(m, f"{run([(m, idx)]):.1f} ms for all {n} rows")
