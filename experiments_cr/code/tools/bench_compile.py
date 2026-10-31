import time, torch
import lcs.samplers as S
torch.manual_seed(0)
n, V = 300, 248320
# realistic-ish LM logits: zipfian head + gaussian bulk
base = -1.2 * torch.log(torch.arange(1, V + 1, device="cuda").float())
z = (base[None, :] + torch.randn(n, V, device="cuda") * 1.5)[:, torch.randperm(V, device="cuda")] * 2.0
z[:, :5] = float("-inf")
T = torch.full((n,), 1.3, device="cuda")
def bench(fn, ops, label):
    for _ in range(3): fn(z, T, ops, False, True)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20): fn(z, T, ops, False, True)
    torch.cuda.synchronize(); print(f"{label:28s} {(time.time()-t)/20*1000:6.2f} ms")
cfgs = {"temp": [], "min_p": [("min_p", torch.full((n,), .1, device="cuda"))],
        "p_less": [("p_less", torch.zeros(n, device="cuda"))],
        "top_nsigma": [("top_nsigma", torch.ones(n, device="cuda"))],
        "top_p": [("top_p", torch.full((n,), .9, device="cuda"))],
        "top_h": [("top_h", torch.full((n,), .4, device="cuda"))],
        "top_k": [("top_k", torch.full((n,), 50., device="cuda"))]}
for k, ops in cfgs.items(): bench(S.process_rows, ops, f"eager {k}")
comp = torch.compile(S.process_rows, dynamic=True)
for k, ops in cfgs.items():
    try: bench(comp, ops, f"compiled {k}")
    except Exception as e: print("compile fail", k, repr(e)[:200])
