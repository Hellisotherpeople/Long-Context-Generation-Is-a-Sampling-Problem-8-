import os, torch
import lcs.samplers as S
torch.manual_seed(0)
n, V = 64, 151936
base = -1.2 * torch.log(torch.arange(1, V + 1, device="cuda").float())
z = (base[None, :] + torch.randn(n, V, device="cuda") * 1.5)[:, torch.randperm(V, device="cuda")] * 2.0
z[:, :5] = float("-inf")
T = torch.rand(n, device="cuda") * 1.5 + 0.5
cfgs = {"temp": [], "min_p": [("min_p", torch.full((n,), .1, device="cuda"))],
        "p_less": [("p_less", torch.zeros(n, device="cuda"))], "p_less_norm": [("p_less_norm", torch.zeros(n, device="cuda"))],
        "eta": [("eta", torch.full((n,), 9e-4, device="cuda"))], "epsilon": [("epsilon", torch.full((n,), 3e-4, device="cuda"))],
        "top_nsigma": [("top_nsigma", torch.ones(n, device="cuda"))], "greedy": [("greedy", torch.zeros(n, device="cuda"))],
        "top_p": [("top_p", torch.full((n,), .9, device="cuda"))], "top_h": [("top_h", torch.full((n,), .4, device="cuda"))],
        "top_k": [("top_k", torch.full((n,), 50., device="cuda"))], "typical": [("typical", torch.full((n,), .9, device="cuda"))],
        "chain": [("top_k", torch.full((n,), 20., device="cuda")), ("top_p", torch.full((n,), .8, device="cuda"))]}
ok = True
for k, ops in cfgs.items():
    os.environ["LCS_NO_COMPILE"] = "1"; S._COMPILED.clear()
    fe, se, pe, _, ke = S.process_rows(z.clone(), T, ops, False, True)
    os.environ.pop("LCS_NO_COMPILE"); os.environ["LCS_STRICT"] = "1"; S._COMPILED.clear()
    fc, sc, pc, _, kc = S.process_rows(z.clone(), T, ops, False, True)
    same_mask = torch.equal(torch.isfinite(fe), torch.isfinite(fc))
    dstat = (se[:, :6] - sc[:, :6]).abs().max().item()
    print(f"{k:12s} mask_equal={same_mask} max|dstats|={dstat:.2e} kept_med={se[:,3].median().item():.0f}")
    ok &= same_mask and dstat < 1e-3
print("ALL OK" if ok else "MISMATCH")
