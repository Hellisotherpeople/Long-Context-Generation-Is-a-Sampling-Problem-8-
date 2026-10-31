import glob, json, os, sys
import numpy as np
from transformers import AutoTokenizer
run = sys.argv[1]; model = sys.argv[2]
tok = AutoTokenizer.from_pretrained(model)
def load_stats(tag):
    parts = sorted(glob.glob(f"{run}/stats/{tag}.p*.npz"))
    P = {}; S = {}; T = {}
    for p in parts:
        d = np.load(p)
        for i, pos in enumerate(d["pos"]): S[int(pos)] = d["stats"][i]
        for i, tp in enumerate(d["tpos"]): T[int(tp)] = (d["lp1"][i], d["rank1"][i], d["lq"][i])
    return S, T
for f in sorted(glob.glob(f"{run}/gen/*.npz")):
    tag = os.path.basename(f)[:-4]
    toks = np.load(f)["tokens"]
    S, T = load_stats(tag)
    n = len(toks)
    pos = sorted(S); tpos = sorted(T)
    lq = np.array([T[p][2] for p in tpos]); rk = np.array([T[p][1] for p in tpos])
    st = np.array([S[p] for p in pos])
    gaps = n - len(pos)
    print(f"{tag:10s} n={n} stats={len(pos)} (missing {gaps}) tok={len(tpos)} lq_inf={np.sum(~np.isfinite(lq))} "
          f"H1={st[:,0].mean():.2f} HT={st[:,1].mean():.2f} kept={np.median(st[:,3]):.0f}/{st[:,3].mean():.1f} mass={st[:,4].mean():.3f} Hq={st[:,5].mean():.2f} "
          f"rank_mean={rk.mean():.2f} rank1_frac={np.mean(rk==1):.2f}")
    if len(sys.argv) > 3 and tag == sys.argv[3]:
        print(tok.decode(toks[:400]))
        print('.....')
        print(tok.decode(toks[-300:]))
