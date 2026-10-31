"""Realised candidate-set size per sampler x T from lcs per-step stats (first 2048 tokens, pre-loop).
python -m lcs.cr_pool r1_16k -> results/cr_pool_<exp>.txt"""
import glob, os, re, sys, numpy as np, pandas as pd
H = os.path.expanduser("~/lcg-cr")
exp = sys.argv[1]; rows = []
for f in glob.glob(f"{H}/runs/{exp}/*/stats/*.npz"):
    m = f.split("/")[-3]; u = os.path.basename(f)
    mm = re.match(r".*__(.+?)(?:@T([0-9.]+))?__s\d+\.p0\.npz", u)
    if not mm: continue
    s = np.load(f)["stats"][:2048, 3]
    s = s[np.isfinite(s)]
    if len(s): rows.append(dict(model=m, sampler=mm.group(1), T=float(mm.group(2) or 0), mean=s.mean(), med=np.median(s), single=(s <= 1).mean()))
d = pd.DataFrame(rows)
g = d.groupby(["sampler", "T"]).agg(mean_kept=("mean", "mean"), median_kept=("med", "median"), single_frac=("single", "mean"), n=("mean", "size")).round(2).reset_index()
w = g.pivot(index="sampler", columns="T", values="mean_kept")
top = max(w.columns); w[f"growth_{top}/1.0"] = (w[top] / w[1.0]).round(1) if 1.0 in w and top != 1.0 else None
txt = g.to_string(index=False) + "\n\nmean kept by T:\n" + w.to_string()
open(f"{H}/results/cr_pool_{exp}.txt", "w").write(txt + "\n"); print(txt)
