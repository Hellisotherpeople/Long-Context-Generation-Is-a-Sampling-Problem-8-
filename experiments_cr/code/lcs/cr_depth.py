"""R12 context-depth analysis (pre-registered: T=1 entropy vs depth D; quality gap tight - tail-admitting vs D).
python -m lcs.cr_depth -> results/cr_depth.txt, results/cr_depth_gen.csv"""
import glob, os
import numpy as np, pandas as pd
from .cr_tables import load
H = os.path.expanduser("~/lcg-cr"); K = ["model", "uid"]
TAIL = {"topp0.9", "topp0.95", "toph0.4", "topk50", "temp"}; TIGHT = {"minp0.1", "minp0.05", "nsigma1", "pless"}


def main():
    md, wd, _ = load("depth-native")
    md["D"] = md.uid.str.extract(r"^D(\d+)__")[0].astype(int)
    md = md.join(wd.groupby(K).j_overall.mean().rename("q"), on=K)
    rows = []
    for f in glob.glob(f"{H}/runs/depth-native/*/stats/*.npz"):
        s = np.load(f)["stats"][:2048]
        rows.append(dict(model=f.split("/")[-3], uid=os.path.basename(f).split(".p0")[0],
                         H1=float(np.nanmean(s[:, 0])), kept=float(np.nanmedian(s[:, 3]))))
    md = md.merge(pd.DataFrame(rows), on=K, how="left")
    md["grp"] = np.where(md.sampler.isin(TAIL), "tailadm", np.where(md.sampler.isin(TIGHT), "tight", "greedy"))
    md.to_csv(f"{H}/results/cr_depth_gen.csv", index=False)
    out = ["T=1 mean entropy (nats, first 2K generated tokens) by depth:",
           md[md["T"] == 1.0].pivot_table(index="model", columns="D", values="H1").round(2).to_string(), ""]
    g = md[md.grp != "greedy"].groupby(["T", "D", "grp"]).q.mean().unstack("grp"); g["gap"] = g.tight - g.tailadm
    out += ["Judged quality, tight vs tail-admitting, by T and D (all models):", g.round(2).to_string(), ""]
    per = md[md.grp != "greedy"].groupby(["model", "T", "D", "grp"]).q.mean().unstack("grp")
    per["gap"] = per.tight - per.tailadm
    out += ["Per-model gap at T=1.3:", per.xs(1.3, level="T").gap.unstack("D").round(2).to_string(), "",
            "Per-model gap at T=1.6:", per.xs(1.6, level="T").gap.unstack("D").round(2).to_string(), "",
            "n per (model,D): " + str(md.groupby(["model", "D"]).size().min()) + "-" + str(md.groupby(["model", "D"]).size().max())]
    txt = "\n".join(out); open(f"{H}/results/cr_depth.txt", "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
