"""Static soft-cap test on identical contexts (CPU): how often is top-nsigma collapsed with / without soft-capping?

For each model we teacher-force 4 PG-19 book prefixes and, at every position, compare the top-nsigma (n=1) collapse
condition  Delta > n*sigma  on (a) the model's own logits and (b) the same logits with the soft-cap added (Qwen) or removed
(Gemma).  Temperature does not enter (the condition is T-invariant), so this isolates the logit-geometry effect from
the trajectory effects of generation.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("THREADS", 32)))
H = os.path.expanduser("~/lcg-cr")
CAP = 30.0


def stats(z):
    top2 = torch.topk(z, 2, dim=-1).values
    d = top2[:, 0] - top2[:, 1]
    sd = z.std(-1, unbiased=False)
    p1 = torch.softmax(z, -1).max(-1).values
    return d, sd, p1, top2[:, 0]


def run(name, mdir, tokdir, cap_mode, n_books=4, L=2048):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.bfloat16)
    if cap_mode == "remove":
        for c in (model.config, getattr(model.config, "text_config", None)):
            if c is not None and getattr(c, "final_logit_softcapping", None):
                c.final_logit_softcapping = None
    model.eval()
    res = {"collapse_own": [], "collapse_alt": [], "p1": [], "zmax_own": [], "zmax_alt": []}
    for f in sorted(glob.glob(f"{H}/data/book_tok/{tokdir}/prefix/b*.npy"))[:n_books]:
        ids = torch.tensor(np.load(f)[:L].astype(np.int64))[None]
        with torch.no_grad():
            z = model(ids).logits[0].float()        # uncapped for Gemma (cap removed), raw for Qwen
        if cap_mode == "remove":
            own, alt = CAP * torch.tanh(z / CAP), z      # model's real (capped) logits vs uncapped
        else:
            own, alt = z, CAP * torch.tanh(z / CAP)      # Qwen raw vs soft-capped
        d0, s0, p1, m0 = stats(own)
        d1, s1, _, m1 = stats(alt)
        res["collapse_own"] += (d0 > s0).tolist()
        res["collapse_alt"] += (d1 > s1).tolist()
        res["p1"] += p1.tolist()
        res["zmax_own"] += m0.tolist()
        res["zmax_alt"] += m1.tolist()
    p1 = np.array(res["p1"])
    out = {"model": name, "n_pos": len(p1), "cap_mode": cap_mode}
    for k in ("own", "alt"):
        c = np.array(res[f"collapse_{k}"])
        out[f"collapse_{k}"] = float(c.mean())
        out[f"collapse_{k}_p1>0.9"] = float(c[p1 > 0.9].mean())
        out[f"collapse_{k}_p1<0.5"] = float(c[p1 < 0.5].mean())
        out[f"zmax_{k}_median"] = float(np.median(res[f"zmax_{k}"]))
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["qwen", "gemma"]
    outs = []
    if "qwen" in which:
        outs.append(run("qwen3.5-0.8b-base", f"{H}/models/Qwen__Qwen3.5-0.8B-Base", "qwen3.5-0.8b-base", "add"))
    if "qwen4" in which:
        outs.append(run("qwen3.5-4b-base", f"{H}/models/Qwen__Qwen3.5-4B-Base", "qwen3.5-4b-base", "add"))
    if "olmo" in which:
        outs.append(run("olmo3-7b-base", f"{H}/models/allenai__Olmo-3-1025-7B", "olmo3-7b-base", "add"))
    if "gemma12" in which:
        outs.append(run("gemma4-12b-base", f"{H}/models/google__gemma-4-12B", "gemma4-12b-base", "remove"))
    if "gemma31" in which:
        outs.append(run("gemma4-31b-it", f"{H}/models/google__gemma-4-31B-it", "gemma4-12b-base", "remove", n_books=2))
    if "gemmae4" in which:
        outs.append(run("gemma4-e4b-base", f"{H}/models/google__gemma-4-E4B", "gemma4-e4b-base", "remove"))
    if "gemmae4it" in which:
        outs.append(run("gemma4-e4b-it", f"{H}/models/google__gemma-4-E4B-it", "gemma4-e4b-base", "remove"))
    if "gemma" in which:
        outs.append(run("gemma4-e2b-base", f"{H}/models/google__gemma-4-E2B", "gemma4-e2b-base", "remove"))
    old = json.load(open(f"{H}/results/softcap_static.json")) if os.path.exists(f"{H}/results/softcap_static.json") else []
    keep = {o["model"]: o for o in old}
    keep.update({o["model"]: o for o in outs})
    json.dump(list(keep.values()), open(f"{H}/results/softcap_static.json", "w"), indent=1)
