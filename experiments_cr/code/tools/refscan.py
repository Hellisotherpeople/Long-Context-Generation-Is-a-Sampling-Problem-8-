import json, numpy as np, collections, glob, os, re, sys
from lcs.tok import load_tokenizer
PAT = re.compile(r"(i[’']m sorry|i cannot|i can[’']t|i can not|i am unable|i[’']m unable|unable to (write|provide|fulfill)|as an ai)", re.I)
out = {}
for run in sorted(glob.glob("/home/ubuntu/lcg-cr/runs/*/*")):
    exp = run.split("/")[-2]
    if exp.startswith(("_old", "smoke", "loop", "niah", "book", "human", "depth", "code", "reason", "bookk2", "bookpen")):
        continue
    mf = run + "/meta.jsonl"
    if not os.path.exists(mf) or os.path.getsize(mf) == 0:
        continue
    L = [json.loads(l) for l in open(mf)]
    try:
        tok = load_tokenizer(L[0]["model"])
    except Exception as e:
        print(run, "tok fail", e); continue
    n = r = 0
    for l in L:
        u = l["uid"]
        if "_t" in u.split("__")[-1] and not u.endswith("_t0"): continue
        if "_seg" in u and not u.endswith("_seg0"): continue
        f = f"{run}/gen/{u}.npz"
        if not os.path.exists(f): continue
        t = tok.decode(np.load(f)["tokens"][:60], skip_special_tokens=True)
        n += 1; r += bool(PAT.search(t[:200]))
    print(f"{exp:12s} {run.split('/')[-1]:28s} n={n:4d} refusal={r/max(n,1):.2f}", flush=True)
