"""Print the opening of generations in a run: run_show.py <exp>/<model> [n] [nchars]"""
import json, sys
import numpy as np
from lcs.tok import load_tokenizer
em = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
nc = int(sys.argv[3]) if len(sys.argv) > 3 else 500
run = f"/home/ubuntu/lcg-cr/runs/{em}"
L = [json.loads(l) for l in open(run + "/meta.jsonl") if l.strip()]
tok = load_tokenizer(L[0]["model"])
step = max(1, len(L) // n)
for l in L[::step][:n]:
    t = tok.decode(np.load(f"{run}/gen/{l['uid']}.npz")["tokens"][:400], skip_special_tokens=True)
    print("==", l["uid"])
    print(t[:nc].replace("\n", " / "))
