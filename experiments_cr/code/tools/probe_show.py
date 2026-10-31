"""Print probe outputs: probe_show.py <model> <variant> [n]"""
import json, sys
import numpy as np
from lcs.tok import load_tokenizer
model, variant = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
run = f"/home/ubuntu/lcg-cr/runs/probe/{model}"
L = [json.loads(l) for l in open(run + "/meta.jsonl")]
tok = load_tokenizer(L[0]["model"])
for l in [x for x in L if x["meta"]["variant"] == variant][:n]:
    t = tok.decode(np.load(f"{run}/gen/{l['uid']}.npz")["tokens"], skip_special_tokens=True)
    print("==", l["uid"], l["finish"])
    print(t[:900].replace("\n", " / "))
