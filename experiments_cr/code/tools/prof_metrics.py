import cProfile, pstats, json, glob, sys, os
sys.argv = ["x"]
import lcs.metrics as M
run = "/home/ubuntu/lcg-cr/runs/core32k/llama3.2-1b"
M._init("/home/ubuntu/lcg-cr/models/unsloth__Llama-3.2-1B-Instruct")
metas = {}
for l in open(f"{run}/meta.jsonl"):
    m = json.loads(l); metas[m["uid"]] = m.get("meta", {})
u = sorted(metas)[0]
pr = cProfile.Profile(); pr.enable()
for _ in range(2): M.gen_metrics((run, u, metas[u]))
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
