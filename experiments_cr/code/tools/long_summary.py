"""Per-configuration summary of 256K / 1M generations: loop onset, looped fraction, entropy, longest zero-entropy run."""
import json
import numpy as np
for exp, m in [("long256k", "qwen3.5-4b"), ("long256k", "qwen3.5-9b"), ("long1000k", "qwen3.5-0.8b"),
               ("long1000k", "nemotron3-30b-a3b")]:
    try:
        lines = open(f"/home/ubuntu/lcg-cr/runs/{exp}/{m}/metrics.jsonl").readlines()
    except FileNotFoundError:
        continue
    print("==", exp, m)
    for l in lines:
        r = json.loads(l)
        w = r["win"]
        rep = np.array([np.nan if x is None else x for x in w["rep32"]], float)
        H = np.array([np.nan if x is None else x for x in w["H1"]], float)
        best = cur = 0
        for v in (H < 0.01):
            cur = cur + 1 if v else 0
            best = max(best, cur)
        print(f"{r['uid']:40s} onset={r['loop_onset_win']} looped_frac={np.nanmean(rep >= 0.5):.2f} "
              f"H_2nd_half={np.nanmean(H[len(H) // 2:]):.3f} longest_H0_run={best}K")
