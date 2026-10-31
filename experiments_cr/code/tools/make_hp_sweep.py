"""Hyperparameter robustness sweep in natural multi-turn generation (exp mt32k_hp).

Every rule's main hyperparameter across T in {0.7, 1.0, 1.3, 1.6}, plus keep_top in {2, 3, 5} on P-less and top-nsigma.
Configurations already covered by mt32k / mtk2_32k are skipped.  Prompts/engine/multiturn settings from mt32k/<model>.
python3 tools/make_hp_sweep.py <prio> <model> [<model> ...]
"""
import copy
import json
import os
import sys

H = os.path.expanduser("~/lcg-cr")
TEMPS = [0.7, 1.0, 1.3, 1.6]
GRID = ([(f"topp{p}", [["top_p", p]]) for p in (0.8, 0.98)] +
        [(f"minp{m}", [["min_p", m]]) for m in (0.02, 0.2)] +
        [(f"nsigma{n}", [["top_nsigma", n]]) for n in (0.5, 1.5, 2.0)] +
        [(f"toph{a}", [["top_h", a]]) for a in (0.2, 0.6, 0.8)] +
        [(f"topk{k}", [["top_k", k]]) for k in (10, 20, 100)] +
        [(f"eps0.0003", [["epsilon", 3e-4]]), ("eta0.0009", [["eta", 9e-4]]), ("typical0.95", [["typical", 0.95]])] +
        [(f"pless+k{k}", [["p_less", 0.0], ["keep_top", k]]) for k in (3, 5)] +
        [(f"nsigma1+k{k}", [["top_nsigma", 1.0], ["keep_top", k]]) for k in (3, 5)] +
        [("minp0.05+k2", [["min_p", 0.05], ["keep_top", 2]]), ("topp0.95+k2", [["top_p", 0.95], ["keep_top", 2]])])


def main():
    prio, models = sys.argv[1], sys.argv[2:]
    for model in models:
        T = json.load(open(f"{H}/jobs/mt32k/{model}.json"))
        prompts = {}
        for r in T["requests"]:
            if r["meta"].get("seed", 0) == 0:
                prompts.setdefault(r["meta"]["pid"], r)
        reqs = []
        for pid, base in sorted(prompts.items()):
            for name, ops in GRID:
                for t in TEMPS:
                    r = copy.deepcopy(base)
                    r["lcs"] = {"T": t, "ops": ops}
                    r["meta"].update(sampler=name, T=t, seed=0)
                    r["uid"] = f"{pid}__{name}@T{t}__s0"
                    reqs.append(r)
        J = copy.deepcopy(T)
        J.update(job_id=f"mt32k_hp/{model}", out_dir=f"{H}/runs/mt32k_hp/{model}", requests=reqs)
        os.makedirs(f"{H}/jobs/mt32k_hp", exist_ok=True)
        jf = f"{H}/jobs/mt32k_hp/{model}.json"
        json.dump(J, open(jf, "w"))
        q = {"job": jf, "gpus": int(J.get("tp", 1)), "model_dir": J["model"], "kind": "gen"}
        for d in ("done",):
            for fn in os.listdir(f"{H}/queue/{d}"):
                if fn.endswith(f"_mt32k_{model}.json"):
                    env = json.load(open(f"{H}/queue/{d}/{fn}")).get("env")
                    if env:
                        q["env"] = env
        json.dump(q, open(f"{H}/queue/pending/{prio}_mt32k_hp_{model}.json", "w"))
        print(model, "configs", len(GRID) * len(TEMPS), "requests", len(reqs))


if __name__ == "__main__":
    main()
