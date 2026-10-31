"""Add another seed to an existing job: same requests with uid suffix __s<k>, written into the SAME run dir.

python3 tools/make_seed.py <prio> <seed> <exp>/<model> [<exp>/<model> ...]
"""
import copy
import json
import os
import sys

H = os.path.expanduser("~/lcg-cr")


def main():
    prio, seed, ems = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    for em in ems:
        exp, model = em.split("/")
        j = json.load(open(f"{H}/jobs/{em}.json"))
        reqs = []
        for r in j["requests"]:
            if not r["uid"].endswith("__s0"):
                continue
            r = copy.deepcopy(r)
            r["uid"] = r["uid"][:-4] + f"__s{seed}"
            r.setdefault("meta", {})["seed"] = seed
            reqs.append(r)
        j2 = dict(j)
        j2["requests"] = reqs
        jf = f"{H}/jobs/{exp}/{model}__seed{seed}.json"
        json.dump(j2, open(jf, "w"))
        q = {"job": jf, "gpus": int(j.get("tp", 1)), "model_dir": j["model"], "kind": "gen"}
        for fn in os.listdir(f"{H}/queue/done"):
            if fn.endswith(f"_{exp}_{model}.json"):
                env = json.load(open(f"{H}/queue/done/{fn}")).get("env")
                if env:
                    q["env"] = env
        json.dump(q, open(f"{H}/queue/pending/{prio}_{exp}_{model}__seed{seed}.json", "w"))
        print(em, "seed", seed, "requests", len(reqs))


if __name__ == "__main__":
    main()
