"""Cross the prompts/engine of one job with the sampler configs of another.

python3 tools/make_clone.py <prio> <template exp/model> <configs exp/model> <new exp/model> [tp] [model_dir_override]
"""
import copy, json, os, sys
H = os.path.expanduser("~/lcg-cr")
prio, tmpl, cfgsrc, new = sys.argv[1:5]
tp = int(sys.argv[5]) if len(sys.argv) > 5 else None
mdir = sys.argv[6] if len(sys.argv) > 6 else None
T = json.load(open(f"{H}/jobs/{tmpl}.json"))
C = json.load(open(f"{H}/jobs/{cfgsrc}.json"))
prompts = {}
for r in T["requests"]:
    key = (r["meta"]["pid"], r["meta"].get("seed", 0))
    prompts.setdefault(key, r)
cfgs = {}
for r in C["requests"]:
    cfgs.setdefault((r["meta"]["sampler"], r["meta"]["T"]), r["lcs"])
reqs = []
for (pid, seed), base in sorted(prompts.items()):
    for (s, t), lcs in sorted(cfgs.items()):
        r = copy.deepcopy(base)
        r["lcs"] = copy.deepcopy(lcs)
        r["meta"].update(sampler=s, T=t)
        tag = "greedy" if s == "greedy" else f"{s}@T{t}"
        r["uid"] = f"{pid}__{tag}__s{seed}"
        reqs.append(r)
exp, model = new.split("/")
J = copy.deepcopy(T)
J.update(job_id=new, out_dir=f"{H}/runs/{new}", requests=reqs)
if tp:
    J["tp"] = tp
if mdir:
    J["model"] = mdir
os.makedirs(f"{H}/jobs/{exp}", exist_ok=True)
json.dump(J, open(f"{H}/jobs/{new}.json", "w"))
q = {"job": f"{H}/jobs/{new}.json", "gpus": int(J.get("tp", 1)), "model_dir": J["model"], "kind": "gen"}
import glob as _g
for _f in _g.glob(f"{H}/queue/*/*.json"):  # inherit env (e.g. long-context flags) from earlier entries
    _q = json.load(open(_f))
    if _q.get("model_dir") == q["model_dir"] and _q.get("env"):
        q["env"] = _q["env"]
        break
json.dump(q, open(f"{H}/queue/pending/{prio}_{exp}_{model}.json", "w"))
print(new, len(reqs), "configs", len(cfgs), "prompts", len(prompts), {k: v for k, v in J.items() if k != "requests"})
