"""Split the unfinished requests of a running generation job into N new jobs with their own out_dirs.

python3 tools/split_job.py <exp/model> <N> <prio>   (kill the original job first: tools/killjob.sh <pattern>)
New runs: runs/<exp>/<model>__p<i>; analyses strip the "__p<i>" suffix.  The original run keeps what it finished.
"""
import json, os, sys
H = os.path.expanduser("~/lcg-cr")
em, n, prio = sys.argv[1], int(sys.argv[2]), sys.argv[3]
exp, model = em.split("/")
j = json.load(open(f"{H}/jobs/{em}.json"))
done = set()
mf = f"{H}/runs/{em}/meta.jsonl"
if os.path.exists(mf):
    done = {json.loads(l)["uid"] for l in open(mf) if l.strip()}
todo = [r for r in j["requests"] if r["uid"] not in done]
old = [f for f in os.listdir(f"{H}/queue/done") + os.listdir(f"{H}/queue/running") if f.endswith(f"_{exp}_{model}.json")]
env = {}
for f in old:
    for d in ("done", "running"):
        p = f"{H}/queue/{d}/{f}"
        if os.path.exists(p):
            env = json.load(open(p)).get("env", {}) or env
for i in range(n):
    part = dict(j)
    part["requests"] = todo[i::n]
    part["job_id"] = f"{em}__p{i}"
    part["out_dir"] = f"{H}/runs/{em}__p{i}"
    jf = f"{H}/jobs/{exp}/{model}__p{i}.json"
    json.dump(part, open(jf, "w"))
    q = {"job": jf, "gpus": int(j.get("tp", 1)), "model_dir": j["model"], "kind": "gen"}
    if env:
        q["env"] = env
    json.dump(q, open(f"{H}/queue/pending/{prio}_{exp}_{model}__p{i}.json", "w"))
    print("queued", jf, len(part["requests"]))
print("done already:", len(done), "remaining:", len(todo))
