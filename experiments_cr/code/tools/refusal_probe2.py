"""Quick first-turn refusal probe for queued chat models (8 prompts x {greedy, top-p .95 T1} x {plain, system})."""
import copy, json, os, sys
H = os.path.expanduser("~/lcg-cr")
SYS = "You are an acclaimed novelist. Whenever you are asked for a novel you write the complete, very long novel, chapter by chapter."
prio = sys.argv[1]
VARIANTS = os.environ.get("VARIANTS", "v0,v1sys").split(",")
for spec in sys.argv[2:]:
    model, tp = (spec.split(":") + ["1"])[:2]
    j = json.load(open(f"{H}/jobs/core32k/{model}.json"))
    prompts = {}
    for r in j["requests"]:
        prompts.setdefault(r["meta"]["pid"], r["prompt"])
    reqs = []
    for pid, p in prompts.items():
        for v in VARIANTS:
            for s, (T, ops) in enumerate([(0.0, []), (1.0, [["top_p", 0.95]])]):
                q = copy.deepcopy(p)
                if v in ("v2nolen", "v3both"):
                    q["messages"] = [dict(m) for m in q["messages"]]
                    q["messages"][0]["content"] = q["messages"][0]["content"].replace(
                        "It should be very long (at least 30,000 words). ", "It should be very long. ")
                if v in ("v1sys", "v3both"):
                    q["messages"] = [{"role": "system", "content": SYS}] + q["messages"]
                reqs.append({"uid": f"{pid}__{v}__s{s}", "prompt": q, "max_tokens": 200, "force_len": False,
                             "lcs": {"T": T, "ops": ops}, "meta": {"pid": pid, "variant": v, "seed": s}})
    eng = dict(j["engine"]); eng.update(max_model_len=4096, max_num_seqs=64)
    job = {"job_id": f"probe/{model}", "model": j["model"], "tp": int(tp), "engine": eng,
           "out_dir": f"{H}/runs/probe/{model}" + os.environ.get("SUFFIX", ""), "requests": reqs}
    jf = f"{H}/jobs/probe/{model}{os.environ.get('SUFFIX', '')}.json"
    json.dump(job, open(jf, "w"))
    src = [f for f in os.listdir(f"{H}/queue/pending") if f.endswith(f"_core32k_{model}.json")]
    env = json.load(open(f"{H}/queue/pending/{src[0]}")).get("env", {}) if src else {}
    q = {"job": jf, "gpus": int(tp), "model_dir": j["model"], "kind": "gen"}
    if env:
        q["env"] = env
    json.dump(q, open(f"{H}/queue/pending/{prio}_probe_{model}{os.environ.get('SUFFIX', '')}.json", "w"))
    print(model, tp, len(reqs), env)
