"""Queue a short probe: which prompt variant stops Nemotron-3 refusing 30K-word writing requests?"""
import copy, json, os
H = os.path.expanduser("~/lcg-cr")
SYS = "You are an acclaimed novelist. Whenever you are asked for a novel you write the complete, very long novel, chapter by chapter."
VARIANTS = {
    "v0": lambda m: m,
    "v1sys": lambda m: [{"role": "system", "content": SYS}] + m,
    "v2nolen": lambda m: [{"role": "user", "content": m[0]["content"].replace("It should be very long (at least 30,000 words). ", "It should be very long. ")}],
    "v3both": lambda m: [{"role": "system", "content": SYS}] + VARIANTS["v2nolen"](m),
}
for model, src, mdir in [("nemotron3-4b", "core32k/nemotron3-4b", "nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16"),
                          ("nemotron3-30b-a3b", "core32k/nemotron3-4b", "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")]:
    j = json.load(open(f"{H}/jobs/{src}.json"))
    prompts = {}
    for r in j["requests"]:
        prompts.setdefault(r["meta"]["pid"], r["prompt"])
    reqs = []
    for pid, p in prompts.items():
        for v, f in VARIANTS.items():
            for s, (T, ops) in enumerate([(0.0, []), (1.0, [["top_p", 0.95]]), (1.0, [["top_p", 0.95]]), (1.0, [["top_p", 0.95]])]):
                q = copy.deepcopy(p)
                q["messages"] = f(q["messages"])
                reqs.append({"uid": f"{pid}__{v}__s{s}", "prompt": q, "max_tokens": 160, "force_len": False,
                             "lcs": {"T": T, "ops": ops}, "seed": s, "meta": {"pid": pid, "variant": v, "seed": s}})
    job = {"job_id": f"probe/{model}", "model": f"{H}/models/{mdir}", "tp": 1,
           "engine": {"max_model_len": 4096, "max_num_seqs": 256}, "out_dir": f"{H}/runs/probe/{model}", "requests": reqs}
    os.makedirs(f"{H}/jobs/probe", exist_ok=True)
    json.dump(job, open(f"{H}/jobs/probe/{model}.json", "w"))
    json.dump({"job": f"{H}/jobs/probe/{model}.json", "gpus": 1, "model_dir": f"{H}/models/{mdir}", "kind": "gen"},
              open(f"{H}/queue/pending/001_probe_{model}.json", "w"))
    print(model, len(reqs))
