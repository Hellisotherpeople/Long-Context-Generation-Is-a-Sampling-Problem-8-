"""Soft-cap mechanism experiment: top-nsigma with its set computed on soft-capped (Qwen, Ministral) or un-capped
(Gemma) raw logits, PG-19 book continuation, same 16 books x 5 temperatures as book32k."""
import copy, json, os
H = os.path.expanduser("~/lcg-cr")
for model, method, tag in [("qwen3.5-4b-base", "top_nsigma_softcap", "nsigma1sc"),
                           ("gemma4-e4b-base", "top_nsigma_uncap", "nsigma1uc"),
                           ("ministral3-3b-base", "top_nsigma_softcap", "nsigma1sc")]:
    j = json.load(open(f"{H}/jobs/book32k/{model}.json"))
    reqs = []
    for r in j["requests"]:
        if r["meta"]["sampler"] != "nsigma1":
            continue
        r = copy.deepcopy(r)
        r["lcs"]["ops"] = [[method, 1.0]]
        r["meta"]["sampler"] = tag
        r["uid"] = r["uid"].replace("nsigma1@", f"{tag}@")
        reqs.append(r)
    j.update(job_id=f"capx/{model}", out_dir=f"{H}/runs/capx/{model}", requests=reqs)
    os.makedirs(f"{H}/jobs/capx", exist_ok=True)
    json.dump(j, open(f"{H}/jobs/capx/{model}.json", "w"))
    src = [f for f in os.listdir(f"{H}/queue/done") if f.endswith(f"_book32k_{model}.json")]
    old = json.load(open(f"{H}/queue/done/{src[0]}"))
    q = {k: old[k] for k in ("job", "gpus", "model_dir", "kind", "env") if k in old}
    q["job"] = f"{H}/jobs/capx/{model}.json"
    json.dump(q, open(f"{H}/queue/pending/024_capx_{model}.json", "w"))
    print(model, len(reqs), q)
