"""Re-queue Nemotron-3 writing experiments with a prompt variant that avoids its capability refusals.

python3 tools/nemotron_rerun.py <variant: v1sys|v2nolen|v3both> <prio> exp/model [exp/model ...]
The refusal-contaminated run directories are moved to runs/_refusal/<exp>/<model> (kept for the appendix).
"""
import json
import os
import shutil
import sys

H = os.path.expanduser("~/lcg-cr")
SYS = ("You are an acclaimed novelist. Whenever you are asked for a novel you write the complete, very long novel, "
       "chapter by chapter.")


def apply(msgs, v):
    msgs = [dict(m) for m in msgs]
    if v in ("v2nolen", "v3both"):
        msgs[0]["content"] = msgs[0]["content"].replace("It should be very long (at least 30,000 words). ",
                                                        "It should be very long. ")
    if v in ("v1sys", "v3both"):
        msgs = [{"role": "system", "content": SYS}] + msgs
    return msgs


def main():
    v, prio = sys.argv[1], sys.argv[2]
    for em in sys.argv[3:]:
        exp, model = em.split("/")
        jf = f"{H}/jobs/{exp}/{model}.json"
        j = json.load(open(jf))
        if not os.path.exists(jf + ".refusal"):
            shutil.copy(jf, jf + ".refusal")
        for r in j["requests"]:
            r["prompt"]["messages"] = apply(r["prompt"]["messages"], v)
            r.setdefault("meta", {})["prompt_variant"] = v
        json.dump(j, open(jf, "w"))
        run = f"{H}/runs/{exp}/{model}"
        if os.path.exists(run):
            dst = f"{H}/runs/_refusal/{exp}/{model}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(run, dst)
        mdir = j["model"]
        q = {"job": jf, "gpus": int(j.get("tp", 1)), "model_dir": mdir, "kind": "gen"}
        import glob as _g
        for _f in _g.glob(f"{H}/queue/*/*.json"):  # inherit env (e.g. long-context flags) from earlier entries
            _q = json.load(open(_f))
            if _q.get("model_dir") == q["model_dir"] and _q.get("env"):
                q["env"] = _q["env"]
                break
        tag = f"{prio}_{exp}_{model}_{v}.json"
        json.dump(q, open(f"{H}/queue/pending/{tag}", "w"))
        print("queued", tag, len(j["requests"]))


if __name__ == "__main__":
    main()
