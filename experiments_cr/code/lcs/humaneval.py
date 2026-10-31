"""Build a self-contained pairwise human-evaluation kit from generated runs.

~/lcg-cr/ana/bin/python -m lcs.humaneval build --exp core32k --models qwen3.5-9b,qwen3.5-27b \
      --configs "topp0.9@T1.0,minp0.1@T1.6,toph0.4@T1.6,nsigma1@T1.0,pless@T1.0,topk50@T1.3" --out ~/lcg-cr/humaneval
Produces <out>/index.html (single file, no server needed), <out>/items.json, <out>/README.md.
Annotators see the task prompt and, for each of two anonymous texts, excerpts from the beginning, middle
and end; they answer three forced-choice questions. 10% of items are attention checks (one side is an
obviously degenerate control). Results download as JSON; analyse with `humaneval analyze`.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import random

import numpy as np

EXC = 700  # tokens per excerpt

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Long text comparison</title>
<style>body{font-family:Georgia,serif;max-width:1400px;margin:auto;padding:20px;background:#fafafa}
.cols{display:flex;gap:20px}.col{flex:1;background:#fff;border:1px solid #ccc;padding:12px;height:60vh;overflow-y:scroll}
h3{margin:4px 0}.ex{white-space:pre-wrap;font-size:15px;line-height:1.45}.lab{color:#888;font-size:12px;margin-top:10px}
.q{margin:10px 0}button{font-size:16px;padding:8px 18px;margin:4px}#prog{color:#666}</style></head><body>
<h2>Which long text is better?</h2>
<p>Two texts were written by AI for the same task (shown below). Each text is about 25,000 words; you see three
excerpts of each (beginning, middle, end). Read both and answer the questions. Please judge the <b>whole text</b>,
especially whether it stays coherent and avoids repetition all the way to the end.</p>
<div id="prog"></div><div id="task" style="background:#eef;padding:8px"></div>
<div class="cols"><div class="col"><h3>Text A</h3><div id="A"></div></div><div class="col"><h3>Text B</h3><div id="B"></div></div></div>
<div class="q">1. Which text is better <b>overall</b>? <span id="q1"></span></div>
<div class="q">2. Which text stays more <b>coherent</b> through the end? <span id="q2"></span></div>
<div class="q">3. Which text is <b>less repetitive</b>? <span id="q3"></span></div>
<button onclick="next()">Next &rarr;</button>
<script>
const ITEMS = __ITEMS__;
let i = 0, ans = {}, out = [];
const wid = prompt("Please enter your worker ID") || "anon";
function opts(q){return ["A","B","Tie"].map(o=>`<label><input type=radio name=${q} value=${o} onclick="ans['${q}']='${o}'"> ${o}</label>`).join(" ")}
function exs(t){return ["beginning","middle","end"].map(k=>`<div class=lab>[${k}]</div><div class=ex>${t[k]}</div>`).join("")}
function show(){const it=ITEMS[i];document.getElementById("prog").textContent=`Item ${i+1} of ${ITEMS.length}`;
 document.getElementById("task").textContent="Task given to the AI: "+it.task;
 document.getElementById("A").innerHTML=exs(it.A);document.getElementById("B").innerHTML=exs(it.B);
 ["q1","q2","q3"].forEach(q=>document.getElementById(q).innerHTML=opts(q));ans={};window.scrollTo(0,0);}
function next(){if(!(ans.q1&&ans.q2&&ans.q3)){alert("Please answer all three questions.");return;}
 out.push({id:ITEMS[i].id, worker:wid, ...ans, t:Date.now()}); i++;
 if(i>=ITEMS.length){const b=new Blob([JSON.stringify(out)],{type:"application/json"});const a=document.createElement("a");
  a.href=URL.createObjectURL(b);a.download=`answers_${wid}.json`;a.click();document.body.innerHTML="<h2>Thank you! Your answers were downloaded; please upload the file.</h2>";return;}
 show();}
show();
</script></body></html>"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(a):
    from .tok import load_tokenizer
    rng = random.Random(0)
    cfgs = a.configs.split(",")
    items, key = [], {}
    from .judge import load_run
    for model in a.models.split(","):
        run = os.path.expanduser(f"~/lcg-cr/runs/{a.exp}/{model}")
        first = json.loads(open(f"{run}/meta.jsonl").readline())
        tok, items_ = load_run(run, first["model"])           # merges multi-turn / segmented parts
        job = json.load(open(os.path.expanduser(f"~/lcg-cr/jobs/{a.exp}/{model}.json")))
        task = {}
        for r in job["requests"]:
            p = r["prompt"]
            txt = p["messages"][-1]["content"] if p.get("type") == "chat" else "Continue the book."
            task[r["uid"]] = txt.split("\n\n")[1] if txt.count("\n\n") >= 1 else txt
        bypid = {}
        for u, m, toks, _ in items_:
            pid, c = u.split("__")[0], u.split("__")[1]
            if c in cfgs:
                bypid.setdefault(pid, {})[c] = (u, toks)
        def excerpts(toks):
            L = len(toks)
            return {"beginning": esc(tok.decode(toks[:EXC], skip_special_tokens=True)),
                    "middle": esc(tok.decode(toks[L // 2:L // 2 + EXC], skip_special_tokens=True)),
                    "end": esc(tok.decode(toks[L - EXC:], skip_special_tokens=True))}
        for pid, d in bypid.items():
            for c1, c2 in itertools.combinations(sorted(d), 2):
                if rng.random() > a.frac:
                    continue
                x, y = (c1, c2) if rng.random() < 0.5 else (c2, c1)
                iid = f"it{len(items)}"
                items.append({"id": iid, "task": task.get(d[x][0], ""), "A": excerpts(d[x][1]), "B": excerpts(d[y][1])})
                key[iid] = {"model": model, "pid": pid, "A": x, "B": y}
    # attention checks: pair a real text with a degenerate control (a repeated sentence)
    n_chk = max(1, len(items) // 10)
    for j in range(n_chk):
        base = rng.choice(items)
        bad = {k: esc(("The end. " * 120).strip()) for k in ("beginning", "middle", "end")}
        good = base["A"]
        x_is_good = rng.random() < 0.5
        iid = f"chk{j}"
        items.append({"id": iid, "task": base["task"], "A": good if x_is_good else bad, "B": bad if x_is_good else good})
        key[iid] = {"check": True, "good": "A" if x_is_good else "B"}
    rng.shuffle(items)
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/index.html", "w").write(HTML.replace("__ITEMS__", json.dumps(items)))
    json.dump(key, open(f"{a.out}/key.json", "w"), indent=1)
    open(f"{a.out}/README.md", "w").write(
        f"# Human evaluation kit\n\n{len(items)} items ({n_chk} attention checks). Serve index.html (any static host, "
        "or split items across workers). Each worker downloads answers_<id>.json. Keep key.json private.\n"
        "Analyse: python -m lcs.humaneval analyze --out <dir> --answers <dir-with-answer-files>\n")
    print(f"built {len(items)} items -> {a.out}")


def analyze(a):
    key = json.load(open(f"{a.out}/key.json"))
    ans = [x for f in glob.glob(f"{a.answers}/answers_*.json") for x in json.load(open(f))]
    bad_workers = {x["worker"] for x in ans if key[x["id"]].get("check") and x["q1"] != key[x["id"]]["good"]}
    wins = {}
    for x in ans:
        k = key[x["id"]]
        if k.get("check") or x["worker"] in bad_workers:
            continue
        for q in ("q1", "q2", "q3"):
            if x[q] == "Tie":
                continue
            w, l = (k["A"], k["B"]) if x[q] == "A" else (k["B"], k["A"])
            wins.setdefault(q, []).append((w, l))
    for q, pairs in wins.items():
        cfgs = sorted({c for p in pairs for c in p})
        s = {c: 0.0 for c in cfgs}
        for _ in range(200):  # Bradley-Terry (MM updates)
            new = {}
            for c in cfgs:
                w = sum(1 for p in pairs if p[0] == c)
                den = sum(1.0 / (np.exp(s[p[0]]) + np.exp(s[p[1]])) for p in pairs if c in p)
                new[c] = np.log(max(w, 1e-3) / max(den, 1e-9))
            m = np.mean(list(new.values()))
            s = {c: v - m for c, v in new.items()}
        print(q, "workers excluded:", len(bad_workers), {c: round(v, 2) for c, v in sorted(s.items(), key=lambda t: -t[1])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--exp", default="core32k")
    ap.add_argument("--models", default="")
    ap.add_argument("--configs", default="")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.expanduser("~/lcg-cr/humaneval"))
    ap.add_argument("--answers", default="")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else analyze(a)


if __name__ == "__main__":
    main()
