"""Analyse SWE-bench Verified agent sweeps (mini-swe-agent trajectories + official harness reports).

python -m lcs.ana_swe -> ~/lcg-cr/results/swe_{inst,cells}.parquet
Per trajectory: resolved, exit status, steps, format errors, cut-off turns, repeated actions (an action identical to an
earlier one; longest run of identical consecutive actions), peak prompt length.  A trajectory "loops" when it repeats
the same action 3+ times in a row or more than half of its actions are repeats.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

H = os.path.expanduser("~/lcg-cr")
ACT = re.compile(r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```", re.S)


def traj_row(path):
    t = json.load(open(path))
    info = t.get("info", {})
    msgs = t.get("messages", [])
    acts, fmt, cut, ptoks, ctoks = [], 0, 0, [], []
    for m in msgs:
        if m.get("role") == "assistant":
            ex = m.get("extra", {})
            a = [x.get("command", "") for x in ex.get("actions", [])] or ACT.findall(m.get("content") or "")
            if len(a) == 1:
                acts.append(a[0].strip())
            r = ex.get("response") or {}
            u = r.get("usage") or {}
            if u:
                ptoks.append(u.get("prompt_tokens", 0)); ctoks.append(u.get("completion_tokens", 0))
            ch = (r.get("choices") or [{}])[0]
            cut += ch.get("finish_reason") == "length"
        elif m.get("role") == "user" and isinstance(m.get("content"), str):
            c = m["content"]
            fmt += ("Format error" in c) or ("reached the output token limit" in c)
    seen, rep, run, best = set(), 0, 1, 1 if acts else 0
    for i, a in enumerate(acts):
        rep += a in seen
        seen.add(a)
        if i and a == acts[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    n = len(acts)
    return {"exit": info.get("exit_status"), "patch_len": len(info.get("submission") or ""),
            "steps": info.get("model_stats", {}).get("api_calls", len(ptoks)), "n_actions": n,
            "format_errors": fmt, "cut_turns": cut, "repeat_frac": rep / n if n else np.nan, "max_consec": best,
            "loop": bool(best >= 3 or (n >= 10 and rep / n > 0.5)),
            "peak_prompt": max(ptoks) if ptoks else np.nan, "out_tokens": int(sum(ctoks)),
            "mean_turn_tokens": float(np.mean(ctoks)) if ctoks else np.nan}


def main():
    rows = []
    for d in sorted(glob.glob(f"{H}/runs/*swe*/*/*/")):
        d = d.rstrip("/")
        exp, tag, cfg = d.split("/")[-3:]
        if tag.startswith("_") or not os.path.exists(f"{d}/preds.json"):
            continue
        rf = f"{d}/report.json" if os.path.exists(f"{d}/report.json") else f"{d}/report_partial.json"
        rep = json.load(open(rf)) if os.path.exists(rf) else None
        resolved = set(rep.get("resolved_ids", [])) if rep else set()
        # every graded prediction counts: empty patches, patches that do not apply and test timeouts are unresolved
        evaluated = (set(rep.get("submitted_ids", [])) | set(rep.get("empty_patch_ids", []))) if rep else set()  # SWE-bench convention: harness errors and empty patches count as unresolved
        preds = json.load(open(f"{d}/preds.json"))
        for iid in preds:  # every instance with a prediction counts, with or without a saved trajectory
            p = f"{d}/{iid}/{iid}.traj.json"
            r = traj_row(p) if os.path.exists(p) else {"exit": "NoTrajectory", "patch_len": len(preds[iid].get("model_patch") or ""),
                                                       "steps": np.nan, "n_actions": 0, "format_errors": 0, "cut_turns": 0,
                                                       "repeat_frac": np.nan, "max_consec": 0, "loop": False,
                                                       "peak_prompt": np.nan, "out_tokens": 0, "mean_turn_tokens": np.nan}
            r.update(exp=exp, tag=tag, model=tag.split("_c")[0].replace("_think", ""), think=tag.endswith("_think"), config=cfg, instance=iid,
                     resolved=(iid in resolved) if iid in evaluated else np.nan)
            rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(f"{H}/results", exist_ok=True)
    df.to_parquet(f"{H}/results/swe_inst.parquet"); df = df[df.exp != "swe"]
    df["submitted"] = (df.exit == "Submitted") & (df.patch_len > 0)
    df["ctx_overflow"] = df.exit.fillna("").str.contains("ContextWindow|context", case=False)
    df["step_limit"] = df.exit == "LimitsExceeded"
    cells = df.groupby(["exp", "model", "think", "config"]).agg(
        n=("instance", "size"), resolved=("resolved", "mean"), submitted=("submitted", "mean"),
        loop=("loop", "mean"), step_limit=("step_limit", "mean"), ctx_overflow=("ctx_overflow", "mean"),
        format_err=("format_errors", "mean"), cut_turns=("cut_turns", "mean"), steps=("steps", "mean"),
        repeat_frac=("repeat_frac", "mean"), peak_prompt=("peak_prompt", "mean"),
        turn_tokens=("mean_turn_tokens", "mean")).reset_index()
    cells.to_parquet(f"{H}/results/swe_cells.parquet")
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(cells.round(3).to_string())


if __name__ == "__main__":
    main()
