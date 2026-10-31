"""One mini-swe-agent SWE-bench Verified batch run under one sampler configuration (runs in the agent venv).

Text-based (backticks) mini-swe-agent config, so every token of every action is sampled by our logits processor and
no model-specific tool-call parser is involved.  Epoch AI images keep the repository history after the base commit;
tools/pull_swe_images.sh prunes it once per image (marker /LCS_PRUNED); unpruned images are pruned here at start.

python -m lcs.swe_agent --api-base http://localhost:8100/v1 --served m --name minp0.1@T1.0 --T 1.0 \
    --ops '[["min_p",0.1]]' --out ~/lcg-cr/runs/swe/<model>/<name> [--chat-kwargs '{}'] [--workers 16] [--ids file]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PRUNE = r"""cd /testbed && b=$(git symbolic-ref -q --short HEAD || true)
for r in $(git for-each-ref --format='%(refname)'); do [ "$r" = "refs/heads/$b" ] || git update-ref -d "$r"; done
for r in $(git remote); do git remote remove "$r"; done
git stash clear 2>/dev/null; git reflog expire --expire=now --all && git gc -q --prune=now && echo LCS_PRUNED $(git rev-list --all | wc -l)"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--ops", default="[]")
    ap.add_argument("--extra", default="{}", help="merged into the lcs config, e.g. {\"pen\": {\"presence\": 1.5}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat-kwargs", default="{}")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ids", default="")
    ap.add_argument("--step-limit", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()

    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    import minisweagent.run.benchmarks.swebench as sb
    import yaml
    from minisweagent.config import builtin_config_dir

    orig = sb.get_sb_environment

    def get_env(config, instance):
        env = orig(config, instance)
        out = env.execute({"command": "test -f /LCS_PRUNED && echo LCS_PRUNED || { " + PRUNE + "; }"}, timeout=900)
        if out.get("returncode") != 0 or "LCS_PRUNED" not in out.get("output", ""):
            raise RuntimeError(f"git prune failed: {out}")
        return env

    sb.get_sb_environment = get_env

    lcs = {"tag": f"swe_{a.name}", "T": a.T, "ops": json.loads(a.ops), "nolog": True, **json.loads(a.extra)}
    cfg = yaml.safe_load(open(builtin_config_dir / "benchmarks" / "swebench_backticks.yaml"))
    # resource limits per agent sandbox: agents run arbitrary commands, and one runaway process (fork bomb, memory
    # explosion) must not take down the host that serves every other job
    cfg["environment"]["run_args"] = ["--rm", "--memory=16g", "--memory-swap=16g", "--pids-limit=4096", "--cpus=8"]
    # the 60 s command timeout only kills the docker client; kill the command (and its children) inside the container too
    cfg["environment"]["interpreter"] = ["timeout", "-k", "5", "65", "bash", "-c"]
    cfg["agent"]["step_limit"] = a.step_limit
    cfg["agent"]["cost_limit"] = 0  # local model: no cost limit
    cfg["model"].update(model_name=f"hosted_vllm/{a.served}", model_class="litellm_textbased", cost_tracking="ignore_errors",
                        # also accept the conventional ```bash fence: under sampling models sometimes write it, and we
                        # measure long-context behaviour, not adherence to the custom fence name
                        action_regex=r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```")
    cfg["model"]["model_kwargs"] = {
        "api_base": a.api_base, "api_key": "EMPTY", "drop_params": True, "timeout": 3600,
        "temperature": 1.0, "top_p": 1.0, "max_tokens": a.max_tokens,
        "extra_body": {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                       "chat_template_kwargs": json.loads(a.chat_kwargs),
                       "vllm_xargs": {"lcs_json": json.dumps(lcs)}},
    }
    os.makedirs(a.out, exist_ok=True)
    cf = os.path.join(a.out, "config.yaml")
    yaml.safe_dump(cfg, open(cf, "w"), sort_keys=False)
    argv = ["swebench", "--subset", "verified", "--split", "test", "-o", a.out, "-w", str(a.workers), "-c", cf]
    if a.ids:
        ids = [l.strip() for l in open(a.ids) if l.strip()]
        argv += ["--filter", "^(" + "|".join(i.replace(".", r"\.") for i in ids) + ")$"]
    sys.argv = argv
    sb.app()


if __name__ == "__main__":
    main()
