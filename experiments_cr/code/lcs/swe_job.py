"""SWE-bench Verified agent sweep for one model (scheduler "py" job, vLLM venv).

Starts a vLLM OpenAI server with the LCS logits processor on the job's GPUs, runs lcs.swe_agent (agent venv) for every
sampler configuration (several configurations concurrently, all sharing the server), then stops the server.
Resumable: mini-swe-agent skips instances already in <out>/<config>/preds.json.

python -m lcs.swe_job <job.json>
job.json: {"model": dir, "served": name, "tp": 1, "max_model_len": 131072, "engine_args": [...],
           "chat_kwargs": {}, "ids": file|"", "configs": [[name, T, ops(, extra)], ...], "parallel_configs": 4,
           "workers": 16, "step_limit": 150, "max_tokens": 8192, "out": dir}
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

H = os.path.expanduser("~/lcg-cr")
VPY = "/home/ubuntu/glm53-vllm/.venv/bin/python"
APY = f"{H}/agent/bin/python"


def wait_ready(port, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            return
        except Exception:
            time.sleep(10)
    raise RuntimeError("vLLM server not ready")


def n_preds(d):
    try:
        return sum(1 for v in json.load(open(f"{d}/preds.json")).values() if v is not None)
    except Exception:
        return 0


def main(jf):
    J = json.load(open(jf))
    n_target = len([l for l in open(J["ids"]) if l.strip()]) if J.get("ids") else 500
    incomplete = []
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    port = 8100 + int(gpus[0])
    out = J["out"]
    os.makedirs(out, exist_ok=True)
    try:  # a stale server on our port would silently serve another job's agents
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        raise RuntimeError(f"port {port} already serving")
    except OSError:
        pass
    env = dict(os.environ, PYTHONPATH=f"{H}/code", LCS_STATS_DIR=f"{out}/_stats", VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
               PATH=os.path.dirname(VPY) + ":" + os.environ.get("PATH", ""))
    cmd = [VPY, "-m", "vllm.entrypoints.openai.api_server", "--model", J["model"], "--served-model-name", J["served"],
           "--port", str(port), "--max-model-len", str(J.get("max_model_len", 131072)),
           "--tensor-parallel-size", str(J.get("tp", 1)), "--gpu-memory-utilization", str(J.get("gpu_mem", 0.90)),
           "--enable-prefix-caching", "--generation-config", "vllm", "--trust-remote-code",
           "--max-num-seqs", str(J.get("max_num_seqs", 256)),
           "--logits-processors", "lcs.processor:LCSProcessor"] + J.get("engine_args", [])
    srv_log = open(f"{out}/server.log", "a")
    # same process group as this job (the scheduler kills jobs by group); SIGTERM -> SystemExit so `finally` runs
    srv = subprocess.Popen(cmd, env=env, stdout=srv_log, stderr=subprocess.STDOUT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    running = {}
    try:
        wait_ready(port, srv)
        print("server ready on", port, flush=True)
        todo = list(J["configs"])
        while todo or running:
            while todo and len(running) < J.get("parallel_configs", 4):
                name, T, ops, *extra = todo.pop(0)
                d = f"{out}/{name}"
                os.makedirs(d, exist_ok=True)
                a = [APY, "-m", "lcs.swe_agent", "--api-base", f"http://localhost:{port}/v1", "--served", J["served"],
                     "--name", name, "--T", str(T), "--ops", json.dumps(ops), "--out", d,
                     "--extra", json.dumps(extra[0] if extra else {}),
                     "--chat-kwargs", json.dumps(J.get("chat_kwargs", {})), "--workers", str(J.get("workers", 16)),
                     "--step-limit", str(J.get("step_limit", 150)), "--max-tokens", str(J.get("max_tokens", 8192))]
                if J.get("ids"):
                    a += ["--ids", J["ids"]]
                lf = open(f"{d}/run.log", "a")
                running[name] = subprocess.Popen(a, env=dict(env, TERM="dumb"), stdout=lf, stderr=subprocess.STDOUT,
                                                 cwd=f"{H}/code")
                print("start", name, flush=True)
            time.sleep(30)
            if srv.poll() is not None:
                raise RuntimeError(f"vLLM server died rc={srv.returncode}")
            for name, p in list(running.items()):
                if p.poll() is not None:
                    print("done", name, "rc", p.returncode, flush=True)
                    del running[name]
                    got = n_preds(f"{out}/{name}")
                    if p.returncode == 0 and got >= n_target:
                        open(f"{out}/{name}/.agent_done", "w").close()
                    else:
                        print("INCOMPLETE", name, got, "/", n_target, flush=True)
                        incomplete.append(name)
    finally:
        for p in list(running.values()) + [srv]:
            if p.poll() is None:
                p.terminate()
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()
    if incomplete:
        sys.exit(f"incomplete configs: {incomplete}")


if __name__ == "__main__":
    main(sys.argv[1])
