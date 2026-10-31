# Truncation sampling at length — code

Batched truncation samplers with per-step instrumentation for vLLM (v1, 0.30), plus the experiment drivers,
metrics and figure code used in the paper.

## Samplers (`lcs/samplers.py`, `lcs/processor.py`, `lcs/dry.py`)
temperature, greedy, top-k, top-p, min-p, typical, eta, epsilon, top-nσ, P-less, P-less-norm, Top-H, min-k,
`keep_top` (absorption-free union with the top-k tokens), presence/frequency/repetition penalties, DRY.
Temperature is applied first, then truncation; the raw argmax is always kept. Rules are verified against naive
per-row reference implementations (`tools/test_reference.py`, `tools/test_dry.py`, `tools/test_compiled.py`).

Requests opt in via `SamplingParams(extra_args={"lcs": {"tag": uid, "T": 1.3, "ops": [["top_h", 0.4]], ...}})` with native
vLLM sampling neutral (temperature=1, top_p=1, top_k=0, min_p=0). Per generated position the processor logs:
entropy of the raw and tempered distribution, raw top probability, candidate-set size and retained mass, entropy of the
final sampling distribution, end pressure (probability of the banned stop tokens), and the rank / log-probability of the
sampled token (plus a check that it lies inside the candidate set).

## Drivers
- `lcs/gen.py JOB.json` — resumable generation with KV-aware admission control (no preemption), forced length,
  segmented million-token chains (`chain`), and multi-turn continuation (`multiturn`).
- `lcs/sched.py` — restart-safe GPU job queue.
- `lcs/make_jobs.py`, `lcs/exp_*.py` — experiment definitions (core grid, penalties/DRY/min-k, keep_top, multi-turn,
  loop-escape, book continuation, depth, long generation, NIAH to 1M, reasoning).
- `lcs/score.py`, `lcs/evalppl.py`, `lcs/judge.py`, `lcs/humaneval.py` — scoring, evaluator NLL, LLM judge,
  human-evaluation kit.
- `lcs/metrics.py`, `lcs/analyze.py`, `lcs/ana_*.py`, `lcs/figs.py` — metrics and figures.
