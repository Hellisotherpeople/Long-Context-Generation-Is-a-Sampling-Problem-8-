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

## Regenerating figures from the committed results
`fig_teaser`, `fig_swe`, `fig_swe_gemma` and `fig_map` only need `experiments_cr/results/`:
`cd experiments_cr/code && LCG_HOME=$PWD/.. python -c "from lcs import cr_figs as c; c.style(); c.fig_teaser(); c.fig_swe(); c.fig_swe('gemma'); c.fig_map()"`.
The other figures read the raw run directories (`~/lcg-cr/runs`). `fig_predict` (in `figs_cr2.py`) now normalizes sampler tags on
both sides of its merge. Before this fix, top-nσ with n = 1, 2, 3 was silently dropped (133 instead of 154 cells), so rerun it on the run host.
