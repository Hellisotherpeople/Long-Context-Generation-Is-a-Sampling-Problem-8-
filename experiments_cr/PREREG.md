# Camera-ready pre-registration (written before any wave output is read)

All waves: lcs processor (temperature first, then truncation; raw argmax always kept), vLLM 0.30.0, bf16 on H200.
Samplers (SLIM): temp, top-k 50, top-p 0.9, min-p 0.1, top-nsigma n=1, P-less, Top-H a=0.4, plus greedy.
Fresh seed (1) so nothing overlaps the Oct-2026 seed-0 campaign in ~/lcgen.

## R1 (2026-10-31): replication on the draft's 10 models
8 fiction prompts (1/genre), T in {1.0,1.5,2.0}, forced length 16K (32K extra for gpt-oss-120b; Qwen2.5 capped by ctx).
Primary: Q4 loop rate (lcs.metrics loop flag) and judged window quality by depth (same judge as ~/lcgen).
Lexical metrics are diagnostics only.

## R2 (2026-10-31): non-fiction transfer
6 non-fiction prompts, 6 models, T in {1.0,1.5}, forced 16K. Same outcomes as R1.

## R3 (2026-10-31): agentic coding, SWE-bench Verified (mini-swe-agent harness from ~/lcgen, docker eval)
Fixed random subsets (seed 20261031): ids100 / ids50 in data/swe.
R3a  Qwen3.5-9B, 7 samplers x T{1.0,1.5,2.0}, 100 instances, step limit 250.
R3b  long-horizon variant: 6 samplers x T{1.0,1.5}, 50 instances, step limit 500, 262K ctx.
R3c  scale variant: Gemma-4-31B-it, 7 samplers x T{1.0,1.5}, 50 instances.
Primary: resolve rate (paired by instance; McNemar / sign test vs top-p 0.9 at same T). Secondary: loop, format error, steps.
