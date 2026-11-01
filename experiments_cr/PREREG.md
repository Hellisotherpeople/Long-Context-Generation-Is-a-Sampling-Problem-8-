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

## R4 (2026-10-31): natural-length multi-turn continuation
9 paper models (gpt-oss excluded), 8 fiction prompts, T{1.0,1.5,2.0}, 4K-token turns with "Continue." until 16K total, EOS allowed.
Primary: judged window quality by depth and loop rate; natural stop count reported.

## R5 (2026-10-31): healthy-region map
4 models (Llama-1B, Qwen2.5-3B, Llama-8B, Qwen3.5-9B), 22 sampler settings x T{0.7..3.0} (7 values), forced 8K.
Pre-registered definition: a cell is "healthy" if no terminal loop AND judge window coherence >= 6 at every window.
Primary quantity: width of the healthy temperature interval per sampler setting.

## R6 (2026-10-31): long-CoT agent-free coding, LiveCodeBench v6 (175 problems), thinking mode, 32K budget
New models vs Oct campaign: Qwen3.6-27B, Qwen3.8-27B; 2 samples; exp_reason2 sampler grid. Primary: pass@1 (official tests).

## R6 (2026-10-31): long-CoT coding, LiveCodeBench v6 (175 problems), thinking mode, 32K budget
New models vs Oct campaign: Qwen3.6-27B, Qwen3.8-27B; 2 samples; exp_reason2 sampler grid. Primary: pass@1 (official tests, lcs.ana_lcb).

### Amendment 2026-10-31 (after R5 outputs were read)
The pre-registered R5 healthy-rate map is reported as primary. A second, post-hoc summary is added and labelled as such:
the set of temperatures where mean judged window quality >= 4.0. Chosen after seeing the data, so it is descriptive only.

## R7 (2026-10-31): agentic scale replication. Long-horizon SWE grid (50 instances, 500 steps, 262K) on Qwen3.5-27B (dense) and
Qwen3.5-35B-A3B (MoE), 7 samplers x T{1.0,1.5,2.0}. Primary: paired sign test vs top-p 0.9 at same T.
## R8 (2026-10-31): mechanism ablation. Truncation computed at T=1, sampling at t in {1.5,2.0,3.0} ("temperature last") for top-p,
Top-H, min-p, P-less, top-k; 4 models, 8 fiction prompts, forced 8K. Prediction (stated before running): freezing the candidate set
removes the T cliff for top-p/Top-H, i.e. judged window quality at t=2.0 stays within 1 point of T=1.0, unlike R5.
## R9 (2026-10-31): seed-2 replication of R1 on 4 models (Llama-1B, Qwen2.5-3B, Mistral-7B, Llama-8B).
## R9b (2026-10-31): seed-2 replication of R1 extended to the other 6 paper models (same grid).
## R5b (2026-10-31): R5 map replicated on 4 further families (Gemma-4-12B, OLMo-3-7B, Ministral-3-8B, Nemotron-3-4B), same grid and outcomes; written before any R5b output.
### R5b amendment: Ministral-3-8B failed to load in vLLM 0.30 (Mistral3ForConditionalGeneration inspect error); replaced by Qwen2.5-14B before any R5b output was read.
## R7b (2026-10-31): long-horizon SWE grid on Gemma-4-E4B-it, T{1.0,1.5}, same outcomes.
## R13 (2026-10-31): judge validation. R1 window task re-judged by GLM-4.5-Air (different family). Primary: Spearman rho with gpt-oss-120b on window overall, and whether the R1 sampler ordering by first/last window is preserved.

## R14 (2026-10-31): established long-generation benchmarks (replacing reliance on bespoke prompts, at the user request).
LongBench-Write (Bai et al., LongWriter, ICLR 2025; 60 EN prompts, 100-20K words required) on 8 models incl. 70B/120B/122B,
and WritingBench (Wu et al., 2025; 555 EN queries) on 4 models. 7 SLIM samplers + greedy, T{1.0,1.5,2.0}, natural stopping.
Primary: official LongBench-Write S_l and S_q (judge = gpt-oss-120b with the official rubric) and WritingBench critic score.
Prediction (stated before running): no significant separation at T=1; at T=2 tail-admitting rules drop far more than top-nsigma.
### R12 amendment: Llama-3.1-8B D=131072 exceeds its 131,072-token window once 8K generation is added (CUDA assert); that cell is dropped. Max depth for Llama-3.1-8B is 64K.
## R12b (2026-11-01): context depth on Gemma-4-12B and Qwen3.5-27B, D{2K,16K,64K}, same grid/outcomes as R12.
## R15b (2026-11-01): Arena-Write with 2 samples on 6 models (adds Mistral-Nemo-12B, Nemotron-3-30B-A3B, OLMo-3-7B); same outcomes as R15.
## R18 (2026-11-01): fresh 100 SWE-bench Verified instances (disjoint from ids100, seed 20261101), Qwen3.5-9B, 250 steps, top-p/top-nsigma/min-p/Top-H at T{1,2}. Out-of-sample replication of the R3a T=2 split. Primary: paired sign test vs top-p at same T.
### WritingBench critic amendment (2026-11-01): responses longer than 28K tokens are truncated to first+last 14K tokens so the critic prompt fits its 32K context (one Gemma-4-31B response crashed the run).
