# Running findings (camera-ready campaign). Every number here traces to experiments_cr/results/.

## R1: draft's 10 models, 8 fiction prompts, forced 16K, seed 1 (interim, Qwen2.5-14B/32B still filling)
Source: results/cr_r1_16k.txt (judge = gpt-oss-120b, window overall 1-10; loop = terminal-loop flag).

1. Forced to 16K, NO sampler produces healthy text to the end. Pre-registered "healthy" rate is 0.00 in 21/22 cells
   (top-nsigma T=2.0: 0.03). Last-window judge score is ~1.0-2.6/10 everywhere.
2. At T=1.0 the samplers are indistinguishable on quality (last window 2.2-2.6, CIs overlap) and nearly all loop
   (top-p 0.96, min-p 0.99, P-less 0.99, top-nsigma 1.00, Top-H 0.97). The draft's claim that full-distribution-aware
   samplers avoid late-context collapse at default temperature is NOT supported.
3. Temperature changes the failure mode, and the sampler decides which one:
   - Rules whose pool grows with T (pure sampling, top-p, top-k, Top-H) trade loops for word salad:
     first-window judge at T=2.0 = 1.00 / 1.05 / 1.75 / 1.75.
   - top-nsigma (temperature-invariant) keeps first-window quality at T=2.0 (4.55, same as T=1.0 4.61) but still loops (0.72).
   - min-p and P-less sit in between (first window 3.04 / 2.89 at T=2.0).
4. Implication for the paper: "sampler choice" governs local quality under elevated temperature, not whether long
   forced generation survives. Matches the Oct-2026 ~/lcgen phase result and the rebuttal's agentic finding.

## Mechanism: realised candidate-set size (results/cr_pool_r1_16k.txt; first 2,048 tokens, ~73 generations/cell, 10 models)
median |C| at T = 1.0 / 1.5 / 2.0  (fraction of steps with a single candidate at T=1.0)
- P-less      1 / 1 / 31      (0.93)  -> P-less at T=1 is greedy decoding on 93% of steps; explains its 99% loop rate.
- top-nsigma  1 / 1 / 1       (0.54)  -> flat in T (per-step invariant by construction; realised means 3.4/4.1/4.6 differ only because trajectories differ).
- min-p 0.1   1 / 3 / 13      (0.60)
- Top-H 0.4   1 / 11 / 54     (0.62)  -> grows with T; NOT temperature-invariant, so the draft's "full-distribution-aware family" splits.
- top-p 0.9   2 / 37,654 / 83,518 (0.48) -> explodes; explains word salad at T>=1.5.
- top-k 50    51 / 51 / 51 (argmax always kept, so 51 when argmax is outside the top 50 ties).
Reading: the rule that matters is how |C| responds to temperature. Tight-at-T=1 rules (P-less, nsigma, min-p) are near-greedy
and loop; T-growing rules escape loops only by admitting the noise tail.

## R3 agentic coding, SWE-bench Verified, Qwen3.5-9B (interim: graded configs only; T=1.0 cells of R3a still running)
Resolved / instances (official harness):
- R3a, 100 instances, step limit 250, T=2.0: top-nsigma 51, P-less 50 | top-p 0, top-k 0, Top-H 0, pure sampling 0. Pure sampling at T=1.5: 0.
- R3b long-horizon (50 instances, 500 steps, 262K ctx):
  T=1.0: top-nsigma 26, Top-H 28, min-p 23, top-p 23, pure 23  (no separation at deployment temperature)
  T=1.5: top-nsigma 26, min-p 24, Top-H 9, top-p 0, pure 0
Reading: identical to the fiction mechanism. Temperature-growing rules break the agent completely at T>=1.5; rules whose
pool stays tight keep their T=1.0 resolve rate. At T=1.0 rules are indistinguishable. Paired tests pending full grading.

## BUG (found 2026-10-31): 5090 top-nsigma cells are invalid
code/samplers_fda.py (copied from ~/long-context-rebuttal) computed sigma with std() over logits containing -inf padding
-> NaN -> empty mask -> every step emits token 0 ("!!!!"). Affects ALL top-nsigma generations in E1 (Qwen3-4B, 84/84)
and E2 (Qwen3-8B, 42/42), and the rebuttal's long_nonfic top-nsigma arm (12/12). Fixed (finite-only std, argmax kept)
with a regression test. The H200 lcs/samplers.py already used a finite-only std; checked 240 H200 top-nsigma
generations, none affected. E1/E2 top-nsigma cells are excluded from the paper.

## R4 natural-length multi-turn (7 of 9 models judged; "Continue." turns of 4K, EOS allowed, 16K budget)
results/cr_r4_mt.txt. At T=1.0 loop rates drop vs forced length (top-p 0.56, top-k 0.29, pure 0.07, top-nsigma 0.51,
min-p 0.73, Top-H 0.73, P-less 0.88) and last-window quality rises to 3.5-3.9 for all; still no separation (CIs overlap).
At T=1.5-2.0 the same split as R1: top-p/top-k/Top-H/pure collapse to 1.0-2.0 first-window, top-nsigma holds 5.1-5.3.
P-less loops most at every T (0.84-0.88 at T<=1.5).

## E1 Qwen3-4B-2507 (5090, 14 prompts x 2 seeds, forced 16K), excluding invalid top-nsigma
First window high (7.6-8.0) for every rule at T=1.0, last window 1.5-2.0: same collapse. top-p first window 3.0 at T=1.5,
1.0 at T=2.0; min-p and P-less hold first-window 7.6-7.9 through T=2.0 on this model.
