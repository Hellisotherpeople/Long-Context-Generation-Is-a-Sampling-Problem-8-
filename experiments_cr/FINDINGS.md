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
