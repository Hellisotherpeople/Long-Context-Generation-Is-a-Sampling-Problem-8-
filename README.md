# Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat)

Code, results and LaTeX source for the NeurIPS 2026 Evaluations & Datasets Track paper by Allen G. Roush, Nguyen Nhat Minh, Ravid Shwartz-Ziv, Judah Goldfeder and Sanjay Basu.

- Paper: [OpenReview](https://openreview.net/forum?id=Gyng0YSjXM) · [NeurIPS poster page](https://neurips.cc/virtual/2026/poster/139582)

## Repository layout

- [`experiments_cr/code/`](experiments_cr/code/): batched truncation samplers for vLLM with per-step instrumentation, the experiment drivers, scoring, metrics and figure code. **Start with [`experiments_cr/code/README.md`](experiments_cr/code/README.md)** for the sampler list, how requests opt in, and how to regenerate figures.
- [`experiments_cr/results/`](experiments_cr/results/): the result files (CSV/JSON/TXT) behind the paper's tables and figures.
- [`experiments_cr/figs/`](experiments_cr/figs/), [`experiments_cr/tables/`](experiments_cr/tables/): generated figures and tables.
- [`Long_Context/`](Long_Context/): LaTeX source of the paper (`main.tex`, `refs.bib`, `checklist.tex`).

## License

All code and data in this repository are released under the [MIT License](LICENSE).
