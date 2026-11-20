# Same Bytes, Different Failures

UCS532 sessional, problem **P49: Compressing a Vision Transformer to 10 MB**. Manan Kapoor.

DeiT-Small (88.24 MB, 79.72% ImageNet top-1) compressed four ways to the same measured file size
(9.72–9.80 MB), each with the same 3000-step recovery budget and three seeds. Question: at equal
bytes, do the routes fail on the same images?

## Results

| Route | What it does | File | Clean top-1 | Damaged (mean of 14) | CPU, x86 / M2 |
|---|---|---|---|---|---|
| — | DeiT-S, uncompressed | 88.24 MB | 79.72 | 62.64 | 71.4 / 17.8 ms |
| A | 3-bit, one scale per 36 weights | 9.718 MB | 76.50 | 55.04 | 71.4 / 17.8 ms |
| B | ternary (−1, 0, +1), 5 weights per byte; 15 layers at 8-bit | 9.785 MB | 72.43 | 44.35 | 71.4 / 17.8 ms |
| C | 4-bit + 24.375% of MLP channels pruned | 9.796 MB | 75.70 | 53.55 | 64.1 / 16.1 ms |
| D | DeiT-Tiny distilled from DeiT-S, 8-bit; 38 layers at 16-bit | 9.765 MB | 71.41 | 48.16 | 29.9 / 8.0 ms |

Accuracy is the mean of 3 seeds. CPU time: fp32, 4 threads, batch 1, median of 50 runs.

1. **Accuracy loss of the final model (A): 3.22 points**, measured in the same pipeline as the baseline.
2. **Same size, different failures.** Routes disagree 1.2–2.0× more than two seeds of one route do.
3. **Ternary is hit hardest by noise:** 17.7–20.2 points below 3-bit on noise, 5.9–6.4 on digital damage.
4. **The clean ranking flips.** B beats D on clean images in all 3 seeds (+0.9 to +1.1, McNemar
   p ≤ 3.3e-08); D beats B on damaged images in all 3 (+3.7 to +3.9, 95% CI ≈ [+3.2, +4.4]).
   D is a different architecture, so this is a deployment result, not a law.
5. **A smaller file is not a faster model.** A and B run at the uncompressed model's speed; pruning (C) gains ~10%;
   only D is much faster (2.2× on M2, 2.4× on x86).
6. **The `params × bits ÷ 8` size formula** is fair for per-channel quantizers on large models
   (+2–5%) but understates real files at small group sizes and small models: +29.1% for 2-bit
   DeiT-S, +53.5% for DeiT-Tiny.

**Report (CVPR format): [`report/main.pdf`](report/main.pdf).** Every number and how it was checked: [`results.md`](results.md).

## Notebooks

Each notebook runs the real code on the saved evidence and is stored with its outputs, so it can be read on
GitHub without running anything. To re-run: `.venv/bin/pip install nbclient ipykernel`, then open in Jupyter.

| Notebook | What it shows |
|---|---|
| [`01_file_size`](notebooks/01_file_size.ipynb) | Formula vs real file size; every size rebuilt from network shapes (5 networks) |
| [`02_the_four_routes`](notebooks/02_the_four_routes.ipynb) | Quantisation on real numbers; the four route files; decoding them back into models |
| [`03_results`](notebooks/03_results.ipynb) | Main table; seed floor vs route disagreement; the clean/damaged ranking flip; damage families |
| [`04_calibration_and_trend`](notebooks/04_calibration_and_trend.ipynb) | Calibration before/after temperature scaling; the ternary trend-line test |
| [`05_cpu_latency`](notebooks/05_cpu_latency.ipynb) | CPU time per photo on Apple M2 and Intel; is a smaller file faster? |
| [`06_how_the_models_were_trained`](notebooks/06_how_the_models_were_trained.ipynb) | Training recipe, training curves, the Kaggle re-check (99.71% match) |

## Layout

```
results.md         every number, its source, and how it was checked
reproduce.sh       re-derives and checks every statistic and table (6 checks, ~2 min, no GPU)
report/            CVPR-format report (main.tex, main.pdf; build: tectonic -X compile main.tex)
notebooks/         six executed notebooks that print the real source code
proposal/          UCS532 proposal (HTML source + PDF)
src/
  phase1_audit.py       real file sizes vs the formula
  build_routes.py       build each route to an exact byte target
  train_routes.py       recovery training, 3000 steps (GPU)
  evaluate_routes.py    clean + corrupted accuracy, ECE (GPU)
  stats.py              McNemar, bootstrap, Bonferroni
  offline_analysis.py   seed floor, per-corruption, calibration
  decode_bin.py         load a .bin back into a model
  rescore.py            re-score every model from its .bin (Kaggle T4)
  o7_trend.py           the ternary trend-line test (O7)
  rescore_check.py      do the model files reproduce the original run (99.71%)
  robustness_source.py  where route D's robustness comes from
  make_figures.py       draws every figure from saved results
  cpu_latency.py        CPU latency per route
  main_table.py         the result tables above
evidence/
  models/               the 21 compressed model files (.bin)
  logs/                 training/eval logs with per-image right/wrong arrays
  rescore/              Kaggle re-score, x86 latency, main tables
  audit_deit/, audit_multiarch/, analysis/, latency/
docs/                   code walkthrough, plan, notes; history/ keeps superseded versions
```

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./reproduce.sh
```

Training and evaluation ran on an H100 slice (until 2026-09-25); the re-score ran on a Kaggle T4
(`docs/GPU_RUN.md`). Their outputs are in `evidence/`.
