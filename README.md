# Same Bytes, Different Failures

UCS532 sessional, problem **P49: Compressing a Vision Transformer to 10 MB**. Manan Kapoor.

DeiT-Small (88.24 MB, 79.72% ImageNet top-1) compressed four ways to the same measured file size
(9.72–9.80 MB), each with the same 3000-step recovery budget and three seeds. Question: at equal
bytes, do the routes fail on the same images?

**Reproduce the main table (one command, CPU only, about 2 min):**

```bash
bash run_all.sh
```

It creates `.venv`, installs `requirements.txt`, downloads the 21 model files from the
[`models-v1` release](https://github.com/manankapoor23/vit-compression/releases/tag/models-v1) and checks
their SHA-256, then re-derives every statistic and table from the saved per-image results and checks each
against the archived output (8 checks), and prints Table 1. No dataset or GPU is needed.

## Results

| Route | What it does | File | Clean top-1 | Damaged (mean of 14) | CPU, x86 / M2 |
|---|---|---|---|---|---|
| — | DeiT-S, uncompressed | 88.24 MB | 79.72 | 62.64 | 71.4 / 17.8 ms |
| A | 3-bit, one scale per 36 weights | 9.718 MB | 76.50 | 55.04 | 71.4 / 17.8 ms |
| B | ternary (−1, 0, +1), 5 weights per byte; 15 layers at 8-bit | 9.785 MB | 72.43 | 44.35 | 71.4 / 17.8 ms |
| C | 4-bit + 374 of 1,536 MLP channels pruned (24.35%) | 9.796 MB | 75.70 | 53.55 | 64.1 / 16.1 ms |
| D | DeiT-Tiny distilled from DeiT-S; 12 of 50 weight matrices 8-bit, 38 at 16-bit | 9.765 MB | 71.41 | 48.16 | 29.9 / 8.0 ms |

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
run_all.sh         one command: set up, download models, run reproduce.sh, print the main table
reproduce.sh       re-derives and checks every statistic and table (8 checks, ~2 min, no GPU)
report/            CVPR-format report (main.tex, main.pdf; build: tectonic -X compile main.tex)
notebooks/         six executed notebooks that print the real source code
proposal/          UCS532 proposal (HTML source + PDF)
slides/            10-slide presentation (PPTX)
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
  bd_flip.py            ternary vs distilled Tiny: McNemar and Bonferroni intervals
  mixqvit_size.py       Mix-QViT's 11.4 MB rebuilt in their own layout
  make_figures.py       draws every figure from saved results
  cpu_latency.py        CPU latency per route
  main_table.py         the result tables above
evidence/
  models/               SHA256SUMS of the 21 model files (the .bin files are in the models-v1 release)
  logs/                 training/eval logs with per-image right/wrong arrays
  rescore/              Kaggle re-score, x86 latency, main tables
  audit_deit/, audit_multiarch/, analysis/, latency/
docs/                   code walkthrough, plan, notes; history/ keeps superseded versions
```

## Dataset

- **ImageNet-1k (ILSVRC 2012)**, WebDataset version
  [`timm/imagenet-1k-wds`](https://huggingface.co/datasets/timm/imagenet-1k-wds) (access requires accepting
  the ImageNet terms). Not included in this repository.
- **Training subset:** the first 200 of the 1,024 training shards in sorted order (about 20% of the training
  set; `src/train_routes.py`, `loaders`). Every route used the same 200 shards.
- **Clean test:** all 50,000 validation images.
- **Damaged test:** 14 of the 15 ImageNet-C corruptions at severity 3, generated on the fly from the first
  5,000 validation images with [`imagecorruptions`](https://github.com/bethgelab/imagecorruptions)
  (glass blur excluded: 865 s per 1,000 images, against 1–67 s for the others). Every model saw identical damaged images.
- **Seeds:** 0, 1 and 2 for every route, fixed in `src/train_routes.py` (lines 209–210) and printed at the
  top of each training log in `evidence/logs/`.

## Hardware and runtime

| Step | Hardware | Time |
|---|---|---|
| Training, one 3000-step run | NVIDIA H100 80GB, MIG 3g.40gb slice | about 25 min (route A log: 1.9–2.4 steps/s) |
| Re-score of all 21 model files + x86 CPU timing | Kaggle, NVIDIA T4 + Intel Xeon @ 2.00GHz | 51 min (`evidence/rescore/kaggle_run.log`) |
| CPU latency | Apple M2 and Intel Xeon, 4 threads, batch 1, median of 50 runs | not recorded separately |
| `bash run_all.sh` | Apple M2 laptop, CPU only | about 2 min (tested from a fresh clone, 23 Nov 2026) |

Training and evaluation need ImageNet and a GPU and are not re-run by `run_all.sh`; their logs and per-image
results are in `evidence/`. The GPU commands are in `docs/GPU_RUN.md`.

## Credits

| Reused | Source | Used for |
|---|---|---|
| DeiT-Small and DeiT-Tiny pretrained weights | Touvron et al., ICML 2021; [facebookresearch/deit](https://github.com/facebookresearch/deit), loaded through timm | starting point of every route; distillation teacher |
| timm | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (Apache-2.0) | model definitions and weights (DeiT, Swin-Tiny, ViT-Large for the size audit) |
| ImageNet-1k | Deng et al., 2009; [image-net.org](https://www.image-net.org/); shards from `timm/imagenet-1k-wds` | training subset and test set |
| ImageNet-C corruptions | Hendrycks and Dietterich, ICLR 2019; [`imagecorruptions`](https://github.com/bethgelab/imagecorruptions) (Michaelis et al., 2019) | damaged test images |
| Ternary threshold rule | Li et al., Ternary Weight Networks, 2016 | route B (threshold 0.7 × mean \|w\|) |
| Equal fine-tuning budget as the control | Kuzmin et al., NeurIPS 2023 | experimental design |
| PyTorch, webdataset, NumPy, SciPy | open-source libraries | training, data loading, statistics |
| CVPR LaTeX template | [cvpr-org/author-kit](https://github.com/cvpr-org/author-kit) | report formatting |

All other code in this repository was written for this project (see the AI declaration below).

## AI declaration

AI assistants (Claude, via Claude Code) were used to develop and debug code for model-size auditing,
compression-route construction, training, evaluation, statistical analysis, and figure generation; to assist
in drafting and editing the proposal and final report; and to cross-check numerical results and references
against experimental evidence and primary literature. All code in `src/`, `reproduce.sh`, `run_all.sh` and
the notebooks was generated with AI assistance. AI-assisted code and analyses were executed and checked against
archived experimental outputs, including through `reproduce.sh`. I take responsibility for the final submitted
work, including its experimental results, interpretations, limitations, and conclusions.

Solo project: Manan Kapoor (Roll No. 1024030467). Contribution statement: see the end of
[`report/main.pdf`](report/main.pdf).
