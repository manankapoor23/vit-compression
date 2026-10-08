# Same Bytes, Different Failures

**Four routes to a sub-10 MB Vision Transformer, and what each one costs.**
UCS532 sessional project, problem **P49 — Compressing a Vision Transformer to 10 MB** · Manan Kapoor

---

## The project in 30 seconds

DeiT-Small is an image classifier (a Vision Transformer) that takes **88.20 MB** on disk. We
squeezed it under **10 MB** four different ways, made all four files the **same real size**
(9.72–9.80 MB), trained each one for the **same 3000 steps**, three times each with different
random seeds, and asked:

> If four compressed models are the same size, do they fail on the same images?

**No.** And separately: the file size that ViT compression papers print is computed with a formula
that is wrong by 1.5% to 53.5%.

---

## The five results to remember

| # | Result | Number to quote |
|---|---|---|
| 1 | **The size formula is wrong** | `params × bits ÷ 8` understates a 2-bit DeiT-S file by **29.1%**, DeiT-Tiny by **53.5%**. Checked on 5 architectures. |
| 2 | **Ternary weights fail specifically on noise** | vs 3-bit, ternary loses **17.7–20.2 points** on noise but only **5.9–6.4** on digital corruptions (clean gap 3.5–3.9) |
| 3 | **Routes disagree more than random seeds do** | **1.2–2.0×** the disagreement between two seeds of the same route |
| 4 | **The "best" route depends on the corruption** | Clean accuracy picks ternary (B) over distilled Tiny (D) by 0.99 points; D wins on 7 of 14 corruptions, B wins on 2 |
| 5 | **Fewer bytes is not faster** (P49 Pareto table) | On a laptop CPU: A, B 17.8 ms (same as uncompressed), C 16.1 ms, D **8.0 ms**. A, C, D are Pareto-optimal; B is beaten by A |

Every number traces to [`results.md`](results.md), which is the **single authority**.

---

## The four routes ("ChotaDeiT" A–D)

| Route | What it does, plainly | File | Clean accuracy |
|---|---|---|---|
| **A** | Store every weight in 3 bits (8 levels), one scale per group of 36 weights | 9.718 MB | 76.49% |
| **B** | Every weight becomes −1, 0 or +1 (ternary), packed 5 per byte; 15 sensitive layers kept at 8 bits | 9.785 MB | 72.40% |
| **C** | 4 bits everywhere, plus delete 24.375% of the MLP's hidden channels | 9.796 MB | 75.68% |
| **D** | Train the smaller DeiT-Tiny to copy DeiT-Small (distillation), then 8-bit | 9.765 MB | 71.41% |

Longer, plain-language version: [`docs/4-model-routes.md`](docs/4-model-routes.md).

---

## Where everything is

```
README.md            <- you are here
results.md           <- every number, with how it was checked. READ SECOND.
proposal/            <- the UCS532 proposal (PDF + HTML source)
reproduce.sh         <- re-runs the checks on a laptop in ~40 s
requirements.txt

src/                 <- all code (pipeline order below)
  phase1_audit.py        1. measure real file sizes vs the formula
  build_routes.py        2. build each route to an exact byte target
  train_routes.py        3. recovery training, 3000 steps        (GPU)
  finalize.py               recovery helper for crashed runs     (GPU)
  evaluate_routes.py     4. clean + corrupted accuracy, ECE      (GPU)
  stats.py               5. paired significance tests
  aggregate_seeds.py        mean ± std across the 3 seeds
  offline_analysis.py    6. seed floor, per-corruption, calibration (laptop)
  arch_audit_summary.py     5-architecture size check            (laptop)
  cpu_latency.py            CPU latency per route, P49 Pareto axis (laptop)
  decode_bin.py             read a .bin back into a model; round-trip self-test (laptop)
  rescore.py                re-score every model from its .bin on one GPU (Colab)  -> docs/GPU_RUN.md

evidence/            <- raw outputs; nothing here is hand-edited
  logs/                  68 logs from the GPU box, incl. per-image right/wrong arrays
  analysis/              output of offline_analysis.py
  audit_deit/            size audit of DeiT-S and DeiT-Tiny
  audit_multiarch/       size audit of 5 architectures
  latency/               CPU latency measurements
  vitc_models.tgz        all 21 compressed model files
  vitc_evidence.tgz      original archive pulled off the GPU box (same logs)

docs/
  CODE_WALKTHROUGH.md     every script explained, worked examples, likely code questions
  4-model-routes.md      the routes explained + which paper each comes from
  phase1_audit_explained.md   the size audit, line by line, with viva answers
  vit-compression-plan.md     the original plan (background, ViT basics)
  progress.md            project state
  history/               superseded documents, kept so every correction is traceable
  background/            early notes (what a VLM/ViT is); superseded by the plan
  literature/            full text of Kuzmin et al. 2023, our nearest prior work
```

**New to the topic? Start with [`docs/Project_Course.pdf`](docs/Project_Course.pdf)** — a 3-page course from ML basics to every statistic used here.

**Reading order to explain the project:** this README → `results.md` (top section) →
`docs/4-model-routes.md` → **`docs/CODE_WALKTHROUGH.md`** (every script in plain words, with
worked examples) → `docs/phase1_audit_explained.md`.

---

## How the pipeline fits together

```
 DeiT-Small (88.20 MB)
        |
  [1] phase1_audit.py ------> real sizes vs formula   (laptop OK)
        |
  [2] build_routes.py ------> binary-search each route's knob until the
        |                       file hits the byte target
  [3] train_routes.py ------> 3000 steps, 3 seeds, 20% of ImageNet   (GPU)
        |
  [4] evaluate_routes.py ---> per-image right/wrong on 50,000 clean
        |                       + 5,000 images x 14 corruptions       (GPU)
  [5] stats.py -------------> McNemar test + bootstrap + Bonferroni
        |
  [6] offline_analysis.py --> seed floor, fingerprints, calibration  (laptop)
```

Steps 2–5 ran on a university GPU (H100 slice) from `/workspace/vit-compression`; that access
ended on 2026-09-25. Their outputs are in `evidence/`. Paths inside those scripts (`logs/...`)
are relative to the GPU box, not to this repo.

---

## Check it yourself (laptop, ~40 seconds)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./reproduce.sh
```

It (1) recomputes every statistic from the per-image arrays and diffs it against the saved output,
(2) rebuilds the DeiT-S size audit and compares all 16 quantized file sizes byte-for-byte, and
(3) re-derives all 80 audited file sizes from tensor shapes. It prints `All checks passed.`

---

## Questions you will be asked, and the answers

**Q: Isn't compressing a ViT under 10 MB already done?**
Yes — FTerViT reaches 5.81 MB. We don't claim a smaller model. We claim a *fair comparison*:
four methods at the same real size, which nobody had done.

**Q: What's new here, honestly?**
Not a method — every route is published. What's new is the controlled comparison: equal
*measured* bytes, equal training budget, a measured seed floor, and per-corruption results.
Each ingredient has prior work (cited in `results.md` Part 5); the combination does not.

**Q: Why is `params × bits ÷ 8` wrong?**
It counts only the quantized numbers. A real file also stores one scale per group of weights
(16 bits each) and keeps some layers at 16 bits. Those costs don't shrink when you drop bits, so
the error grows as bits fall and as models get narrower.

**Q: How do you know the size numbers are right?**
Two independent derivations. The file is written and measured with `os.path.getsize()`, and
separately predicted from tensor shapes alone. All 80 files agree to within the JSON header.

**Q: Why three seeds? Isn't that too few?**
For a significance test over seeds, yes — so we never compute one. Seeds give mean ± std and
sign consistency. The tests are over **images** (paired McNemar, bootstrap), where n = 5,000 or
50,000.

**Q: Couldn't random seeds explain the different failures?**
We measured that. Two seeds of one route disagree on 6.3–9.6% of images. Routes disagree
1.2–2.0× more, and the confidence interval excludes zero for every pair.

**Q: Miller et al. say out-of-distribution accuracy tracks clean accuracy. Doesn't that kill
your ranking flip?**
Partly, and we say so. Inside one architecture (A, B, C), the clean order holds on every
corruption. The flip needs route D, which changes the architecture — so we state it as a
*deployment decision* result, not a new phenomenon. The ternary noise fingerprint is the
within-architecture result.

**Q: Why is route D's calibration error 2.8× worse?**
Its error is a uniform underconfidence offset (ECE equals accuracy minus mean confidence exactly),
consistent with distillation at temperature 2. One temperature parameter would mostly fix it, so we
report it as a deployment note, not a finding.

**Q: Why weight-only quantization? Papers report W3A3.**
Different, harder problem. We never compare our accuracies to W3A3 papers. All comparisons are
route vs route.

**Q: Did anything go wrong?**
Yes, and it is recorded: nine pipeline bugs (bug table in `results.md`) and a list of retracted
claims. Two bugs were caught *only* because file size was measured alongside accuracy.

---

## Glossary (plain meanings)

| Term | Meaning |
|---|---|
| ViT / DeiT | Vision Transformer: cuts an image into 16×16 patches and processes them like words. DeiT is a ViT trained efficiently on ImageNet. |
| Quantization (INT3, INT4…) | Storing each weight with fewer bits, e.g. 3 bits = 8 possible values |
| Group size / scale | Weights are split into groups; each group stores one multiplier (scale) to convert small integers back to real numbers |
| Ternary | Each weight is −1, 0 or +1 (times a scale) — 3 levels, not the same as 2-bit (4 levels) |
| Base-3 packing | Five ternary weights fit in one byte (3⁵ = 243 ≤ 256), i.e. 1.6 bits per weight |
| Structured pruning | Deleting whole channels so the matrix actually gets smaller |
| Distillation | A small "student" model learns to copy a big "teacher" model's outputs |
| QAT | Quantization-aware training: training while the weights are quantized, so the model adapts |
| ImageNet-C | ImageNet images with synthetic damage: noise, blur, weather, digital artefacts |
| ECE | Expected calibration error: how far the model's confidence is from its actual accuracy |
| McNemar's test | Significance test for two models judged on the *same* images |
| Bootstrap | Resampling the images many times to get a confidence interval |
| Bonferroni | Making the significance threshold stricter when running many tests |
| Seed floor | How much two copies of the same route disagree just from random training differences |
| Accuracy on the line | Miller et al. 2021: across models, corrupted accuracy usually rises in step with clean accuracy |
