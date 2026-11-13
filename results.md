# Results

**Model:** DeiT-Small, 22,050,664 parameters, 88.20 MB at FP32
**Budget:** 10 MB = **3.63 bits per parameter**
**Status:** Phases 1–3 complete. Offline re-analysis and five-architecture size audit completed
2026-09-25 on the saved per-image arrays (no GPU). Every number is measured or computed.
**Evidence:** `evidence/logs/` (68 raw logs, per-image `correct` arrays), `evidence/analysis/`,
`evidence/audit_multiarch/`, `evidence/vitc_models.tgz` (all 21 model files). Previous version of
this file: `docs/history/results_pre_2026-09-25.md`.

---

## Provenance key

| Mark | Meaning |
|---|---|
| **[M]** | Measured — written to disk and read back, or scored on real images |
| **[C]** | Cross-checked — reproduced a second, independent way |
| **[A]** | Assumption, extrapolation, or published figure we did not verify ourselves |

---

# The research problem, resolved — read this first

**In plain words:** we squeezed DeiT-Small to the same ~9.77 MB file four different ways. The four
files look the same on a spec sheet. They are not the same model. And the number compression
papers print for "file size" is itself wrong by a predictable amount.

**The defensible headline (one sentence):**

> At an identical measured file size and an identical training budget, the compression route
> decides **which kinds of corruption** a Vision Transformer fails on — a pattern that neither clean
> accuracy nor the ImageNet-C average (the usual single robustness number) reveals — and the size
> figure the ViT compression literature prints understates the real file by **1.5% to 53.5%**
> (group 256; more at finer groups), depending on bit-width and model width, across five architectures.

| # | Claim | Key evidence | Strongest prior work | Why it survives |
|---|---|---|---|---|
| **1** | **Corruption fingerprints.** Inside one architecture, ternary's deficit concentrates in noise | Ternary vs INT3: **−17.7 to −20.2 pts on noise, −5.9 to −6.4 on digital**, against −3.5 to −3.9 clean; noise-minus-digital difference CI excludes 0 in 6/6 tests | Lost in Pruning (MLSys 2021); 2405.20876 | They compare compressed vs dense, one method at a time. We compare methods, at equal bytes, per corruption |
| **2** | **Routes disagree beyond training randomness** | Cross-route disagreement is **1.2–2.0× the measured seed floor**; excess +1.8 to +7.0 pts clean, CI excludes 0 for all 6 pairs | Hooker 2019/2020; D'Amour 2020 (underspecification) | We *measure* the seed floor and subtract it; the route effect is what is left |
| **3** | **The deployment decision flips** — shrink the architecture (D) or ternarize the weights (B)? | Clean favours B (+0.99, p ≤ 2.6e-08, 3/3 seeds); corruption favours D on 7/14 corruptions and B on 2/14 (significant, 3/3 seeds) | **Miller et al. 2021, *Accuracy on the Line*** | Stated as a decision-level result, **not** as a violation of accuracy-on-the-line: the flip needs D's architecture change (§1.5) |
| **4** | **Size is misreported, on two axes, across 5 architectures** | INT2 overhead **+53.5% → +8.5%** as width grows 192 → 1024; overhead rises as bits fall in 5/5 models; 80/80 files rebuilt from shapes to within the header | M²XFP, SpQR (LLM side) | The accounting is standard in LLM quantization (small groups) and omitted by the ViT papers we checked, a fair convention for their per-channel scales (+2.2–5.4% in Mix-QViT's own layout) that fails at small groups and small models |

**Supporting, not headline:** metadata is 12.6% of route A's file (§3.1); the group-size sweep
(§3.2); calibration is a uniform confidence offset (§1.6); ternary recovers 3.3× faster per step (§1.7).

**What did not survive the 2026-09-25 checks** (details in *Retracted claims*):
"Kuzmin et al. are wrong about quantization overhead"; "calibration is a third independent ranking";
"33–43% unshared failures" as a route effect; "the A–C gap amplifies under corruption"; any claim
that the inversion happens *within* one architecture.

**Novelty, honestly:** an empirical, controlled benchmark — a new *configuration*, not a new
phenomenon. No paper found combines ViT + four routes + measured-byte parity + equal budget +
seed-floor control + per-corruption fingerprints. Every ingredient has prior work, cited in Part 5.
**Coursework: solid. SCIE journal (Scientific Reports / IEEE Access tier): plausible if framed
this way. Top venue: no.**

**The one open question the saved data cannot answer:** whether ternary sits *below* the
same-architecture accuracy line, or on a steep part of it (§1.4, off-the-line test inconclusive).
Settling it needs the 5 group-sweep models scored under corruption — the models are archived; see
*What would settle the last open question*.

---

# Part 1 — Behaviour at equal size (lead contribution)

## 1.1 The twelve models **[M][C]**

Four routes × three seeds. All binary-searched to a 9.77 MB budget line, then trained for
**exactly 3000 optimizer steps**, weight-only QAT (quantization-aware training) with a
straight-through estimator, on 20% of ImageNet-1k (200 of 1024 WebDataset shards). Loss:
cross-entropy with `label_smoothing=0.1` on every route; route D adds KL to a DeiT-S teacher at
temperature 2 (`src/train_routes.py:262–267`).

| Model | Route | Knob (from the logs) | Exported size | codes / scales / fp16 (MB) |
|---|---|---|---|---|
| ChotaDeiT-A | Uniform INT3 | **group size 36** | **9.718 MB** | 8.22 / 1.22 / 0.28 |
| ChotaDeiT-B | Ternary + base-3 packing | **15 layers at INT8** | **9.785 MB** | 9.34 / 0.16 / 0.28 |
| ChotaDeiT-C | INT4 all + structured prune | **24.375% of MLP channels** (1536 → 1162) | **9.796 MB** | 9.23 / 0.29 / 0.27 |
| ChotaDeiT-D | DeiT-Tiny + distillation | **38 layers at FP16** | **9.765 MB** | 1.70 / 0.03 / 8.03 |

**Cross-checks [C]:**
- All 17 seeded and sweep files in `evidence/vitc_models.tgz` match their training logs **to the
  byte**.
- The codes/scales/fp16 accounting rebuilds each exported file to ≤ 0.006 MB.
- Group 36 is confirmed arithmetically: predicting route A's file from the g=32 sweep point by scale
  cost alone gives **9.7181 MB at g=36** (measured 9.718109) against 9.7529 at g=35.

**Size spread: 9.718–9.796 MB = 0.80% of the 9.766 MB mean.** Report as "within 0.8%", never as
exact equality. The spread is between the four routes' targets; each export hits its own target
exactly.

> **Superseded — do not reuse.** Earlier versions of this file listed the knobs as 35 / 14 / 24.37%
> / 37. Those are the `int()`-truncated knobs of the **pre-fix** single runs (logged as 35.875,
> 14.84, 0.2437, 37.5), whose sizes drifted to A 9.753 / B 9.665 / D 9.620 MB — bug #5. Group 35
> predicts 9.7529 MB, matching the pre-fix A to 0.0001 MB. All results below use the post-fix models.

**Measured ternary zero-fraction: 43.1% [M].** FTerViT reports 37.3% **[A]**. Entropy at 43.1% zeros
= **1.555 bits**, against base-3 packing's 1.600 — entropy coding would gain **0.045 bits/weight**.
Accuracy-optimal ternarization produces no exploitable sparsity.

## 1.2 Headline accuracies **[M][C]**

Mean ± standard deviation over 3 seeds. Clean accuracy on the full **50,000**-image validation set;
corruption on a **5,000**-image subset (14 ImageNet-C corruptions at severity 3, generated once with
`imagecorruptions` and cached so every model saw byte-identical inputs; `glass_blur` excluded for
runtime). Corrupting all 50,000 images needed ~15 GB per worker, over the 90 GB container limit.

| Route | Clean (n=50,000) | Corrupted (n=5,000) | Drop | Retained | ECE clean |
|---|---|---|---|---|---|
| **A** — INT3 | **76.49 ± 0.05** | **54.99 ± 0.26** | 21.50 | 71.9% | **4.35 ± 0.02** |
| **C** — INT4+prune | 75.68 ± 0.01 | 53.49 ± 0.14 | 22.19 | 70.7% | 4.48 ± 0.15 |
| **B** — ternary | 72.40 ± 0.14 | 44.24 ± 0.13 | **28.16** | **61.1%** | 5.41 ± 0.19 |
| **D** — Tiny+distill | 71.41 ± 0.13 | 48.01 ± 0.15 | 23.40 | 67.2% | **12.15 ± 0.10** |

**Cross-checks [C]:**
- Every value above was **recomputed on the Mac from the raw per-image arrays**, matching to the
  printed precision; the B-vs-D McNemar p-values reproduce as 5.13e-09 / 2.60e-08 / 1.18e-10.
- The 5,000-image clean arrays are **bit-identical** to the first 5,000 entries of the separate
  50,000-image runs, for all four routes: the pipeline is deterministic.
- Training-time accuracy on a 25,600-image held-out split agrees within 0.22 points (A 76.56,
  B 72.51, C 75.79, D 71.63).

**Seed noise: 0.01–0.14 points clean, 0.13–0.26 corrupted.**

## 1.3 Routes disagree more than seeds do **[M]**

**Why this section exists:** two models that differ *only in random seed* already fail on different
images (D'Amour et al. 2020, *underspecification*: "many predictors with equivalently strong
held-out performance … can behave very differently in deployment domains"). So "routes fail on
different images" means nothing until the seed floor (the disagreement you get from seeds alone) is
measured and subtracted.

**Method:** disagreement = fraction of images where exactly one of two models is correct.
Seed floor = within-route pairs (different seeds). Cross-route pairs are restricted to *different*
seeds too, so both sides differ in seed and only one side also differs in route. 95% CI by
bootstrap over images. Script: `src/offline_analysis.py` §1.

**Prediction written before running:** seed floor 3–6%. **It was wrong — the floor is 6.3–9.6%.**
The fragile direction was ours, and it cut against us: much of the old "33–43% unshared failures"
was training randomness.

| Route | Seed-vs-seed disagreement (clean) | Seed Jaccard (clean) | Seed-vs-seed (corrupted) |
|---|---|---|---|
| A | 8.49% | 0.694 | 12.56% |
| B | 9.55% | 0.705 | 13.07% |
| C | 7.72% | 0.726 | 11.76% |
| D | 6.31% | 0.801 | 9.10% |

| Pair | Cross-route | Seed floor | **Excess** | 95% CI | Ratio | Corrupted excess | Ratio |
|---|---|---|---|---|---|---|---|
| A–C | 9.92% | 8.11% | **+1.82** | [+1.70, +1.94] | **1.2×** | +2.95 [+2.80, +3.10] | 1.2× |
| A–B | 12.31% | 9.02% | +3.29 | [+3.14, +3.44] | 1.4× | +7.78 [+7.52, +8.03] | 1.6× |
| B–C | 12.38% | 8.63% | +3.74 | [+3.59, +3.89] | 1.4× | +7.95 [+7.71, +8.19] | 1.6× |
| B–D | 14.13% | 7.93% | +6.20 | [+6.01, +6.40] | 1.8× | +10.03 [+9.76, +10.32] | 1.9× |
| A–D | 14.27% | 7.40% | +6.87 | [+6.67, +7.08] | 1.9× | +9.05 [+8.78, +9.33] | 1.8× |
| C–D | 14.01% | 7.02% | **+6.99** | [+6.79, +7.19] | **2.0×** | +9.25 [+8.99, +9.53] | 1.9× |

**Finding.** Route disagreement exceeds the seed floor for **every pair, clean and corrupted**, with
every interval excluding zero. The size of the route effect is **1.2–2.0×** seed-level churn:
A and C (both "uniform quantization") are barely more different than two seeds of one route; any
pair involving ternary (B) or the smaller architecture (D) is 1.4–2.0×.

## 1.4 Corruption fingerprints — within one architecture **[M]**

A, B and C share **architecture, pretrained starting weights, byte budget and step budget**. Only the
weight representation differs. This is the cleanest comparison in the study.

| Corruption | Family | A | B | C | D | D − B | D vs B, significant seeds |
|---|---|---|---|---|---|---|---|
| gaussian noise | noise | 55.84 | 37.68 | 53.35 | 47.22 | +9.54 | D, 3/3 |
| shot noise | noise | 54.49 | **34.35** | 52.17 | 45.84 | +11.49 | D, 3/3 |
| impulse noise | noise | 55.73 | 37.27 | 53.13 | 46.62 | +9.35 | D, 3/3 |
| defocus blur | blur | 46.13 | 37.21 | 43.25 | 39.17 | +1.96 | D, 2/3 |
| motion blur | blur | 50.57 | 42.43 | 49.31 | 44.47 | +2.03 | D, 2/3 |
| zoom blur | blur | 39.79 | 32.81 | 39.58 | 35.03 | +2.23 | D, 2/3 |
| snow | weather | 49.35 | 37.79 | 48.15 | 44.57 | +6.78 | D, 3/3 |
| frost | weather | 47.91 | 36.95 | 47.07 | 41.39 | +4.44 | D, 3/3 |
| fog | weather | 62.67 | 45.90 | 60.99 | 56.94 | +11.04 | D, 3/3 |
| brightness | weather | 73.25 | 67.25 | 72.93 | 67.75 | +0.50 | 0/3 |
| contrast | digital | 71.19 | 54.64 | 69.92 | 66.37 | +11.73 | D, 3/3 |
| elastic | digital | 39.92 | 37.78 | 39.40 | **31.07** | **−6.71** | **B, 3/3** |
| pixelate | digital | 60.47 | **59.15** | 58.56 | **48.77** | **−10.38** | **B, 3/3** |
| jpeg | digital | 62.55 | 58.21 | 61.00 | 56.96 | −1.25 | 0/3 |

Means over 3 seeds, n=5,000 each. McNemar exact per corruption per seed, Bonferroni over 14
(p < 0.0036).

**Finding — the ternary fingerprint.** Against INT3 (A), ternary (B) loses **17.7–20.2 points on the
noise family** but only **5.9–6.4 on the digital family**, against a clean gap of 3.5–3.9 on the
same images. The noise-minus-digital difference is **+11.7 to +13.8 points (A–B) and +11.5 to +12.0
(C–B)**, with every interval excluding zero (6/6, Bonferroni over 6). On pixelate the ternary deficit
is 1.3 points; on shot noise it is 20.1.

**Honest boundary [M].** Inside DeiT-S, **no corruption reverses the clean order** in all three seeds
(42 McNemar tests, Bonferroni p < 0.0012): the lower-clean model stays lower everywhere, which is
what accuracy-on-the-line predicts. The families were fixed in advance (standard ImageNet-C
taxonomy); **the noise-vs-digital contrast was chosen after seeing the per-corruption table, so it is
exploratory.**

**What the data cannot say yet.** Is ternary *below* the same-architecture accuracy line, or on a
steep part of it? A probit-scale line through A and C predicts B's corrupted accuracy with
residuals of −4.84 / −1.92 / −2.56 points, but the 95% intervals are [−7.9, +15.2],
[−51.6, +36.4], [−7.2, +33.2]. **Two reference points cannot fix a slope. Inconclusive.**

## 1.5 The decision flip: shrink the architecture, or ternarize? **[M]**

| By clean top-1 (n=50,000) | By corrupted top-1 (n=5,000) |
|---|---|
| 1. A — 76.49 | 1. A — 54.99 |
| 2. C — 75.68 | 2. C — 53.49 |
| **3. B — 72.40** | **3. D — 48.01** |
| **4. D — 71.41** | **4. B — 44.24** |

| B vs D | Gap | Test (per seed) | vs seed noise | Signs |
|---|---|---|---|---|
| Clean (n=50,000) | **+0.99** (B ahead) | McNemar **5.1e-09 / 2.6e-08 / 1.2e-10** | 7.3× | **3/3** |
| Corrupted mean (n=5,000) | **−3.77** (D ahead) | bootstrap CI [−4.48, −3.02] / [−4.68, −3.22] / [−4.34, −2.88], Bonferroni-12 | 26.3× | **3/3** |
| Per corruption | D ahead on **7/14**, B ahead on **2/14** | McNemar, Bonferroni-14, significant in 3/3 seeds | — | — |

**What this is.** Both B and D reach the same ~9.77 MB file. A deployer choosing between them on
published (clean) accuracy picks B. Under noise, fog and contrast, D is 9–12 points better; under
pixelate and elastic, B is 7–10 points better. **The ImageNet-C average (D ahead by 3.77) hides a
crossing.**

**What this is not.** Not evidence that clean accuracy fails to predict robustness *within* a model
family. The flip only involves D, the one route that changes the architecture (DeiT-Tiny, 192 wide),
and Miller et al. (ICML 2021) already report that synthetic shifts are where the clean-to-OOD
correlation is weaker across families. **Cite Miller directly and state the claim at the decision
level.**

> At n=5,000 the clean B-vs-D gap read p ≈ 0.01–0.02 and did not clear correction; at n=50,000 it
> clears by five to seven orders of magnitude. "Not significant" at one sample size is not a finding.

## 1.6 Calibration — a uniform confidence offset, not a route property **[M][C]**

ECE (expected calibration error, 15 bins: how far confidence is from accuracy) on clean images
equals **accuracy minus mean confidence to within 0.02 points, for every route**. That identity
holds only if **every confidence bin is underconfident**: the miscalibration is a single global
shift, which one temperature parameter largely removes (temperature scaling, Guo et al. 2017).

| Route | ECE clean | acc − conf, clean | ECE corrupted (mean of 14) | acc − conf, corrupted |
|---|---|---|---|---|
| A | 4.35 | +4.35 (under) | 4.11 | −0.72 (over) |
| B | 5.41 | +5.41 (under) | 4.12 | −2.21 (over) |
| C | 4.48 | +4.46 (under) | 4.08 | −1.05 (over) |
| D | **12.15** | **+12.15 (under)** | **8.83** | **+7.74 (under)** |

**Reading.** All four routes ship underconfident on clean data, consistent with `label_smoothing=0.1`
in the shared recipe **[A — mechanism not tested]**. D's 2.8× larger offset is consistent with
distillation at temperature 2 **[A]**. Under corruption, A/B/C flip to overconfident while D stays
underconfident. **Report as a deployment note, not a finding:** calibrate each route before use; a
temperature fitted on clean data would push A/B/C further into overconfidence under corruption.
Per-image confidences were not saved, so post-temperature ECE cannot be computed from the evidence.

## 1.7 Recovery at a fixed budget **[M]**

Subset accuracy during training, mean of the 3 post-fix seeds (40-batch subset evaluations):

| Step | A | B | C | D |
|---|---|---|---|---|
| 500 | 75.65 | 67.26 | 74.68 | 70.56 |
| 1000 | 75.53 | 69.24 | 74.98 | 70.69 |
| 1500 | 75.75 | 70.49 | 75.45 | 71.16 |
| 2000 | 76.35 | 71.83 | 75.81 | 71.59 |
| 2500 | 76.54 | 72.33 | 76.08 | 71.75 |
| 3000 | 76.63 | 72.89 | 76.11 | 71.88 |

**Nothing converged at 3000 steps.** Steps 1000→3000: A gains **0.55** points per 1000 steps (seeds
0.66 / 0.42 / 0.57), B gains **1.83** (1.66 / 1.83 / 1.99) — **3.3× faster**, from 3.7 points lower.
A linear extrapolation puts a crossover near step ~5,900 **[A — not tested; recovery curves are
concave, so linear extrapolation is optimistic about B]**. The ranking is budget-dependent: *at a
fixed 3000-step recovery budget, ternary starts lower and recovers faster.* The qualitative
direction is known; report as an observation, not a headline.

## 1.8 How these statistics are reported — and how not to

| Level | Unit | Correct test | Report as |
|---|---|---|---|
| **Across images** (fixed models) | images | McNemar exact (paired); bootstrap over images for corruption | p-values and 95% CIs, Bonferroni-corrected for the stated family |
| **Across seeds** | 3 seeds | **None** | **Mean ± std and sign consistency only** |

> **Never quote a p-value over seeds.** Three seeds is far too few to significance-test training
> variance. "+0.99 points, 7.3× seed noise, consistent in 3/3 seeds" is a statement about effect
> size and reproducibility, not a hypothesis test.

Corruption conditions are **not independent** — 14 corruptions of the same 5,000 images are not
70,000 samples. The image-level bootstrap resamples **images**; a naive two-proportion test
inflates significance by roughly √14 ≈ 3.7×.

---

# Part 2 — Size is systematically misreported, across five architectures

## 2.1 Where DeiT-S's parameters are **[M][C]**

Hand calculation from the architecture matched the script to the parameter.

| Component | Params | Share | FP32 |
|---|---|---|---|
| MLP (12 blocks) | 14,178,816 | 64.3% | 56.72 MB |
| Attention (12 blocks) | 7,096,320 | 32.2% | 28.39 MB |
| Classifier head | 385,000 | 1.7% | 1.54 MB |
| Patch embedding | 295,296 | 1.3% | 1.18 MB |
| Position embed + CLS | 76,032 | 0.3% | 0.30 MB |
| LayerNorms | 19,200 | 0.1% | 0.08 MB |
| **Total** | **22,050,664** | **100%** | **88.20 MB** |

## 2.2 Measured file sizes vs the formula, DeiT-S **[M]**

Written to disk, measured with `os.path.getsize()`. Symmetric group-wise quantization, FP16 scales,
`patch_embed`/`head`/norms/biases kept FP16 (standard tooling policy). Raw binary container, not
`torch.save`.

| Format | Group | Nominal | **Measured** | bits/param | Overhead | Fits 10 MB |
|---|---|---|---|---|---|---|
| FP32 | — | 88.20 | 88.24 | 32.01 | +0.0% | no |
| FP16 | — | 44.10 | 44.14 | 16.01 | +0.1% | no |
| INT8 | 256 | 22.05 | 23.04 | 8.36 | **+4.5%** | no |
| INT8 | 32 | 22.05 | 24.20 | 8.78 | +9.8% | no |
| INT4 | 256 | 11.03 | 12.42 | 4.51 | **+12.7%** | **no** |
| INT4 | 128 | 11.03 | 12.59 | 4.57 | +14.2% | no |
| INT4 | 32 | 11.03 | 13.59 | **4.93** | +23.2% | no |
| INT3 | 256 | 8.27 | 9.77 | 3.54 | **+18.1%** | yes |
| INT3 | 128 | 8.27 | 9.94 | 3.60 | +20.2% | yes, by 65 KB |
| INT3 | 32 | 8.27 | 10.93 | 3.97 | +32.2% | no |
| INT2 | 256 | 5.51 | 7.12 | 2.58 | **+29.1%** | yes |
| INT2 | 32 | 5.51 | 8.28 | 3.00 | +50.1% | yes |

**Finding 1.** The error **grows as bits fall**: +4.5% at INT8 → +29.1% at INT2. Scales and
un-quantized layers are fixed costs that do not shrink when the codes do.
**Finding 2.** A "4-bit" model at group 32 is a **4.93-bit** model — same label, 9.4% larger file.
**Finding 3.** INT4 cannot reach 10 MB in any configuration — 12.42 MB at best, **24% over**.

## 2.3 Five architectures **[M][C]**

`src/phase1_audit.py --random-init` on each architecture. File sizes depend only on tensor shapes, so
weights need not be downloaded — **verified: random-init and pretrained audits are byte-identical on
all 16 quantized files for both DeiT-S and DeiT-Tiny** (the two `torch.save` float files differ by a
constant 391 bytes of container framing from a newer torch; no claim uses them).
Second derivation: `src/arch_audit_summary.py` rebuilds every quantized file from shapes alone
(codes + `ceil(n/g)` FP16 scales + FP16 leftovers); all **80 files** match, the only gap being a
7–15 KB JSON header.

Overhead = measured file ÷ `params × bits ÷ 8` − 1. Group 256 unless stated.

| Model | Params | Width | FP16 share | INT8 | INT4 | INT3 | INT2 | "INT4" at g=32 |
|---|---|---|---|---|---|---|---|---|
| DeiT-Tiny | 5,717,416 | 192 | 7.2% | +8.0% | +23.2% | +33.3% | **+53.5%** | +33.3% |
| DeiT-Small | 22,050,664 | 384 | 3.7% | +4.5% | +12.7% | +18.1% | +29.1% | +23.2% |
| Swin-Tiny | 28,288,354 | 96* | 3.0% | +3.8% | +10.7% | +15.3% | +24.5% | +21.3% |
| DeiT-Base | 86,567,656 | 768 | 1.9% | +2.7% | +7.2% | +10.2% | +16.3% | +17.9% |
| ViT-Large | 304,326,632 | 1024 | 0.8% | +1.5% | +3.9% | +5.4% | **+8.5%** | +14.7% |

\* Swin is hierarchical; 96 is its first-stage width.

**Finding 4 — two axes, now across architectures.** Overhead rises as bits fall in **5/5** models,
and falls as width grows in **4/4** plain ViTs (INT2: +53.5% → +29.1% → +16.3% → +8.5%). Head, patch
embedding and position embeddings scale with *width*; total parameters scale with *width squared*,
so narrower models carry a larger un-quantizable share.

**Finding 5 — a floor that does not shrink.** At group 32, symmetric FP16 scales cost 16/32 = 0.5
bits per quantized weight, **12.5% of a 4-bit code**, for every model size. Even ViT-Large's "4-bit"
file is **+14.7%** over nominal.

**Baseline [M]:** DeiT-Tiny INT8 = **6.22 MB** (g128). Published accuracy 72.2% **[A]**.

## 2.4 The un-quantized leftovers, DeiT-S **[M]**

817,000 params (3.7% of the model) = 1.63 MB at FP16, constant across bit-widths.

| Format | Leftovers as % of file | Fully quantizing saves |
|---|---|---|
| INT8 | 7.0% | 2.9% |
| INT4 | 13.0% | 8.0% |
| INT3 | 16.4% | 11.0% |
| **INT2** | **22.4%** | **16.2%** |

DeiT-Tiny at INT2: leftovers are **36.6%** of the file.

## 2.5 The literature: two subfields, two accounting practices **[C]**

**LLM quantization counts metadata.** M²XFP (2026) defines *"EBW = B_elem + (B_meta + B_scale)/k"*;
SpQR accounts 0.625 bits/weight of grouping metadata; BayesQ's storage budget includes per-block
scales. **This accounting is not our invention.**

**The ViT compression papers we checked do not:**

| Paper | Budget definition | Reports MB? | Counts scales? |
|---|---|---|---|
| LampQ (2025) | *"average bit-width … mean across all parameters"*, `Σcᵢ Bᵢ ≤ b_t Σcᵢ` | No | No |
| MixFrag (2026) | `Σᵢ Σⱼ bⱼ xᵢ,ⱼ ≤ B_target`, reports W3/A3 | No | No |
| Mix-QViT (CVPRW 2026) | Eq. 28c `Σ_l Σ_u Σ_j α\|w\| bⱼ ≤ C_M` | **"11.4 MB (4-bit) DeiT-S"** | No |
| Hessian-guided MPQ of ViTs (*Sci. Rep.* 2026) | not stated | **"10.83 MB" at 4-bit, DeiT-S** | Not stated; reconstructs as no |

**Both reported sizes reconstruct from the nominal formula [C]:**

```
  Mix-QViT   22,050,664 × 4 ÷ 8 + patch_embed & head at 8-bit  = 11.37 MB  vs reported 11.4   (0.03 MB)
  Sci. Rep.  21,665,664 × 4 ÷ 8   (DeiT-S minus classifier head) = 10.833 MB vs reported 10.83  (0.003 MB)
```

The *Sci. Rep.* 8-bit figure (21.3 MB) does not reconstruct as cleanly (predicts 21.67); stated, not
hidden.

**Corrected 2026-10-08 (authors' frame).** The earlier "9.0–19.2% short" compared Mix-QViT's figure with
*our* files, which keep patch_embed/head at FP16 and use small groups. Mix-QViT quantizes per layer /
per channel (Eq. 6–8) with patch_embed and head at 8-bit. Rebuilt that way from DeiT-S shapes [C]:
per-channel symmetric FP16 scales + FP16 leftovers = **11.66 MB (+2.2%)**; asymmetric scale+zero +
FP32 leftovers = **12.01 MB (+5.4%)**. Per-channel scales cost 0.031 bits/weight vs 0.444 at group 36.
Group-wise in the same layout: g256 +3.0%, g128 +4.4%, g64 +7.3%, g32 +13.1%.
**Reading:** the formula is a fair convention for per-channel quantizers (LampQ defines its budget as
"the mean across all parameters" on purpose). It fails under small groups and small models, which a
hard MB budget forces. *Sci. Rep.* quantizer granularity not checked → no % claimed for it.

**Honest counterexample:** FTerViT reports "5.81 MB flash" as a genuine footprint including FP32
biases and statistics, validated on an ESP32-S3 partition. The pattern is *not* "the field is
wrong" — it is that **ViT bit-allocation papers optimise a budget that is not the deployment
budget, a practice the LLM side has moved past.**

---

# Part 2b — CPU latency: the third Pareto axis (problem P49) **[M]**

P49 asks for accuracy × size × CPU latency. Latency depends on architecture and arithmetic, not
weight values, so it is measured with random weights: `src/cpu_latency.py`, Apple M2, PyTorch 2.14,
batch 1, 10 warm-up + 50 timed runs, median. Raw: `evidence/latency/cpu_latency.json`.

| Model | Size | Top-1 (n=50,000) | FP32 kernels, 4 threads | 1 thread | torch.ao dynamic INT8, 4 threads |
|---|---|---|---|---|---|
| DeiT-S uncompressed | 88.24 MB | 79.8% **[A — published, not re-measured]** | 17.8 ms | 23.2 ms | 101.7 ms |
| A — INT3 | 9.718 MB | 76.49% | 17.8 ms | 23.2 ms | — |
| B — ternary | 9.785 MB | 72.40% | 17.8 ms | 23.2 ms | — |
| C — INT4 + prune | 9.796 MB | 75.68% | 16.1 ms | 21.0 ms | 88.6 ms |
| D — DeiT-Tiny | 9.765 MB | 71.41% | **8.0 ms** | 9.7 ms | 30.2 ms |

**Finding.** At equal size, A, C and D are Pareto-optimal on accuracy × latency; **B is dominated
by A** (same latency, 4.09 points lower). 3-bit and ternary formats have no CPU kernel: they decode
to float and run at the decoded architecture's speed, so **9× fewer bytes bought zero
milliseconds**. torch.ao dynamic INT8 (qnnpack) is 3.8–5.7× *slower* than FP32 on this ARM CPU —
**not yet verified** on x86 (fbgemm) or a phone.

**Stated accuracy loss (P49):** ChotaDeiT-A loses ~3.3 points against the *published* 79.8%. Our
pipeline never scored the uncompressed DeiT-S, so this number is **[A]** until Phase 4 measures it.

---

# Part 3 — Metadata at a byte budget (supporting)

## 3.1 How big the scale bill is, and what Kuzmin et al. actually said **[M][C]**

Route A spends **1.22 MB of its 9.718 MB file (12.6%) on scales.**

Kuzmin et al. (NeurIPS 2023, §6, read in full from `docs/literature/kuzmin2023_pruning_vs_quantization_fulltext.txt`): unstructured pruning needs a
1-bit mask per weight, *"a 6.25% storage overhead at a minimum"* on FP16 weights, while *"the only
storage overhead [for quantization] is a single scaling factor per tensor (or channel)."*

**In their frame — overhead as a share of an FP16 weight — their comparison holds at every
configuration we built:**

| Scale layout (symmetric, FP16 scales) | bits / weight | % of an FP16 weight | vs pruning's 6.25% |
|---|---|---|---|
| per-channel (384-wide rows) | 0.042 | 0.26% | smaller |
| group 128 | 0.125 | 0.78% | smaller |
| **group 36 (route A)** | **0.444** | **2.78%** | **smaller** |
| group 32 | 0.500 | 3.12% | smaller |
| group 32, *asymmetric* (not built) | 1.000 | 6.25% | equal |

**What survives is a denominator argument, not a correction.** When the constraint is a byte budget,
the relevant denominator is the compressed file, not FP16: the same 0.444 bits is **12.6% of route
A's file**. Kuzmin's statement is correct for their question (compression ratio vs FP16) and
uninformative for ours (what fits in 10 MB).

## 3.2 The group-size sweep **[M][C]**

Route A's knob, swept at fixed INT3, seed 0, identical 3000-step budget:

| Group | Measured size | Clean top-1 | Cost of moving one step finer |
|---|---|---|---|
| 512 | 8.586 MB | 74.68% | — |
| 256 | 8.672 MB | 75.31% | +0.086 MB buys +0.63 pts = **7.33 pts/MB** |
| 128 | 8.843 MB | 75.84% | +0.171 MB buys +0.53 pts = 3.10 pts/MB |
| 64 | 9.186 MB | 76.16% | +0.343 MB buys +0.32 pts = 0.93 pts/MB |
| 32 | 9.870 MB | 76.68% | +0.684 MB buys +0.52 pts = **0.76 pts/MB** |

**Accuracy side:** returns collapse **9.6×** (7.33 → 0.76 pts/MB); 1.56 pts/MB averaged over the
sweep. **This is the known group-size trade-off** (why g=128 is the common default in LLM
quantization); here it is measured in megabytes on a trained ViT. Supporting data, not a contribution.

**Size side — a serializer validation, not a finding [C].** Codes are the same size at every group
by construction, so every gap must be scales: `16/g` bits × 21,910,664 quantized weights predicts all
four gaps to ≤ 0.7 KB, and route A's own file (g=36) to 0.0000 MB. This proves the file writer is
correct. It is not evidence about the world.

**Between routes there is no size-accuracy trade:** route A is the smallest of the four files
*and* the most accurate, clean and corrupted — it strictly dominates B, C and D.

---

# Part 4 — A reporting protocol

What a compression paper should report. Every item is available at save time.

| Report | Because |
|---|---|
| **Measured file size in bytes**, from `os.path.getsize()` | `params × bits ÷ 8` understates by 1.5–53.5% at group 256 across the five architectures and four bit-widths audited, and more at finer groups |
| **Scale granularity** (per-tensor / per-channel / group size) and symmetric vs asymmetric | Per-channel costs 0.04 bits/weight; group 32 costs 0.5 (symmetric) or 1.0 (asymmetric) — up to 25% of a 4-bit code |
| **Which layers stay un-quantized** | patch_embed + head + norms are 3.7% of DeiT-S parameters but **22.4% of an INT2 file** (36.6% for DeiT-Tiny) |
| **The serialization format** | A `torch.save` container added 45% to our quantized files before we switched to raw binary |
| **Per-corruption results, not only the ImageNet-C mean** | The mean hides a B-vs-D crossing (§1.5) and a 5× noise-specific ternary deficit (§1.4) |
| **A seed floor for any disagreement metric** | Seeds alone produce 6.3–9.6% disagreement on clean ImageNet (§1.3) |

---

# Part 5 — Positioning against prior work

| Work | What it did | What we add, and what we concede |
|---|---|---|
| [Kuzmin et al., NeurIPS 2023](https://arxiv.org/abs/2307.02973) | Pruning vs quantization at equal **theoretical** ratio, equal fine-tuning epochs, accuracy only | **Concede the equal-budget control — cite it as theirs.** Add: measured bytes; four routes incl. their stated future work (prune+quant); structured pruning; behaviour beyond accuracy. Their overhead remark holds in their frame (§3.1) |
| [Miller et al., *Accuracy on the Line*, ICML 2021](https://arxiv.org/abs/2107.04649) | OOD accuracy is *"strongly correlated with in-distribution performance"*; weaker for *"synthetic distribution shifts from CIFAR-10-C"* | Explains why a cross-architecture flip (B vs D) is not surprising in itself. Our within-architecture data is consistent with their monotone line (§1.4). **Cite; frame §1.5 at the decision level** |
| [D'Amour et al. 2020, underspecification](https://arxiv.org/abs/2011.03395) | Equally accurate predictors *"can behave very differently in deployment domains"* | We measure that floor directly (seeds) and show routes exceed it by 1.2–2.0× (§1.3) |
| [Lost in Pruning, MLSys 2021](https://arxiv.org/abs/2103.03014) | Pruned vs dense networks diverge beyond test accuracy | Method vs method at equal bytes, not compressed vs dense **[A — read from abstract/summary only]** |
| [2405.20876](https://arxiv.org/pdf/2405.20876) | Post-hoc pruning of CNNs: calibration and corruption robustness | ViTs; four routes; per-corruption fingerprints **[A — read from search summary only]** |
| [Hooker et al. 2019](https://arxiv.org/abs/1911.05248) / [2020](https://arxiv.org/abs/2010.03058) | Compressed vs dense models diverge on identifiable examples | Route vs route at equal size, with a seed floor. **Cite as support** |
| [Compression vs corruption in CNNs, 2512.24971](https://arxiv.org/html/2512.24971) | Quantization/pruning/clustering vs CIFAR-C, **not equalised**; concludes compression can *improve* robustness | Equalisation is the point — without it a less-compressed model simply looks more robust |
| Kuzmin, Appendix I | *"QAT leads to learning completely new representations"* (qualitative) | A quantitative, seed-controlled measurement (§1.3) |

---

# Retracted claims

Recorded so they are not reintroduced.

| Claim | Why it was wrong |
|---|---|
| "Entropy coding recovers ~0.5 bits/weight on ternary ViTs" | Measured zero-fraction 43.1% (ours) / 37.3% (FTerViT). Near-uniform. Gain is **0.045 bits**. |
| "A and C are tied on clean accuracy" | McNemar (paired) separates them. Pairing added power, as warned. |
| "z = 11.5 / z = 11.7" | Treated 70,000 correlated samples as independent; unpaired test on paired data; no correction. Inflated ~3.7×. |
| "Base-3 packing achieves 1.585 bits" | 1.585 is the entropy **floor**. Base-3 packing achieves **1.600** (= 8/5). |
| "Clean accuracy cannot distinguish A from C" | True at n=5,000, false at n=50,000 (p = 2.3e-09 / 6.0e-10 / 3.3e-08). A power artifact. |
| "The packing win is 24%" | FTerViT's 2.11 bits includes 0.30 MB of FP32 extras. Packing touches only the codes: **18.9%**. |
| "Route A buys its accuracy at ~51.8 pts/MB over D" | Built on pre-fix sizes that made A look larger than D. A is smaller **and** better: no trade exists. |
| "Granularity's marginal return collapses 10×" | It collapses **9.64×**. |
| **"Kuzmin et al.'s overhead claim is false for group-wise quantization"** (2026-09-12) | In their frame (% of FP16) it holds at every config we built, including route A (2.78% < 6.25%). The "~1.0 bits/param at group 32 = 25% of a 4-bit model" figure silently assumed **asymmetric** scales we never implemented; symmetric is **0.500 bits = 12.5%**. What survives is the denominator argument (§3.1). |
| **"Knobs: group 35 / 14 layers / 24.37% / 37 layers"** | Pre-fix truncated knobs. Post-fix models use **36 / 15 / 24.375% / 38** (§1.1). |
| **"100% of the granularity spread is metadata — Contribution B measured directly"** | True by construction (codes don't depend on group size). A serializer validation, not evidence (§3.2). |
| **"33–43% of each pair's failures are unshared"** as a route effect | Two seeds of the *same* route already leave 20–31% unshared (Jaccard 0.69–0.80). The route effect is the excess over that floor: 1.2–2.0× (§1.3). |
| **"The A–C gap grows 1.9× under corruption"** | Difference-in-differences CI includes zero in all three seeds. Not established. |
| **"Calibration is a third, independent ranking"** | ECE = accuracy − confidence exactly: a uniform underconfidence offset, removable by one temperature; D's larger offset tracks the distillation recipe (§1.6). |
| **"Clean accuracy does not predict corruption robustness"** (general form) | Within DeiT-S the clean order holds on every corruption (§1.4). The flip requires an architecture change (§1.5). |
| "B gains +3.3 per 1000 steps vs A's +0.9; crossover ~step 4400" | Single pre-fix run, inconsistent step windows. Post-fix, 3 seeds, steps 1000→3000: **1.83 vs 0.55**; extrapolated crossover ~5,900 **[A]**. |

---

# Bugs found, and why they matter methodologically

Every one was invisible to accuracy alone. **Two were caught only because file size was measured
alongside it**, which is itself an argument for the reporting protocol.

| # | Bug | Symptom | Cost if missed |
|---|---|---|---|
| 1 | `torch.save` pickle framing around packed bytes | INT8 read 12.20 bits/param instead of 8.36 | 45% wrong sizes |
| 2 | Scales computed but never written to file | File smaller than its own accounting | Unloadable models |
| 3 | `webdataset` given a shell glob, not brace expansion | **Zero shards, no error** | Training on nothing |
| 4 | Three zombie `nohup` loops competing | Degrading it/s, silent SIGKILLs, 221 MB leaked shm | ~2 hours |
| 5 | `int(round(x))` in builder vs `int(x)` in trainer | 1.8% size drift | Weakened equal-bytes claim |
| 6 | Sensitivity ranked on **trained** weights in finalize | B read 68.99% instead of 73.25%; 2.4% size drift | 4.3 points on B |
| 7 | `.detach().cpu()` aliasing — `load_state_dict` overwrote the reference in place | B read 69.34%; 6σ vs a prior measurement | 3.9 points on B |
| 8 | `pip install imagecorruptions` pulled numpy 2.x | torch and cv2 both broke | Whole environment |
| 9 | `skimage.gaussian(multichannel=)` removed in ≥0.19 | `glass_blur` crashed the pool | 14 of 15 corruptions |
| 10 | Bug #5's truncated knobs copied into this document after the fix | All four knobs off by one for two weeks | A wrong methods section |

---

# Known limitations — state these before anyone asks

1. **Weight-only quantization.** Activations stay bfloat16. The 3-bit ViT papers quantize both
   (W3A3, W3A8), a harder problem. **Our 76.49% must never be compared against I&S-ViT's 55.78% or
   CPT-V's 61.16%.** Route-vs-route only.
2. **Round-to-nearest, not GPTQ/AWQ.** A shared quantizer keeps the comparison fair; calibration-based
   quantizers would lift all routes.
3. **20% of ImageNet, 3000 steps, nothing converged.** We measure recovery at a fixed budget.
4. **Sizes span 0.80%**, not an exact match.
5. **Sensitivity ranking is weight variance**, a crude proxy. Affects routes B and D only.
6. **Only D changes the architecture**, so every cross-route rank flip is confounded with
   architecture (§1.5). Within-architecture evidence is §1.4.
7. **14 of 15 corruptions, severity 3 only.** `glass_blur` excluded for runtime.
8. **Corruption evaluated on 5,000 images**, clean on 50,000.
9. **Per-image confidences were not saved**, so temperature-scaled ECE cannot be computed.
10. **Three seeds.** Enough for sign consistency and a seed floor; not for testing seed variance.
11. **No `.bin` file was ever decoded back and evaluated.** Accuracy comes from the training checkpoint
    with identical quantization simulated on the fly; the file is written by the same policy. They
    agree by construction (biases/norms differ only by FP16 rounding). **Update 2026-10-08:**
    `src/decode_bin.py` now decodes all 21 files and re-encodes them **byte-identically** (21/21);
    scoring the decoded files on ImageNet is pending a GPU run (`docs/GPU_RUN.md`).
12. **Latency on one CPU** (Apple M2). The uncompressed baseline's accuracy is published, not measured.

---

# What would settle the last open question

**Question:** is ternary's noise deficit *off* the same-architecture accuracy line (a route
property), or on a steep part of it (what any lower-accuracy DeiT-S would show)?

**Experiment:** score the five archived group-sweep models (`ChotaDeiT-A-g{32,64,128,256,512}-s0.bin`,
clean 74.68–76.68%) plus A and C under the same 14 corruptions. That gives a 7-point same-architecture
line per corruption; B's residual then has a real interval.

**Cost:** ~1 hour on the GPU box including regenerating the corruption cache **[A — estimate]**.
**Pre-registered reading:** if B's noise-family residual CI excludes zero, claim 1 becomes a route
property. If it includes zero, claim 1 is reported as "ternary loses robustness at the rate its clean
deficit predicts" — still a controlled measurement, and still worth the table.

---

# The claim, as it can be defended

> Compression comparisons are usually equalised by compression ratio or bit-width and scored on
> clean accuracy. We compress DeiT-Small four ways — uniform INT3, ternary with base-3 packing, INT4
> with structured pruning, and distillation into DeiT-Tiny — to the same **measured** file size
> (9.72–9.80 MB) under an identical 3000-step recovery budget (following Kuzmin et al.), with three
> seeds each. Route-induced disagreement exceeds a measured seed floor by 1.2–2.0×. Within a single
> architecture, ternarization's deficit is concentrated in noise corruptions (17.7–20.2 points
> against 5.9–6.4 on digital ones and 3.5–3.9 clean). Between a smaller architecture and ternary
> weights at the same file size, the preferred route flips with the corruption family, a crossing
> the ImageNet-C mean hides. Separately, across five ViT architectures, the nominal size formula used
> by ViT compression papers understates the real file by 1.5–53.5% at group 256, worsening with fewer bits and
> narrower models — an accounting LLM quantization already performs. We release a reporting protocol
> and all per-image outputs.

---

## Phase 4 — Kaggle T4 re-score (2026-10-08)

Run: `src/rescore.py` on Tesla T4, float16 autocast, every route model **decoded from its `.bin`**.
Raw: `evidence/rescore/rescore.json`, `cpu_latency_x86.json`, `kaggle_run.log`. 51 min, no failures.

**`.bin` files = reported models.** 12 route models × 50,000 clean images: 1,728 / 600,000 per-image
answers changed (0.288%), from float16 vs bfloat16 rounding. Seed floor is 6.3–9.6%, so rounding is
~20–30× below luck. Route means move ≤ 0.03 pts (A 76.50, B 72.43, C 75.70, D 71.41).

**Stated accuracy loss (P49).** DeiT-S FP32 in this pipeline: **79.72%** (published 79.8). DeiT-Tiny FP32: 72.02%.
Loss to route A (3-seed mean): **3.22 pts**.

**B/D flip replicates on regenerated corruptions** (new fixed-seed ImageNet-C, severity 3, 5,000 images;
not byte-identical to the original cache, so reported as a separate run):

| Seed | Clean B − D | McNemar p | Damaged D − B (mean of 14) | 95% bootstrap CI |
|---|---|---|---|---|
| 0 | +1.02 | 1.2e-09 | +3.83 | [+3.34, +4.33] |
| 1 | +0.92 | 3.3e-08 | +3.88 | [+3.40, +4.38] |
| 2 | +1.12 | 1.4e-11 | +3.71 | [+3.22, +4.20] |

Bonferroni bar (6 tests): p < 0.0083. All pass. Archived clean p-values were 5.1e-09, 2.6e-08, 1.2e-10.

**Calibration (O5).** Temperature fitted on even images, ECE on odd. D goes from worst (~12) to best
(~3.0) after one-number temperature scaling; A/B/C stay near 3.7–4.3. So D's calibration gap is a
uniform overconfidence, not a deeper defect.

**x86 CPU latency** (Intel Xeon @ 2.00 GHz, batch 1, median of 50, int8-dynamic, 1 thread):
A/B 98.95 ms · C 86.91 ms · D 38.52 ms · DeiT-S FP32 119.25 ms.
D vs A/B speed ratio depends on settings: int8 1 thr 2.57×, int8 4 thr 2.08×, fp32 1 thr 2.97×, fp32 4 thr 2.38×.
**Like-for-like with the M2 chart (fp32, 4 threads): 2.38× on Intel vs 2.23× on M2.** Intel int8 is 1.21× faster than fp32 for DeiT-S (1 thr); on M2 int8 was 4.6–5.7× slower — a qnnpack effect.

**O6 — is ternary off the same-architecture line? (pre-registered test, run 2026-10-08)**
Line: 7 same-shape models (5 group-sweep + A + C, per seed), probit axes, fitted on the same 5,000
images as the damage set; joint image bootstrap. Raw: `evidence/rescore/o6_line.json`.

| Seed | Noise residual | 95% CI (pre-registered) | 98.3% CI (Bonferroni, 3 seeds) | Plain-axis residual, 98.3% CI |
|---|---|---|---|---|
| 0 | −5.03 | [−8.82, −1.81] | [−9.47, −0.54] | −4.70 [−9.26, +0.70] |
| 1 | −4.56 | [−9.02, −0.61] | [−9.68, +0.79] | −4.13 [−9.65, +1.97] |
| 2 | −4.40 | [−8.58, −1.14] | [−9.27, +0.42] | −4.04 [−9.15, +1.63] |

**Reading: suggestive, not established.** Direction is the same in all 3 seeds and on both scales
(ternary 4–5 pts below the line on noise). It passes the pre-registered 95% rule, but fails a
Bonferroni-strict interval in 2 of 3 seeds and the plain-axis second derivation in 3 of 3.
**Extrapolation caveat:** ternary's clean accuracy (72.7–72.9% on these 5,000 images) is below every
reference point (74.7–76.8%), so the line is extended ~2 pts past its data.
Claim as written: "ternary falls 4–5 points below the same-architecture line on noise in every seed;
the gap is consistent but not statistically established after correction."

**Where D's robustness comes from (2026-11-13, `src/robustness_source.py`, Kaggle re-score data).**
Uncompressed DeiT-Tiny (timm weights) already beats ternary B by **+5.51 pts** on damaged photos (mean of
14); route D beats B by only **+3.81**. Distillation + 8-bit *lowered* DeiT-Tiny's damaged accuracy by
1.70 pts (clean −0.61). So D's lead over B under damage comes from the smaller network, not from
distillation. Caveat: DeiT-Tiny 32-bit is timm's pretrained model, not fine-tuned under our 3000-step budget.
