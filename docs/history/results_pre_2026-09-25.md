# Results

**Model:** DeiT-Small, 22,050,664 parameters, 88.20 MB at FP32
**Budget:** 10 MB = **3.63 bits per parameter**
**Status:** Phases 1–3 complete. Every number below is measured or computed, never estimated.

---

## Provenance key

| Mark | Meaning |
|---|---|
| **[M]** | Measured — written to disk and read back, or scored on real images |
| **[C]** | Cross-checked — reproduced a second, independent way |
| **[A]** | Assumption or published figure we did not verify ourselves |

---


---

# Executive summary — three contributions, ordered by how hard they are to falsify

| # | Contribution | Depends on | Falsifiable by |
|---|---|---|---|
| **A** | **Compressed size is systematically misreported**, and the error scales on two axes: +4.5%→+29.1% as bits fall, +29.1%→+53.5% as the model shrinks | Measurement only | Nothing. These are file sizes on disk. |
| **B** | **Kuzmin et al. (NeurIPS 2023) state quantization's overhead is "a single scaling factor per tensor." That is false for group-wise quantization** — measured 0.120 bits/param at group 128, ~1.0 at group 32 (25% of a 4-bit model) | Measurement only | Nothing. |
| **C** | At equal measured size, clean accuracy does not predict corruption robustness, calibration, or which images fail. **The ranking inverts in 3/3 seeds**, clean p ≈ 1e-8 to 1e-10 at n=50,000 | 12 trained models (4 routes × 3 seeds) | **Tested. Survives** |

**A and B are the load-bearing contributions.** C is the demonstration of why they matter. A and B
are measurements of files on disk and cannot be overturned by training randomness. C rested on
trained models, so it was re-run across three seeds; **seed noise is 0.13–0.40 points, an order of
magnitude below every gap it depends on** (0.01–0.14 points clean, 0.13–0.26 corrupted).

**Why B matters most for novelty.** "We ran a careful comparison" is a common paper. "An explicit
claim in a NeurIPS paper is wrong, and here is the measurement" is not. Kuzmin's dismissal is
correct for per-tensor scaling and wrong for the group-wise scaling that AWQ, GPTQ and every
modern quantizer actually use.

# Contribution A — size is systematically misreported

## 1.1 Where the parameters are **[M][C]**

Cross-checked: hand calculation from the DeiT-S architecture matched the script to the parameter.

| Component | Params | Share | FP32 |
|---|---|---|---|
| MLP (12 blocks) | 14,178,816 | 64.3% | 56.72 MB |
| Attention (12 blocks) | 7,096,320 | 32.2% | 28.39 MB |
| Classifier head | 385,000 | 1.7% | 1.54 MB |
| Patch embedding | 295,296 | 1.3% | 1.18 MB |
| Position embed + CLS | 76,032 | 0.3% | 0.30 MB |
| LayerNorms | 19,200 | 0.1% | 0.08 MB |
| **Total** | **22,050,664** | **100%** | **88.20 MB** |

## 1.2 Measured file sizes vs the published formula **[M]**

Written to disk, measured with `os.path.getsize()`. Symmetric group-wise quantization, FP16
scales, `patch_embed`/`head`/norms/biases kept FP16 (standard tooling policy).

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

**Finding 1.** The formula's error **grows as bits fall**: +4.5% at INT8 → +29.1% at INT2. Scales
and un-quantized layers are fixed costs that do not shrink when the codes do. The formula is least
reliable exactly in the sub-4-bit regime a 10 MB target forces.

**Finding 2.** A "4-bit" model at group 32 is a **4.93-bit** model. Same label, 9.4% larger file.

**Finding 3.** INT4 cannot reach 10 MB in any configuration — 12.42 MB at best, **24% over**.

## 1.3 The un-quantized leftovers **[M]**

817,000 params (3.7% of the model) = 1.63 MB at FP16, constant across bit-widths.

| Format | Leftovers as % of file | Fully quantizing saves |
|---|---|---|
| INT8 | 7.0% | 2.9% |
| INT4 | 13.0% | 8.0% |
| INT3 | 16.4% | 11.0% |
| **INT2** | **22.4%** | **16.2%** |

Scale metadata: **0.120 bits/param at group 128** (symmetric, FP16 scales only). Asymmetric
quantization with zero-points roughly doubles it.

## 1.4 The error grows as models shrink too — DeiT-Tiny **[M]**

5,717,416 params, 22.87 MB FP32. Budget line 13.99 bits/param.

| | DeiT-Small | DeiT-Tiny |
|---|---|---|
| INT8 overhead | +4.5% | +8.0% |
| INT4 overhead | +12.7% | +23.2% |
| **INT2 overhead** | **+29.1%** | **+53.5%** |
| Leftovers at INT2 | 22.4% | **36.6%** |
| Leftover share of params | 3.7% | 7.2% |

**Why:** head, patch embedding and position embeddings scale with *width*; total params scale with
*width squared*. Halving the width doubles the un-quantizable share. **Two axes of failure for
`params × bits ÷ 8`: fewer bits and fewer parameters.**

**Baseline established [M]:** DeiT-Tiny INT8 = **6.22 MB** (g128). Published accuracy 72.2% **[A]**.

## 1.5 Literature verification **[C]**

| Paper | Budget definition | Reports MB? | Counts scales? |
|---|---|---|---|
| LampQ (2025) | *"average bit-width … mean across all parameters"*, `Σcᵢ Bᵢ ≤ b_t Σcᵢ` | No | No |
| MixFrag (2026) | `Σᵢ Σⱼ bⱼ xᵢ,ⱼ ≤ B_target`, reports W3/A3 | No | No |
| Mix-QViT (CVPRW 2026) | Eq. 28c `Σ_l Σ_u Σ_j α\|w\| bⱼ ≤ C_M` | **Yes — "11.4 MB (4-bit) DeiT-S"** | No |

**Mix-QViT's number is reconstructible, which makes this checkable rather than asserted [C]:**

```
  params × 4 bits ÷ 8                 = 11.03 MB
  + patch_embed & head kept at 8-bit  =  0.34 MB
  =                                     11.37 MB   vs reported 11.4 MB   (0.03 MB match)
```

Our measured 4-bit DeiT-S is **12.42–13.59 MB** — the formula understates by **9–19%**.

**Honest counterexample:** FTerViT reports "5.81 MB flash" as a genuine footprint including FP32
biases and statistics, validated on an ESP32-S3 partition. The pattern is *not* "the field is
wrong" — it is that **bit-allocation papers optimise a budget that is not the deployment budget.**

---


---

# Contribution B — a reporting protocol

Follows directly from Phase 1. What a compression paper should report, and why each item is not
optional.

| Report | Because |
|---|---|
| **Measured file size in bytes**, from `os.path.getsize()` | `params × bits ÷ 8` understates by 4.5–29.1% depending on bit-width, and 8.0–53.5% depending on model size |
| **Scale granularity** (per-tensor / per-channel / group size) | Per-tensor overhead is negligible; group 32 costs ~1.0 bits/param — **25% of a 4-bit model.** Kuzmin et al. assume the former and conclude quantization has no meaningful overhead |
| **Symmetric or asymmetric** | Asymmetric adds a zero-point per group, roughly doubling metadata |
| **Which layers are left un-quantized** | patch_embed + head + norms are 3.7% of DeiT-S parameters but **22.4% of an INT2 file**, and 36.6% for DeiT-Tiny |
| **The serialization format** | A `torch.save` container added 45% to our quantized files before we switched to raw binary |

**The protocol costs nothing to adopt** — every number is available at save time — and without it,
two papers reporting "4-bit" can differ by 23% in deployed bytes.

## B.2 The granularity trade, measured on trained models **[M][C]**

Route A's only knob is group size. Sweeping it at fixed INT3 — seed 0, identical 3000-step
budget, exported through the same path as ChotaDeiT-A:

| Group | Measured size | Clean top-1 | Cost of moving one step finer |
|---|---|---|---|
| 512 | 8.586 MB | 74.68% | — |
| 256 | 8.672 MB | 75.31% | +0.086 MB buys +0.63 pts = **7.33 pts/MB** |
| 128 | 8.843 MB | 75.84% | +0.171 MB buys +0.53 pts = 3.10 pts/MB |
| 64 | 9.186 MB | 76.16% | +0.343 MB buys +0.32 pts = 0.93 pts/MB |
| 32 | 9.870 MB | 76.68% | +0.684 MB buys +0.52 pts = **0.76 pts/MB** |

**Finding 4.** Granularity is a genuine size-accuracy trade — **1.56 pts/MB averaged over the whole
sweep** (+1.284 MB for +2.00 points) — but the marginal return **collapses 9.6×**, from 7.33 pts/MB
at the coarse end to 0.76 pts/MB at the fine end. Each halving of the group size doubles the
metadata bill while accuracy gains flatten. **Past roughly group 128 you are buying scales, not
accuracy.**

**Cross-check: the entire size difference is scale metadata and nothing else [C].** FP16 scales
cost exactly `16/g` bits per quantised parameter. With N_q = 21,910,664 (total params minus the
0.28 MB of FP16 leftovers in ChotaDeiT-A's accounting), that predicts every gap in the table:

| Step | Predicted from `16/g` alone | Measured | Error |
|---|---|---|---|
| 512 → 256 | +0.086 MB | +0.086 MB | 0.4 KB |
| 256 → 128 | +0.171 MB | +0.171 MB | 0.2 KB |
| 128 → 64 | +0.342 MB | +0.343 MB | 0.6 KB |
| 64 → 32 | +0.685 MB | +0.684 MB | **0.7 KB** |

Four gaps, worst error **0.0007 MB**. Codes are identical in size at every group; **100% of the
1.284 MB spread is metadata.** This is Contribution B measured directly rather than argued —
the term Kuzmin et al. dismiss is here the *only* term.

## B.3 Between routes there is no size-accuracy trade at all **[M]**

Ground-truth exported sizes, all 12 models hitting their build target to **0.000 MB**:

| Route | Measured size | Clean | Corrupted |
|---|---|---|---|
| **A** — INT3 g35 | **9.718 MB** (smallest) | **76.49%** (best) | **54.99%** (best) |
| D — Tiny+distill | 9.765 MB | 71.41% | 48.01% |
| B — ternary | 9.785 MB | 72.40% | 44.24% |
| C — INT4+prune | 9.796 MB (largest) | 75.68% | 53.49% |

**Size spread: 0.80% of the 9.766 MB mean.** Route A is simultaneously the smallest file, the most
accurate clean, and the most robust — it **strictly Pareto-dominates all three others on both
axes** (−0.047 to −0.078 MB, +0.81 to +5.08 clean points, +1.50 to +10.75 corrupted points).

**Put the two together:** inside a route, granularity is a trade you must price. Between routes, it
is not a trade — one route dominates. The field prices the first carefully and picks the second by
convention.

# Setup for Contribution C — the four models

## 2.1 Construction **[M]**

All four binary-searched to a 9.77 MB budget line — landing at 9.718 / 9.785 / 9.796 / 9.765 MB,
a 0.80% spread — then trained for **exactly 3000 optimizer steps** (the
control), weight-only QAT with a straight-through estimator, on 20% of ImageNet-1k (200 of 1024
WebDataset shards).

| Model | Route | Knob | codes / scales / fp16 (MB) | Max abs weight err |
|---|---|---|---|---|
| ChotaDeiT-A | Uniform INT3 | group 35 | 8.22 / 1.22 / 0.28 | 0.2218 |
| ChotaDeiT-B | Ternary + base-3 packing | 14 layers at INT8 | 9.34 / 0.16 / 0.28 | 1.4267 |
| ChotaDeiT-C | INT4 all + structured prune | prune 24.37% of MLP channels | 9.23 / 0.29 / 0.27 | 0.1188 |
| ChotaDeiT-D | DeiT-Tiny + distillation | 37 layers at FP16 | 1.70 / 0.03 / 8.03 | 0.0032 |

**Measured ternary zero-fraction: 43.1% [M].** FTerViT reports 37.3% **[A]**. Both near-uniform.
Entropy at 43.1% zeros = **1.555 bits**, against base-3 packing's 1.600 — **entropy coding would
gain 0.045 bits/weight.** Two independent quantizers agree: accuracy-optimal ternarization
produces **no exploitable sparsity.**

## 2.2 Final sizes and accuracy **[M][C]**

Accuracy on 25,600 held-out validation images. Cross-checked against the independent Phase 3
evaluation (5,000 images) — all four agree within **0.2 points**.

| Model | Measured size | Top-1 (25.6k) | Top-1 (Phase 3, 5k) | Δ |
|---|---|---|---|---|
| ChotaDeiT-A | 9.718 MB | 77.14% | 77.06% | −0.08 |
| ChotaDeiT-C | 9.796 MB | 75.91% | 75.76% | −0.15 |
| ChotaDeiT-B | 9.785 MB | 73.25% | 73.42% | +0.17 |
| ChotaDeiT-D | 9.765 MB | 71.41% | 71.52% | +0.11 |

**Size spread: 9.718–9.796 MB = 0.80% of the 9.766 MB mean.** Report as "all four within 0.8%",
never as exact equality. All 12 exports (4 routes × 3 seeds) hit their build target to **0.000 MB**;
the remaining 0.80% is the four routes' different targets, not drift.

> **Superseded numbers — do not reuse.** Before the `int(round())` vs `int()` fix, this table read
> A 9.753 / B 9.665 / C 9.796 / D 9.620 MB. Those pre-fix sizes are what produced the retracted
> "51.8 pts/MB" claim below. The accuracies in the 25.6k column predate the seed sweep; **§C.1 is
> the authority for accuracy.**

## 2.3 Recovery dynamics **[M]**

40-batch subset evaluations during training.

| Step | A | B | D |
|---|---|---|---|
| 500 | 75.19 | 66.88 | 70.64 |
| 1000 | 75.43 | 69.52 | — |
| 1500 | 75.75 | 70.44 | — |
| 2000 | 77.02 | 72.17 | — |
| 2500 | 77.27 | 73.39 | — |
| 3000 | 77.39 | 73.55 | 71.79 |

**Nothing converged at 3000 steps.** A gained +0.9 per 1000 steps; **B gained +3.3** — 3.7× faster,
from 8 points lower. Linear extrapolation puts the crossover near step 4400 **[A — not tested]**.

**This means the equal-step control is binding, and the ranking is budget-dependent.** State it as
a result, not an apology: *at a fixed 3000-step recovery budget, ternary starts lower and recovers
faster.*

---

# Contribution C — the demonstration: behaviour at equal size

**Setup [M]:** 4 routes × **3 training seeds** = 12 models. 5,000 validation images, 14 ImageNet-C
corruptions at severity 3, generated once and cached so every model is scored on byte-identical
inputs. `glass_blur` excluded (865 s per 1,000 images).

## C.1 Results across seeds **[M]**

Mean ± standard deviation over 3 seeds. **Clean accuracy is reported on the full 50,000-image
validation set; corruption on a 5,000-image subset** (corrupting 50,000 images needs ~15 GB per
worker and exceeds the 90 GB container limit). The corrupted bootstrap CIs exclude zero in every
seed regardless, so the asymmetry does not affect any verdict — but it is stated here rather than
buried.

| Route | Clean (n=50,000) | Corrupted (n=5,000) | Drop | ECE (n=50,000) |
|---|---|---|---|---|
| **A** — INT3 | **76.49 ± 0.05** | **54.99 ± 0.26** | 21.50 | **4.35 ± 0.02** |
| **C** — INT4+prune | 75.68 ± 0.01 | 53.49 ± 0.14 | 22.19 | 4.48 ± 0.15 |
| **B** — ternary | 72.40 ± 0.14 | 44.24 ± 0.13 | **28.16** | 5.41 ± 0.19 |
| **D** — Tiny+distill | 71.41 ± 0.13 | 48.01 ± 0.15 | 23.40 | **12.15 ± 0.10** |

**Seed noise: 0.01–0.14 points clean, 0.13–0.26 corrupted.** Every gap below is 7–37× that.

## C.2 THE RESULT — the ranking inverts **[M]**

| By clean top-1 | By corrupted top-1 |
|---|---|
| 1. A — 76.49 | 1. A — 54.99 |
| 2. C — 75.68 | 2. C — 53.49 |
| **3. B — 72.40** | **3. D — 48.01** |
| **4. D — 71.41** | **4. B — 44.24** |

| B vs D | Gap | McNemar p (per seed) | vs seed noise | Signs |
|---|---|---|---|---|
| Clean (n=50,000) | **+0.99** (B ahead) | **5.1e-09 / 2.6e-08 / 1.2e-10** | 7.3× | **3/3** |
| Corrupted (n=5,000) | **−3.77** (D ahead) | CI [−4.48,−3.02] etc., excludes 0 | 26.3× | **3/3** |

**Both directions significant, in every seed.** The clean p-values clear the Bonferroni threshold
of 4.2e-03 by **five to seven orders of magnitude**. An engineer selecting on published accuracy
picks B; under corruption D is 3.77 points better.

> **Note on framing.** This claim is about **direction and magnitude**, not about any comparison
> failing to reach significance. At n=5,000 the B-vs-D clean gap read p≈0.01 and did not clear
> correction; at n=50,000 it clears by orders of magnitude. **"Not significant" at one sample size
> is not a finding** — see the retraction below.

## C.3 Calibration — a third, independent ranking **[M]**

D's ECE is **12.15 ± 0.10 versus 4.35–4.48 for A and C — a 2.8× difference** on 50,000 images,
consistent across all three seeds, while D ranks *third* on clean accuracy. Distillation costs confidence
calibration, and top-1 does not show it.

## C.4 The routes fail on different images **[M]**

Failure-set Jaccard across all pairs and all seeds at n=50,000: **0.574 – 0.675**. Pairwise
disagreement **9.3 – 14.1%**. **33–43% of every pair's combined failures are not shared.**

## C.5 How to report these statistics — and how not to

Two separate levels, never merged:

| Level | Unit | Correct test | Report as |
|---|---|---|---|
| **Across images** (one model pair, fixed models) | 5,000 images | McNemar exact (paired); bootstrap over images for corruption | p-values and 95% CIs, Bonferroni-corrected for 12 tests (p < 0.0042) |
| **Across seeds** (model-to-model variation) | 3 seeds | **None** | **Mean ± std and sign consistency only** |

> **Never quote a p-value over seeds.** Three seeds is far too few to significance-test training
> variance. Report "+1.31 points, 7.8× seed noise, consistent in 3/3 seeds" — which is a
> statement about effect size and reproducibility, not a hypothesis test.

Corruption conditions are also **not independent** — 14 corruptions of the same 5,000 images are
not 70,000 samples. The image-level bootstrap resamples **images**, which handles this; a naive
two-proportion test inflates significance by roughly √14 ≈ 3.7×.

# Retracted claims

Recorded so they are not reintroduced.

| Claim | Why it was wrong |
|---|---|
| "Entropy coding recovers ~0.5 bits/weight on ternary ViTs" | Measured zero-fraction is 43.1% (ours) / 37.3% (FTerViT). Near-uniform. Gain is **0.045 bits**. |
| "A and C are tied on clean accuracy" | McNemar (paired) separates them, p = 4.12e-03. Pairing added power, exactly as warned. |
| "z = 11.5 / z = 11.7" | Treated 70,000 correlated samples as independent; unpaired test on paired data; no correction. Inflated ~3.7×. |
| "Base-3 packing achieves 1.585 bits" | 1.585 is the entropy **floor**. Base-3 packing achieves **1.600** (= 8/5). |
| **"Clean accuracy cannot distinguish A from C"** | True at n=5,000 (p = 0.05–0.26), **false at n=50,000** (p = 2.3e-09 / 6.0e-10 / 3.3e-08). A **power artifact**, retracted. What survives: the A–C gap grows **1.9×** under corruption (0.81 → 1.50 points) |
| **"Route A buys its accuracy at ~51.8 pts/MB over D"** | Built on **pre-fix sizes** (A 9.753 / D 9.620), which made A look *larger* than D. Ground truth: **A is 0.047 MB smaller than D** and better on both axes, so there is no size increase to divide by and **no trade to express as pts/MB at all**. A **Pareto-dominates** D. The 51.8 figure is not reproducible from any current number and should not be re-derived. |
| "Granularity's marginal return collapses 10×" | It collapses **9.64×** (7.33 → 0.76 pts/MB). Rounded up without computing. |
| "Group 35 is the finest group that fits under 10 MB" | Untested. g=34 and below also fit — 9.870 MB at g=32 is still under budget. Binary search stopped at 35; it was never shown to be the limit. |
| "The packing win is 24%" | FTerViT's 2.11 bits is a total including 0.30 MB of FP32 extras. Packing touches only the codes: **18.9%**. |

---

# Bugs found, and why they matter methodologically

Every one was invisible to accuracy alone. **Two were caught only because file size was measured
alongside it** — which is itself an argument for the reporting protocol.

| # | Bug | Symptom | Cost if missed |
|---|---|---|---|
| 1 | `torch.save` pickle framing around packed bytes | INT8 read 12.20 bits/param instead of 8.36 | 45% wrong sizes |
| 2 | Scales computed but never written to file | File smaller than its own accounting | Unloadable models |
| 3 | `webdataset` given a shell glob, not brace expansion | **Zero shards, no error** | Training on nothing |
| 4 | Three zombie `nohup` loops competing | Degrading it/s, silent SIGKILLs, 221 MB leaked shm | ~2 hours |
| 5 | `int(round(x))` in builder vs `int(x)` in trainer | 1.8% size drift | Weakened equal-bytes claim |
| 6 | Sensitivity ranked on **trained** weights in finalize | B read 68.99% instead of 73.25%; **2.4% size drift** | 4.3 points on B |
| 7 | `.detach().cpu()` aliasing — `load_state_dict` overwrote the reference in place | B read 69.34%; 6σ vs a prior measurement | 3.9 points on B |
| 8 | `pip install imagecorruptions` pulled numpy 2.x | torch and cv2 both broke | Whole environment |
| 9 | `skimage.gaussian(multichannel=)` removed in ≥0.19 | `glass_blur` crashed the pool | 14 of 15 corruptions |

---

# Known limitations — state these before anyone asks

1. **Weight-only quantization.** Activations stay bfloat16. The 3-bit ViT papers quantize both
   (W3A3, W3A8), a substantially harder problem. **Our 77.06% must never be compared against
   I&S-ViT's 55.78% or CPT-V's 61.16%.** Route-vs-route only.
2. **Round-to-nearest, not GPTQ/AWQ.** Calibration-based quantizers would lift all four routes
   roughly equally. A shared quantizer keeps the comparison fair — that is the point.
3. **20% of ImageNet, 3000 steps, nothing converged.** We measure recovery at a fixed budget, not
   final achievable accuracy.
4. **Sizes span 0.80%**, not an exact match. Every export hits its own target exactly; the
   spread is between the four routes' targets.
5. **Sensitivity ranking is weight variance**, a crude proxy. Affects routes B and D only.
6. **Route A's group size 35 was found by binary search, not designed.** 12.6% of its file is
   scales. The §B.2 sweep shows it sits deep in the flat region — **0.76 pts/MB at the fine end,
   9.6× worse than the coarse end** — so a coarser group would give up little accuracy for a much
   smaller file. Whether 35 is the *finest* group that fits was never tested (g=32 also fits).
7. **14 of 15 corruptions**, severity 3 only. `glass_blur` excluded for runtime.
8. **A vs C clears the corrected significance threshold by 1.1%.** Marginal, and should be
   reported as such.

---


# The claim, as it can be defended

> Compression papers specify budgets as compression ratios or bit-widths and report accuracy.
> We measure serialized size directly and show it is systematically understated — by 4.5% at
> INT8 rising to 29.1% at INT2, and worsening again as models shrink (8.0%→53.5% for DeiT-Tiny).
> The dominant term is group-wise scale metadata, which prior work
> ([Kuzmin et al., NeurIPS 2023](https://arxiv.org/abs/2307.02973)) explicitly dismisses as
> "a single scaling factor per tensor" — true for per-tensor scaling, false for the group-wise
> scaling used in practice. We propose a reporting protocol, and demonstrate its consequence:
> four compression routes equalised at the same **measured** size and given an identical recovery
> budget differ by up to 10.75 points in corruption robustness, by 2.8× in calibration error, and
> fail on substantially different images — while the best route is also the smallest file.

**Positioning against the nearest prior work:**

| Work | What it did | What we add |
|---|---|---|
| [Kuzmin et al., NeurIPS 2023](https://arxiv.org/abs/2307.02973) | Pruning vs quantization at equal **theoretical** ratio, equal fine-tuning epochs, accuracy only. **Cite for the equal-budget control — it is theirs.** | Measured bytes; four routes; their stated future work (prune+quant combination); structured rather than unstructured pruning; metrics beyond accuracy; **and a correction to their overhead claim** |
| [Compression vs corruption robustness in CNNs](https://arxiv.org/html/2512.24971) | Quantization/pruning/clustering vs CIFAR-C. **Not equalised at all.** Concludes compression *improves* robustness | Equalisation is the point — without it, a less-compressed model simply looks more robust. We control for measured size and find routes differ by up to 10.75 points |
| [Hooker et al., 2019](https://arxiv.org/abs/1911.05248) | Compressed vs dense diverge on identifiable examples | Route vs route at equal size. **Cite as support** |
| Kuzmin, Appendix I | *"QAT leads to learning completely new representations"* (qualitative) | The quantitative measurement: 35–42% unshared failures |
