# The Four Routes — What They Are, Who Did Them First, Where We Actually Stand

**The honest summary in one line:** none of the four routes is new. Every one is an off-the-shelf
method someone published first. **That is the design, not a failure** — the contribution is the
*comparison*, and a comparison is only useful if it uses the methods people actually deploy.

---

## Route A — ChotaDeiT-A: uniform INT3

### What it does, plainly

Store every weight (the learned numbers inside the model) using **3 bits** instead of 32. Three bits
gives 8 possible values, so each weight is snapped to one of 8 rungs on a ladder.

The ladder's height is set per **group** (a block of consecutive weights sharing one **scale** — a
multiplier that converts the small integer back to a real number). Our search landed on **group size
36**, which is unusually small: more groups means more scales, so **1.22 MB of the 9.72 MB file is
scales alone (12.6%)**.

```
  original weights:   -0.0384  0.0121  -0.0902  0.0455 ...
  per group of 36:    find largest |value|, divide into 8 rungs
  stored:             3-bit code per weight + one FP16 scale per group
```

### Who did it first

| Paper | What they did | Result on DeiT-S |
|---|---|---|
| PTQ4ViT (2022) | First serious ViT post-training quantization | **Collapsed at 3-bit** |
| RepQ-ViT (ICCV 2023) | Scale reparameterization | 0.44% (ViT-S, W3A3) — collapsed |
| CPT-V (2022) | Contrastive PTQ | **61.16%** at W3A8 |
| I&S-ViT (2023) | Stability-focused PTQ | **55.78%** at W3A3 |
| Mix-QViT (CVPRW 2026) | Mixed precision over {2,3,4,5,6} bits | reports MB, not measured |

**Novelty of Route A: zero.** 3-bit ViT quantization is a crowded, mature subfield.

---

## Route B — ChotaDeiT-B: ternary + base-3 packing

### What it does, plainly

Every weight becomes one of exactly **three** values: −α, 0, or +α, where α is a per-channel scale.
A weight is set to 0 if it is small (below 0.7 × the average magnitude), +1 if clearly positive,
−1 if clearly negative.

**Ternary is not the same as 2-bit.** 2-bit gives 4 levels and costs 2.000 bits. Ternary gives 3
levels and costs log₂(3) = 1.585 bits in theory. **Base-3 packing** gets close: 5 ternary weights
fit in one byte, since 3⁵ = 243 ≤ 255. That is **1.600 bits per weight** exactly.

```
  five weights:  -1,  0,  0, +1, +1
  as base-3:     0 + 3*1 + 9*1 + 27*2 + 81*2  ->  one byte
```

Because ternary alone only reaches ~4.7 MB, the route has **5 MB of budget spare**. It spends it
promoting the **15 most sensitive layers to INT8**.

### Who did it first

| Paper | What they did |
|---|---|
| TWN (2016) | The 0.7 × mean\|w\| threshold rule we use, unchanged |
| TerViT (2022) | Ternary ViT; got Swin-S to 13.1 MB at >79% |
| BiViT / Bi-ViT (2022–24) | Binary ViTs — one level below this |
| **FTerViT (2026)** | **Fully ternary DeiT-III-S: 5.81 MB at 79.64%**, deployed on ESP32-S3 |
| TerEffic (2025), Kawamura et al. (2025) | The 5-weights-per-byte base-3 packing, exactly as used here |

**Novelty of Route B: zero.** The quantizer is TWN's, the packing is TerEffic's, the
fully-ternary-including-patch-embed idea is FTerViT's.

**One thing we did get independently:** our measured ternary zero-fraction is **43.1%**, against
FTerViT's reported 37.3%. Both near-uniform. Two independent quantizers agreeing that
accuracy-optimal ternarization produces **no exploitable sparsity** is a small but real
contribution — it kills the entropy-coding idea with our own number, not a borrowed one.

---

## Route C — ChotaDeiT-C: INT4 everywhere + structured pruning

### What it does, plainly

Two compressions stacked:

1. **Quantize everything to 4 bits** — including patch embedding and the classifier head, which
   standard tools leave at 16 bits.
2. **Structurally prune the MLP layers.** Each block's MLP expands 384 → 1536 → 384. We delete the
   **24.375% least important hidden channels** (ranked by the L2 norm of their incoming and outgoing
   weights), so 1536 becomes 1162 (`round(1536 × 0.75625)`).

**Structured, not unstructured.** Scattering zeros through a matrix does not shrink a file unless
you store it in a sparse format. Deleting whole channels genuinely makes the matrix smaller.

```
  MLP before:  [384 x 1536] -> [1536 x 384]
  MLP after:   [384 x 1162] -> [1162 x 384]     ~24% fewer numbers
```

### Who did it first

| Paper | What they did |
|---|---|
| UPop (ICML 2023) | Unified progressive structured pruning for VL transformers |
| MoPE-CLIP (CVPR 2024) | Module-wise pruning error metric, width-then-depth |
| **GETA (CVPR 2025)** | **Automatic joint structured pruning AND quantization**, evaluated on DeiT, ViT, Swin, PVT |
| Structural Pruning of Large VLMs (IJCV 2026) | Width vs depth pruning, recovery training, data efficiency |

**Novelty of Route C: zero.** GETA is exactly this combination, automated, and already at CVPR.

---

## Route D — ChotaDeiT-D: distil into DeiT-Tiny + mixed INT8/FP16

### What it does, plainly

Don't compress DeiT-Small at all. Take a model that is *already* small — **DeiT-Tiny** (5.7M
parameters, 192-wide instead of 384) — and train it to copy DeiT-Small's answers.

**Distillation** means the small "student" learns from the big "teacher's" full output distribution
("87% golden retriever, 9% labrador"), which carries far more information than the bare label.

DeiT-Tiny at 16 bits is 11.47 MB — just over budget. So this route only needs **light** compression:
**38 layers stay at FP16**, the rest go to INT8. The file is **82% FP16** — barely compressed at all.

**This is the "use a smaller model instead" arm.** If it wins, the whole compression exercise is
questionable, which is exactly why it must be in the comparison.

### Who did it first

| Paper | What they did |
|---|---|
| **DeiT (2021)** | Invented the distillation token; this is literally their method |
| MiniViT (CVPR 2022) | Weight sharing + distillation for ViTs |
| EfficientVLM (ACL 2023) | Distil-then-prune |

**Novelty of Route D: zero.** It is DeiT's own recipe.

---

## Where we actually have an edge

Not in any route. In the **comparison design**. Three things, all verified:

| Edge | Evidence it's unclaimed |
|---|---|
| **Equal MEASURED bytes, not equal ratios** | LampQ and MixFrag report average bit-width, no MB at all. Mix-QViT reports MB but computes it — its "11.4 MB" reconstructs from `params × bits ÷ 8` to within 0.03 MB. Our measured INT4 is 12.42–13.59 MB. |
| **Four different methods at one fixed budget** | Each paper above compares its method against *its own* baselines at *its own* bit-width. Nobody puts ternary, INT3, prune+quant and distillation side by side at one byte target. |
| ~~Equal training budget as a control~~ | **Conceded — Kuzmin et al. (NeurIPS 2023) use equal fine-tuning epochs. Cite them for it.** We use it; we did not invent it. |
| **More than one number reported** | Every paper above reports top-1 and stops. We add corruption robustness, calibration, and per-example failure agreement. |

---

## Where we DO NOT have an edge — say this before anyone asks

**1. Our quantizers are weaker than the specialists'.**
We use round-to-nearest plus basic QAT. I&S-ViT, RepQ-ViT and DopQ-ViT use calibration-based
rounding and handle the nasty post-Softmax and post-LayerNorm outliers properly. **A specialist
paper will get better accuracy at the same bit-width than we do.**

**2. We solve the EASIER problem, and the numbers are not comparable.**
This is the most important line in the document:

> The 3-bit papers above quantize **weights and activations** (W3A3, W3A8). We quantize **weights
> only** — activations stay in bfloat16. Weight-only is substantially easier.

So our smoke-test 74.55% must **never** be compared against I&S-ViT's 55.78% or CPT-V's 61.16%.
Different problem. Claiming we beat them would be dishonest and a reviewer would catch it instantly.

**3. Our sensitivity ranking is a cheap proxy.**
Routes B and D decide which layers to protect using weight variance. Proper methods use Hessian or
Fisher information. If a result turns on which layers were protected, this is the first thing to
attack.

**4. Group size 36 for Route A is an odd operating point.**
The binary search landed there; we did not design it. 12.6% of the file is scales. **Checked
(`results.md` §B.2):** a sweep over groups 32–512 shows it sits in the flat region — finer groups
buy 0.76 pts/MB there against 7.33 pts/MB at the coarse end.

**5. We train on 20% of ImageNet for 3000 steps.**
Absolute accuracy will sit below published DeiT-S numbers. Every comparison must be
**route-vs-route**, never route-vs-published.

---

## The three sentences to use in the paper

1. Four compression routes — uniform INT3, ternary with base-3 packing, INT4 with structured
   pruning, and distillation into a smaller backbone — all standard, all published, chosen
   deliberately because the contribution is the comparison rather than the methods.
2. All four are tuned to the same **measured serialized size** (9.718–9.796 MB, a 0.80% spread) and given an
   identical training budget (3000 steps), so neither size nor effort can explain any difference.
3. We then ask whether aggregate accuracy predicts robustness, calibration, and example-level
   behaviour — which none of the source papers report.
