# Squeezing a Vision Transformer Under 10 MB
### What a ViT is, how compression works, what's been done, and the one thing nobody has tried

---

## Scope lock — read this first

**The project:** get a Vision Transformer under 10 MB **on disk**, and document what that costs.

**The spine (your original framing, unchanged):** *squeeze an 88 MB DeiT-Small under 10 MB and document
exactly what you lost on the way.* Four routes to the **same serialized size**, equalised on **bytes, not
compression ratio** — the stronger constraint, because scale metadata makes ratios non-comparable.

| # | Route to ≤10 MB | Measured feasibility (Phase 1) |
|---|---|---|
| A | Uniform INT3 | **9.77 MB at group 256.** Group 128 clears by only 65 KB — use 256 or fully quantize |
| B | Ternary + base-3 packing | **~4.7 MB** projected. Needs its own quantizer: INT2 (4 levels) ≠ ternary (3 levels) |
| C | INT4 + structured pruning | INT4 alone is **12.42 MB, cannot fit**. Needs **full quantization AND ~14% pruning** |
| D | Distill into DeiT-Tiny, then INT8 | ~5 MB, 72.2% — the floor you must beat |

**Route C is harsher than it looked.** Fully-quantized INT4 measures 11.58 MB, so reaching 10 MB needs a
further 13.6% — i.e. prune ~14% of weights *on top of* quantizing patch_embed and head. Not a light trim.

**Controls and evaluation:** equal fine-tuning budget across routes (the confound — otherwise divergence
may come from the tuning, not the route). Evaluate on **ImageNet-C**, **calibration (ECE)**, and
**per-example failure agreement**.

**Methods contribution:** the packing fix, stated honestly as **2.110 → 1.600 bits/weight** (see Part 4,
Lever 5). Small, correct, not oversold.

**Secondary experiment, explicitly demoted:** the VLM-encoder probe — freeze the compressed model, train
only a projector, ask whether compression degrades *grounding-relevant* features faster than classification
accuracy. **Known limitation, state it up front:** [Cambrian-1](https://arxiv.org/abs/2406.16860)
(NeurIPS 2024) grafted 13–20+ encoders into LLMs via instruction tuning and found **class-supervised**
encoders are the weakest category, because language supervision is far richer than class labels. DeiT-S is
class-supervised, so its grounding baseline sits near the floor and the compression-induced drop is
measured against little dynamic range. That is why this is secondary, not the spine — and naming the
limitation yourself is viva credit, not a weakness.

**Who a reviewer will cite, and how you are covered:** *Prune-then-Quantize* (order affects ViT accuracy —
you extend to what it never measured); *Hooker 2019 PIE* + predictive multiplicity / Rashomon (per-example
divergence — you are the iso-bytes cut with added metrics: derivative but rigorous, and **you cite them
yourself** rather than getting caught); *Cambrian-1* (why the encoder probe is secondary).

> **CORRECTION (2026-09-11).** An earlier version of this plan proposed entropy coding as the headline.
> **That hypothesis is dead — see Part 4, Lever 5.** FTerViT's own paper reports its ternary weights are
> **37.3% zeros**, i.e. near-uniform across {−1, 0, +1}. **Two separate sub-0.02-bit gaps are involved and
> must not be conflated** — see the ladder in Part 4, Lever 5. In short: the *packing* win is large and real
> (2.110 → 1.600 bits/weight); the *entropy-coding-on-top* win is negligible (1.600 → 1.580). Keep them as
> two lines. Conflating them buries a genuine ~20% win under "entropy coding saves nothing."

**In scope:** anything that changes the number of bytes in the saved file — quantization, entropy coding,
pruning, low-rank, distillation — plus the loss map documenting what each costs.

**Out of scope, however interesting it looks:**

| Tempting detour | Why it's out |
|---|---|
| Visual token compression, KV cache, early exit | Changes the file by **zero bytes** |
| Compression **ordering** as a research question | Four papers since 2024, two in 2026. Crowded, and not about file size |
| "Does compression hurt robustness / calibration / fairness" as a standalone question | Saturated since Hooker 2019 (PIEs) |
| Anything benchmarked on CIFAR | DeiT-S is a 224×224 ImageNet model. Use ImageNet and ImageNet-C |

**Why the loss map is the result and not a detour:** run it on *your own* iso-byte models, anchored to the
10 MB target — never as a standalone "does compression hurt robustness" study, which is saturated. The
difference is the anchor: the budget is yours, the ordering question is someone else's. (arXiv 2603.18426
compares orderings on DeiT-Base at fixed compression *ratios* and evaluates **accuracy only** — no
ImageNet-C, no calibration, no per-example analysis. Cite it; don't inherit its frame.)

**Next action — Phase 1, week 1:** load DeiT-Small from `timm`, save it at FP32 / FP16 / INT8 / INT4,
measure the real files with `ls -l`, break the bytes down by component. No training, no GPU, no reading.

---

## Part 1 — What a Vision Transformer actually is

### The core idea

A **Vision Transformer (ViT)** looks at a picture the way a language model reads a sentence: by chopping
it into pieces and treating each piece like a word.

That's the whole trick. Before ViTs, computer vision used **CNNs** (Convolutional Neural Networks), which
slide small filters across an image looking for edges, then textures, then shapes. A ViT throws that away.
It cuts the image into a grid of squares, turns each square into a list of numbers, and then lets every
square **look at every other square** to figure out what's going on.

### The pipeline, step by step

```
  ① INPUT IMAGE          ② PATCHES              ③ TOKENS               ④ TRANSFORMER          ⑤ ANSWER
   224 × 224 × 3        14×14 grid of         197 vectors,           12 identical
                        16×16 squares         384 numbers each       blocks stacked

  ┌─────────────┐      ┌──┬──┬──┬──┐        ┌──────────────┐       ┌────────────┐
  │             │      │  │  │  │  │        │[CLS] ▪▪▪▪▪▪▪ │       │  Block 1   │
  │             │      ├──┼──┼──┼──┤        │ p1   ▪▪▪▪▪▪▪ │       │  Block 2   │      "Golden
  │    🐕       │ ───► │  │  │  │  │  ───►  │ p2   ▪▪▪▪▪▪▪ │ ────► │    ...     │ ───► Retriever"
  │             │      ├──┼──┼──┼──┤        │ ...          │       │  Block 12  │        (79.8%
  │             │      │  │  │  │  │        │ p196 ▪▪▪▪▪▪▪ │       └────────────┘        correct)
  └─────────────┘      └──┴──┴──┴──┘        └──────────────┘             │
                                                                          ▼
                        196 patches            + position info       read the [CLS]
                                               so it knows where     token, feed to a
                                               each patch came from  classifier "head"
```

**① The image** arrives as 224 × 224 pixels, 3 colour channels (red, green, blue).

**② Patches.** Cut it into a 14 × 14 grid of 16×16-pixel squares. That's **196 patches**. This is done by
the **patch embedding** layer — technically a single convolution with a 16×16 filter and a stride of 16,
which is just a fancy way of saying "chop and flatten."

**③ Tokens.** Each patch becomes a **token**: a vector (a list) of 384 numbers. Think of that vector as a
coordinate in a 384-dimensional space where similar-looking patches land near each other. Two things get
added here:

- A **[CLS] token** (classification token) — an extra, 197th token that doesn't come from the image at all.
  It's a blank scratchpad the model uses to accumulate a summary of the whole picture.
- **Position embeddings** — since the model has no built-in notion of "top left" or "bottom right", you
  add a learned vector to each token encoding where it came from.

**④ The transformer blocks.** Twelve identical blocks stacked in a row. Each block has two halves:

```
   ┌─────────────────────── ONE TRANSFORMER BLOCK ────────────────────────┐
   │                                                                       │
   │   tokens in                                                           │
   │       │                                                               │
   │       ├──────────────────────────────┐                                │
   │       ▼                              │                                │
   │  [ LayerNorm ]                       │  ← normalise (keep numbers     │
   │       ▼                              │     in a sane range)           │
   │  [ SELF-ATTENTION ]                  │  ← every patch looks at every  │
   │   6 heads, 0.59M params              │     other patch and decides    │
   │       ▼                              │     what's relevant            │
   │       ⊕ ◄────────────────────────────┘  ← residual: add the input     │
   │       │                                    back (stops the signal     │
   │       ├──────────────────────────────┐     from getting lost)         │
   │       ▼                              │                                │
   │  [ LayerNorm ]                       │                                │
   │       ▼                              │                                │
   │  [ MLP: 384 → 1536 → 384 ]           │  ← each token thinks about     │
   │   1.18M params                       │     itself, privately          │
   │       ▼                              │                                │
   │       ⊕ ◄────────────────────────────┘                                │
   │       │                                                               │
   │   tokens out                                                          │
   └───────────────────────────────────────────────────────────────────────┘
```

**Self-attention** is the part worth understanding. Each token produces three vectors: a **Query** ("what
am I looking for?"), a **Key** ("what do I offer?") and a **Value** ("here's my content"). Every token's
Query is compared against every other token's Key. High match = pay attention. The token then pulls in a
weighted blend of everyone's Values. That's how a patch of ear learns it's next to a patch of snout, and
the whole thing becomes "dog."

**Multi-head** means this happens 6 times in parallel with different learned projections, so different
heads can specialise (one tracks colour, one tracks position, one tracks texture).

The **MLP** (Multi-Layer Perceptron, also called the feed-forward network) is boring by comparison: two
matrix multiplications with a nonlinearity between them, applied to each token independently. It expands
384 → 1536 → 384. It is also, as you'll see in a moment, **where most of the model's weight lives.**

**⑤ The answer.** After 12 blocks, take the [CLS] token — now a summary of the image — and feed it to the
**classifier head**, a single matrix that maps 384 numbers to 1000 scores, one per ImageNet class.

### The model you're compressing: DeiT-Small

**DeiT** (Data-efficient image Transformer) is Facebook's 2021 recipe for training ViTs without Google's
proprietary 300-million-image dataset. DeiT-Small is the standard mid-size version:

| Property | Value |
|---|---|
| Embedding dimension | 384 |
| Blocks (depth) | 12 |
| Attention heads | 6 |
| Patch size | 16 × 16 |
| **Parameters** | **22.05 million** |
| **Size at FP32** (4 bytes each) | **88.2 MB** ← your starting point |
| ImageNet top-1 accuracy | 79.8% |

---

## Part 2 — Where the 88 MB actually lives

**This diagram should change your plan.** Before optimising anything, know what you're optimising.

```
  DeiT-Small: 22.05M parameters = 88.2 MB at FP32

  MLP layers (12 × 1.18M)   ████████████████████████████████  14.18M   64.3%   56.7 MB
  Attention  (12 × 0.59M)   ████████████████                   7.10M   32.2%   28.4 MB
  Classifier head           █                                  0.39M    1.7%    1.5 MB
  Patch embedding           █                                  0.30M    1.3%    1.2 MB
  Position embeddings       ▏                                  0.08M    0.3%    0.3 MB
  LayerNorms + [CLS]        ▏                                  0.02M    0.1%    0.1 MB
                            └────────────────────────────────┘
                             96.5% of the model is inside the
                             12 transformer blocks
```

Three consequences:

**(a) The MLP is the target.** Two-thirds of the file is the feed-forward layers. Any compression scheme
that doesn't hammer the MLP is rearranging deck chairs.

**(b) The "small" bits stop being small at 10 MB.** Patch embedding + position embeddings + head =
**756,000 parameters**. At FP32 that's **3.0 MB**. If you ternarise the blocks (down to ~5.3 MB) but leave
those three at full precision, they become **36% of your 8.3 MB file.** This is exactly the trap the
FTerViT paper (2026) identifies — earlier ternary ViT papers left patch embedding, LayerNorm and the head
at full precision, and at microcontroller scale those leftovers dominated everything.

**(c) The classifier head is free money if you don't need 1000 classes.** 385,000 parameters exist purely
to distinguish 1000 ImageNet categories. If your target task has 10 classes, that layer becomes 3,850
parameters and you've saved 1.5 MB at FP32 for nothing.

---

## Part 3 — The one number that defines your project

```
   Budget        10 MB  =  10,000,000 bytes  =  80,000,000 bits
   Parameters    22,050,664
   ───────────────────────────────────────────────────────────
   AVERAGE BUDGET  =  80,000,000 / 22,050,664  =  3.63 bits per parameter
```

**Every design decision reduces to: can I represent this weight in 3.63 bits on average?** Write that
number on your wall. Here's the ladder:

```
   88.2 MB │ FP32 (32 bits)      ████████████████████████████████████████  79.8% acc
   44.1 MB │ FP16 (16 bits)      ████████████████████                      79.8%
   22.1 MB │ INT8  (8 bits)      ██████████                                ~79.7%
   11.0 MB │ INT4  (4 bits)      █████                                     ~75.8%  ✗ MISSES BY 10%
  ─────────┼──────────────────────────────────────────────────── 10 MB LINE ──────
   10.0 MB │ 3.63 bits           ████▌                                     ← the target
    8.3 MB │ INT3  (3 bits)      ████                                      unstable
    5.5 MB │ ternary, 2-bit pack ██▌                                       79.6% (FTerViT)
    4.4 MB │ ternary, base-3     ██                                        same weights, tighter packing
    2.8 MB │ binary (1 bit)      █▍                                        collapses
```

> **The ladder above is the NOMINAL formula. Phase 1 measured the real files — below.**

### MEASURED (Phase 1, 2026-09-11) — the numbers that supersede everything above

Every row written to disk and read back with `os.path.getsize()`. Symmetric group-wise quantization,
FP16 scales, `patch_embed` / `head` / norms / biases kept FP16 (standard tooling policy).

| Format | Group | Nominal | **MEASURED** | bits/param | Overhead | Fits 10 MB? |
|---|---|---|---|---|---|---|
| INT8 | 256 | 22.05 | **23.04** | 8.36 | +4.5% | no |
| INT8 | 32 | 22.05 | **24.20** | 8.78 | +9.8% | no |
| INT4 | 256 | 11.03 | **12.42** | 4.51 | +12.7% | **no** |
| INT4 | 128 | 11.03 | **12.59** | 4.57 | +14.2% | **no** |
| INT4 | 32 | 11.03 | **13.59** | 4.93 | +23.2% | **no** |
| INT3 | 256 | 8.27 | **9.77** | 3.54 | +18.1% | yes |
| INT3 | 128 | 8.27 | **9.94** | 3.60 | +20.2% | yes, by 65 KB |
| INT3 | 32 | 8.27 | **10.93** | 3.97 | +32.2% | no |
| INT2 | 256 | 5.51 | **7.12** | 2.58 | +29.1% | yes |
| INT2 | 32 | 5.51 | **8.28** | 3.00 | +50.1% | yes |

**Four findings, all measured:**

1. **The formula's error grows as bits fall — 4.5% at INT8, 12.7% at INT4, 18.1% at INT3, 29.1% at INT2.**
   Scales and un-quantized layers are fixed costs; they don't shrink when the codes do. The formula is
   least reliable precisely in the sub-4-bit regime a 10 MB target forces you into. **This is the
   methodology argument, measured.**
2. **A "4-bit" model at group 32 is a 4.93-bit model.** Same label, 9.4% bigger file than group 256.
3. **INT4 cannot reach 10 MB in any configuration** — 12.42 MB at best, 24% over. Nominal said it missed
   by 10%; the real miss is 24%.
4. **INT3 at group 128 clears by 65 KB — a 0.6% margin.** Switching to *asymmetric* quantization (adding
   zero-points) roughly doubles the 0.120 bits/param scale cost and blows it. **Do not build Route A on
   that row**; use group 256, or fully quantize.

**The leftovers, measured** — `patch_embed` + `head` + norms + biases = 817,000 params (3.7% of the model),
1.63 MB at FP16, constant across bit-widths:

| Format | Leftovers as % of file | Fully-quantized saves |
|---|---|---|
| INT8 | 7.0% | 2.9% |
| INT4 | 13.0% | 8.0% |
| INT3 | 16.4% | 11.0% |
| **INT2** | **22.4%** | **16.2%** |

At FP32 these layers are a rounding error. **At INT2 they are nearly a quarter of the file.**

**Scale metadata: 0.120 bits/param at group 128** (symmetric, FP16 scales only). Asymmetric quantization
with zero-points roughly doubles it — quote 0.12–0.25 bits/param as the range.

### Evidence that the literature reports the nominal formula — verified, three papers

| Paper | What it reports | Scale/zero-point cost counted? |
|---|---|---|
| **LampQ** (2025) | *"The average bit-width of each model is computed as the mean across all parameters."* Constraint `Σ cᵢBᵢ ≤ b_t Σ cᵢ`. **No megabytes anywhere** | **No** |
| **MixFrag** (2026) | Constraint `Σᵢ Σⱼ bⱼ xᵢ,ⱼ ≤ B_target`. Reports W3/A3, W4/A4, W6/A6. **No megabytes** | **No** |
| **Mix-QViT** (CVPRW 2026) | Model-size constraint Eq. 28c is `Σ_l Σ_u Σ_j α |w| bⱼ ≤ C_M`. Reports **"11.4 MB (4-bit precision)" for DeiT-S** | **No** |

**The Mix-QViT number is reconstructible, which makes this checkable rather than asserted:**

```
  params x 4 bits / 8                        = 11.03 MB
  + patch_embed & head kept at 8-bit         =  0.34 MB
  =                                            11.37 MB   vs their reported 11.4 MB
```

A 0.03 MB match. **Your measured 4-bit DeiT-S is 12.42–13.59 MB — the formula understates by 9–19%.**

**The honest counterexample, state it yourself:** **FTerViT** reports "5.81 MB flash (2-bit packed
weights)" as a genuine footprint *including* FP32 biases and per-token statistics, validated by fitting an
actual ESP32-S3 flash partition. That number is sound. The pattern is not "the field is wrong" — it is:

> **Bit-allocation papers optimise a budget that is not the deployment budget. Hardware-deployment papers
> measure properly, and are the minority. Nobody has quantified the gap between the two.** Your audit does:
> 4.5% at INT8 rising to 29.1% at INT2.

---

## Part 4 — How you compress a ViT: the five levers

### Lever 1: Quantization — store each number in fewer bits

A trained weight is a float like `-0.03847291`. Storing that exactly costs 32 bits. But the model doesn't
need that precision. Quantization snaps every weight to one of a small number of allowed values:

```
   FP32 weights: a smooth continuous distribution
                      ▁▂▄▆████▆▄▂▁
        ─────────┼─────────┼─────────┼─────────
              -0.1        0.0       +0.1

   INT4: snap each weight to one of 16 rungs on a ladder
              │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
        ─────────┼─────────┼─────────┼─────────
                        16 levels = 4 bits

   Ternary: snap to one of THREE values {-α, 0, +α}
                        │       │       │
        ─────────┼─────────┼─────────┼─────────
                       -α       0      +α
                  3 levels = log₂(3) = 1.58 bits
```

Two flavours, and the difference matters enormously at low bit-widths:

| | **PTQ** (Post-Training Quantization) | **QAT** (Quantization-Aware Training) |
|---|---|---|
| What it is | Snap the weights of a finished model | Retrain while simulating the snapping |
| Cost | Minutes, ~1000 calibration images | Days of GPU time, full dataset |
| Works down to | ~4 bits for ViTs | ~1.58 bits (ternary) |
| **Verdict for you** | **Not enough alone** — PTQ at 4 bits is 11 MB | **Required** below 4 bits |

**Key fact about ViTs specifically:** they are *harder* to quantize than CNNs. Post-LayerNorm and
post-Softmax activations have extreme, spiky distributions with severe outliers that break naive
quantizers. This is why there's a whole literature (RepQ-ViT, PTQ4ViT, FIMA-Q, I&S-ViT) devoted to nothing
but quantizing ViTs properly.

### Lever 2: Pruning — delete weights entirely

**Unstructured pruning** zeroes out individual small weights scattered through the matrices. It gives high
sparsity but the file doesn't shrink unless you store it in a sparse format — **which pairs beautifully
with Lever 5.** **Structured pruning** removes whole units (an attention head, an MLP channel, a whole
block), giving a genuinely smaller dense model, but hurting accuracy more per parameter removed.

### Lever 3: Low-rank factorization — replace one big matrix with two small ones

An MLP weight matrix is 384 × 1536 = 590k numbers. Approximate it as the product of a 384 × r and an
r × 1536 matrix. At rank r = 96, that's 384×96 + 96×1536 = 184k numbers — a **3.2× cut**. Uses SVD
(Singular Value Decomposition), which finds the best such approximation. Costs accuracy; needs recovery
training.

### Lever 4: Distillation — train a small model to imitate a big one

Rather than compressing DeiT-Small, train a smaller architecture from scratch to copy DeiT-Small's outputs.
The student learns from the teacher's full probability distribution ("87% golden retriever, 9% labrador"),
which carries far more information than the bare label. **DeiT was itself built on this idea** — that's
what the "distillation token" in the original paper is.

### Lever 5: Entropy coding — compress the bit pattern itself, losslessly

**This is the lever your project should be built on, and Part 6 explains why.**

After quantization your weights are just symbols — e.g. `{-1, 0, +1}` for ternary. Storing each in 2 bits
is wasteful, because **the symbols aren't equally likely.** Ternarised weights are mostly zero:

```
   Naive packing: every symbol costs 2 bits regardless of frequency
        0 → 00      +1 → 01      -1 → 10      (11 wasted)
        Cost: 2.00 bits per weight, always

   Entropy coding: frequent symbols get short codes
        Suppose the distribution is  0: 70%,  +1: 15%,  -1: 15%
        0 → 0        (1 bit)
       +1 → 10       (2 bits)
       -1 → 11       (2 bits)
        Cost: 0.7×1 + 0.3×2 = 1.30 bits per weight

   Shannon entropy (the theoretical floor for this distribution):
        H = -(0.7·log₂0.7 + 0.15·log₂0.15 + 0.15·log₂0.15) = 1.18 bits
```

**The output is bit-for-bit identical after decoding. Accuracy loss is exactly zero.** This is not
approximation — it's the same principle as ZIP, applied to weights.

How much does that matter here? Run the numbers on DeiT-Small at ternary:

**...and here is where that reasoning fails, which is worth understanding before you try it.**

The size of the win depends entirely on how skewed the distribution actually is. So check the real number:

| Zero fraction | Entropy | File size for DeiT-S |
|---|---|---|
| 85% (hoped) | 0.76 bits | 2.09 MB |
| 70% | 1.18 bits | 3.26 MB |
| 50% | 1.50 bits | 4.13 MB |
| **37.3% — what FTerViT actually reports** | **1.580 bits** | **4.35 MB** |

**FTerViT's ternary weights are 37.3% zeros — near-uniform across the three values.** The uniform-ternary
floor is log₂(3) = 1.585 bits. Measured entropy is 1.580. **Entropy coding recovers 0.005 bits per weight.**

There is no skew to exploit, because ternarisation methods that *maximise accuracy* naturally produce a
balanced distribution — the zeros aren't free, they cost capacity. What looked like unexploited headroom is
an artifact of assuming sparsity that accuracy-optimal ternary training does not produce.

**What is real — and there are four distinct rungs here, not two:**

| Rung | bits/weight | DeiT-S | Achieved by |
|---|---|---|---|
| FTerViT as shipped | **2.110** | 5.82 MB | naive 2-bit packing |
| **Base-3 packing** | **1.600** | **4.41 MB** | 5 ternary weights per byte (3⁵ = 243 ≤ 256). Fixed-rate, off-the-shelf |
| Uniform-ternary floor log₂(3) | 1.585 | 4.37 MB | *unreachable by fixed-rate packing* — entropy coder only |
| Measured entropy @ 37.3% zeros | 1.580 | 4.35 MB | an actual entropy coder |

**Read the two gaps separately:**

- **Packing gap: 2.110 → 1.600 = 0.510 bits/weight.** Large, real, lossless, off-the-shelf (TerEffic,
  Kawamura et al. 2025). **This is your methods line.**
- **Entropy gap: 1.600 → 1.580 = 0.020 bits/weight.** Negligible. **This is the dead end.**

**Base-3 packing achieves 1.600, not 1.585.** 8/5 = 1.600 exactly; log₂(3) = 1.585 is the entropy *floor*,
reachable only by an entropy coder, which base-3 packing is not. Do not state 1.585 as an achieved value.

> **Settled — quote 19%, not 24%.** FTerViT's paper states weights are packed at 2 bits (4 per byte) and
> that biases and per-token statistics are kept at FP32. That decomposes its 5.81 MB exactly:
> `22.05M x 2 bits = 5.51 MB` of codes, leaving **0.30 MB of FP32 extras**. Base-3 packing replaces only
> the codes: `4.41 + 0.30 = 4.71 MB`, a **18.9% win**. Confirmed from the paper's own accounting, not
> inferred. **Never quote 24%** — that compares a total against a code-bits-only figure.

**Use it in your pipeline. Do not build a project on it.**

---

## Part 5 — What people have already done

### Group A: The ViT quantization community

They build better quantizers. They report accuracy at a bit-width.

| Work | Year | What it did |
|---|---|---|
| PTQ4ViT, RepQ-ViT | 2022–23 | Made post-training quantization work on ViTs at all, by handling the nasty post-Softmax and post-LayerNorm distributions |
| I&S-ViT, DopQ-ViT, FIMA-Q | 2023–25 | Pushed 4-bit PTQ higher. DopQ-ViT reports **75.84%** for 4-bit DeiT-S |
| TerViT, BiViT, Bi-ViT, Q-ViT | 2022–24 | Ternary and binary ViTs. TerViT gets Swin-S to 13.1 MB at >79% |
| Mix-QViT, LampQ, MixFrag | 2025–26 | **Mixed precision** — different bit-widths per layer, chosen by sensitivity, **under a target size budget**. MixFrag frames it as a knapsack problem. LampQ finds `qkv` and `fc1` need more bits than `proj` and `fc2` |
| **FTerViT** | **2026** | **The current champion for your exact target.** Fully ternary — including patch embedding, LayerNorm and head, which prior work left at full precision. **5.81 MB @ 79.64%.** Trained with distillation + a QAT recovery phase. Deployed on an ESP32-S3 microcontroller |

### Group B: The neural-network-coding community

They build better *coders*. They mostly ignore modern ViTs.

| Work | Year | What it did |
|---|---|---|
| Deep Compression | 2015 | The original: prune + quantize + Huffman code. 35–49× on AlexNet/VGG. **Huffman alone contributed 20–30%** |
| DeepCABAC / MPEG NNR | 2019–22 | An **ISO/IEC international standard** (15938-17) for compressing neural networks, using context-adaptive arithmetic coding borrowed from video codecs. Compressed VGG16 **63.6× with no accuracy loss**. Open-source implementation: **NNCodec** |
| EntroLLM | 2025 | Quantization + Huffman for edge LLMs. **65% storage saving over uint4.** States explicitly: vision transformers are not covered |

### Group C: Architectural tricks

| Work | Year | What it did |
|---|---|---|
| MiniViT | CVPR 2022 | Weight sharing across blocks, with small learned transformations to keep them from being identical |
| — | — | **Note:** naive cross-layer weight sharing, which works fine for BERT/ALBERT in NLP, causes **training instability and collapse in ViTs.** This is a known, unsolved difficulty |

---

## Part 6 — The gap: two communities that don't talk to each other

Here is the finding that should shape your entire project.

```
   ┌────────────────────────────────┐        ┌────────────────────────────────┐
   │   GROUP A: ViT QUANTIZATION    │        │   GROUP B: NEURAL NET CODING   │
   │   FTerViT, Mix-QViT, MixFrag   │        │   DeepCABAC, NNCodec, MPEG NNR │
   │   LampQ, RepQ-ViT              │        │   Deep Compression, EntroLLM   │
   ├────────────────────────────────┤        ├────────────────────────────────┤
   │ Optimises: the QUANTIZER       │        │ Optimises: the CODER           │
   │ Metric:    ImageNet top-1      │        │ Metric:    compression ratio   │
   │                                │        │                                │
   │ Reports size as:               │        │ Works on:                      │
   │   params × bits ÷ 8            │        │   VGG, ResNet, EfficientNet,   │
   │   ← the NOMINAL size           │        │   edge LLMs                    │
   │                                │        │   ← NOT modern ViTs at         │
   │ NEVER entropy-codes            │        │     sub-4-bit                  │
   └────────────────────────────────┘        └────────────────────────────────┘
                    ╲                                      ╱
                     ╲                                    ╱
                      ╲          ┌──────────────┐        ╱
                       ╲────────►│  NOBODY IS   │◄──────╱
                                 │     HERE     │
                                 └──────────────┘
                     Best ViT quantizer  ×  Best entropy coder
                     = the TRUE minimum file size for a given accuracy
```

**Why the gap exists, and why it's yours to take.** Group A avoids entropy coding for a good reason:
variable-length codes must be *decoded* before you can multiply with them, which complicates fast
inference. They optimise for a model that runs fast.

**You scoped your project to disk size, not inference speed.** That decision — which looked like a
simplification — hands you a lever the entire quantization literature has ruled out for reasons that don't
apply to you.

And it isn't even a real limitation in practice: FTerViT's own deployment target, the ESP32-S3, stores
weights in **flash** (8–16 MB) and streams them into a much smaller **RAM** (512 KB). Compressed-in-flash,
decode-per-layer is the natural design for that hardware. **You're not choosing a weaker use case — you're
choosing the one that matches the device.**

### Three concrete, unclaimed findings

**Gap 1 — The true size floor for a ViT.** FTerViT sits at 5.81 MB, which works out to **2.11 bits per
parameter**, where fixed-rate base-3 packing achieves **1.600** (→ **4.41 MB**, or ~4.71 MB once FTerViT's
scale overhead carries through). That ~19–24% is real and lossless, but it is off-the-shelf engineering,
not a finding. ~~Entropy coding on top~~ — **struck: it buys a further 0.020 bits/weight.** Treat
~4.4–4.7 MB as the practical floor for a ternary DeiT-S and move on.

**Gap 2 — The loss map.** Every paper in Group A reports one number: ImageNet top-1. That scalar hides
almost everything. The one study that looked deeper found something alarming: **the strongest FP32 model
(DeiT3) dropped 17% under 4-bit quantization and became one of the weakest 4-bit models.** Compression
**inverts the ranking.** Nobody has systematically documented what dies and in what order.

**Gap 3 — Which parent should you even start from?** This follows directly from Gap 2. If rankings invert
under compression, then "compress DeiT-Small" may simply be the wrong starting point. **At an equal final
file size of 10 MB**, does a heavily-compressed DeiT-Base beat a lightly-compressed DeiT-Small, or a
barely-touched DeiT-Tiny? Nobody has run that comparison. It's cheap, and the answer is genuinely unknown.

---

## Part 7 — The project

> **Establish the true minimum file size of an accurate Vision Transformer, by combining the best
> quantizer with the best entropy coder — and document precisely what is lost at every step down.**

```
  PHASE 1              PHASE 2              PHASE 3              PHASE 4
  AUDIT                CODE                 PUSH                 MAP THE LOSS
  ─────────            ─────────            ─────────            ─────────────
  Measure the          Entropy-code         Re-optimise the      Test the survivors
  real bytes           existing models      quantizer FOR        on 7 dimensions,
                       (lossless)           the coder            not just top-1
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  "Where does the      "How small does      "Can I get           "What exactly
   88 MB live, and      it get for free?"    smaller still?"      did I lose?"
   what do standard
   formats really
   weigh on disk?"

  1 week               2 weeks              4–6 weeks            3 weeks
  Low risk             Low risk             Medium risk          Low risk
  ↓                    ↓                    ↓                    ↓
  A table nobody       THE HEADLINE         The method           The citation
  has published        RESULT               contribution         magnet
```

### Phase 1 — Audit (week 1)

Load DeiT-Small from `timm`. Save it at FP32, FP16, INT8, INT4 (several group sizes). **Measure the
actual file on disk with `ls -l`**, not `params × bits`. Break every file down by component. Deliverable:
a table showing where the bytes really are, including scale overhead. Everybody skips this; it's the
foundation for everything after.

### Phase 2 — Pack it properly (week 2) — a methods line, not a result

Take **already-published quantized ViTs** — FTerViT ternary, Mix-QViT mixed-precision, plain INT4 — and
run their weights through **NNCodec** (open-source, ISO-standard, `pip`-installable). Measure the
bitstream. Verify decoded weights are bit-identical, so accuracy is provably unchanged.

**Report: accuracy vs. TRUE file size**, versus everyone else's accuracy vs. nominal size. If the theory
holds, you'll show that published models are 30–55% larger on disk than they need to be, and that
**INT4 quality (~75.8%) becomes reachable under 10 MB** — which uniform INT4 cannot do.

This phase costs almost nothing and cannot really fail: even a negative result ("ViT weights entropy-code
poorly, unlike LLM weights") is publishable and useful, because everyone currently assumes the opposite
without checking.

### Phase 3 — Co-design (weeks 4–9) — the method contribution

Phase 2 uses a quantizer that was never designed for a coder. Now close the loop.

Standard quantizers minimise **error**. Yours should minimise **error subject to a bit-rate constraint** —
exactly the rate–distortion trade-off that underpins every video codec. Concretely:

- **Skew the distribution deliberately.** A ternary quantizer with a higher zero-threshold produces more
  zeros → lower entropy → smaller file. There is an optimal threshold that trades a little accuracy for a
  lot of bits, and it is not the threshold that minimises reconstruction error.
- **Allocate bits by sensitivity *and* compressibility.** Mix-QViT allocates by sensitivity alone. But a
  layer that is insensitive *and* compresses well should get cut twice as hard. Nobody combines the two
  signals.
- **Exploit structure.** Weights in a ViT are not i.i.d. — neighbouring values in a matrix correlate.
  DeepCABAC's context modelling is built to exploit exactly this. Tune the context models to ViT weight
  statistics rather than the CNN statistics they were designed for.

### Phase 4 — The loss map (weeks 10–12) — the citation magnet

For every model you produce at ≤10 MB, report all seven:

| Dimension | Test | What it catches |
|---|---|---|
| Accuracy | ImageNet-1k top-1 | The baseline everyone reports |
| **Robustness** | ImageNet-C (blur, noise, weather) | Degrades far faster than top-1 suggests |
| **Out-of-distribution** | ImageNet-R, -A, -Sketch | Whether it learned concepts or memorised textures |
| **Calibration** | Expected Calibration Error | Compressed models become overconfident. Rarely reported, matters enormously in deployment |
| **Granularity** | Fine-grained (dog breeds) vs coarse (animal/vehicle) | Coarse survives, fine dies first — nobody has found the line |
| **Feature quality** | Linear probe on transfer datasets | Whether the *representation* survived, or just the classifier |
| **Attention** | Attention-map overlap with the FP32 model | Is it still looking at the object? |

**And the one that connects to your longer-term interest:** freeze the compressed encoder, train only a
small projector on top, and measure whether it can still be used as the **eye of a vision-language model**.
A ternary ViT can plausibly hold 79.6% ImageNet accuracy while its features become useless for grounding
language. **Nobody has tested this**, and it's a week of work.

### The baselines you must beat

| Baseline | Size | Accuracy | Why it's dangerous |
|---|---|---|---|
| **DeiT-Tiny at INT8** | **5.0 MB** | **72.2%** | Zero research required. If you land under 72.2%, you have failed |
| **FTerViT-Small** | **5.81 MB** | **79.64%** | The current champion. Your Phase 2 must beat it *on size at equal accuracy* |
| DeiT-Small INT4 (nominal) | 11.0 MB | ~75.8% | Doesn't fit — useful as the "what the default gets you" reference |

State these in your abstract. A reviewer will check them first, and pre-empting that is worth a lot.

---

## Part 8 — Honest risks

| Risk | Likelihood | What to do |
|---|---|---|
| ViT weights entropy-code worse than expected (the distribution is flatter than LLM weights) | Medium | This *is* the experiment. A negative result is still a finding — measure it in week 2, before committing |
| Someone publishes entropy-coded ViTs first | Low–medium | The gap has stood since 2015. Move fast on Phase 2 anyway; it's cheap |
| Reviewers say "entropy coding is just ZIP, not novel" | **High** | Pre-empt it: the novelty is Phase 3 (co-designing the quantizer *for* the coder), plus establishing the frontier. Cite Deep Compression as prior art yourself, prominently |
| Decode overhead makes it impractical | Medium | Report load-time cost honestly. Argue the flash-vs-RAM case for microcontrollers. Don't hide it |
| QAT in Phase 3 is too expensive for your compute | Medium | Fall back to PTQ + a short recovery fine-tune. Phases 1, 2, 4 need no heavy training at all |

**The framing that survives review:** not *"we made ViTs smaller"* — Group A has that covered. Instead:
***"the published size of every compressed ViT is wrong, because nobody counts the bits properly — here is
the real frontier, and here is what it costs you to get there."***

---

## Part 9 — Reading list, in order

| # | Work | Why |
|---|---|---|
| 1 | **FTerViT** (2026) | Your direct competitor and your Phase 2 input. Know it cold |
| 2 | **Deep Compression** (Han et al., 2015) | The original prune+quantize+Huffman. Your intellectual ancestor — cite it before a reviewer does |
| 3 | **DeepCABAC** + **NNCodec** (ISO/IEC 15938-17) | Your tool. Open source, standardised, `pip`-installable |
| 4 | **EntroLLM** (2025) | The same idea on LLMs; reports 65% saving over uint4. Your strongest evidence the headroom exists |
| 5 | **Mix-QViT** / **MixFrag** / **LampQ** | Mixed-precision ViT allocation — read to avoid re-inventing, and to reuse their sensitivity scores in Phase 3 |
| 6 | **RepQ-ViT** / **I&S-ViT** | Why ViTs are hard to quantize (Softmax and LayerNorm outliers) |
| 7 | **MiniViT** (CVPR 2022) | Weight sharing, and why it's unstable in ViTs |
| 8 | ViT quantization robustness / OOD study | The ranking-inversion finding behind Gaps 2 and 3 |
| 9 | The original **ViT** and **DeiT** papers | Foundations |

---

## Part 10 — Glossary

| Term | Meaning |
|---|---|
| **ViT** | Vision Transformer — chops an image into patches and treats them like words |
| **DeiT** | Data-efficient image Transformer — a ViT trained without a giant private dataset |
| **Patch** | A 16×16-pixel square of the image |
| **Token** | One patch, converted into a vector of numbers |
| **Embedding** | Turning something (a patch, a position) into a vector of numbers |
| **Patch embedding** | The first layer, which converts pixels into tokens |
| **Position embedding** | Added to each token so the model knows where the patch came from |
| **[CLS] token** | An extra, 197th token used as a scratchpad to summarise the whole image |
| **Self-attention** | Every token looks at every other token and decides what's relevant |
| **Query / Key / Value** | The three vectors each token produces so attention can work |
| **Head (attention)** | One parallel copy of attention. DeiT-Small has 6 |
| **Head (classifier)** | The final layer mapping the summary to 1000 class scores |
| **MLP / FFN** | Feed-forward network: two matrix multiplies per token. **64% of the parameters** |
| **LayerNorm** | Rescales numbers to keep them in a stable range |
| **Residual connection** | Adding a layer's input back to its output so signal isn't lost |
| **Parameter / weight** | One learned number. DeiT-Small has 22.05 million |
| **FP32 / FP16** | 32-bit and 16-bit floating-point number formats |
| **INT8 / INT4** | 8-bit and 4-bit whole-number formats |
| **Quantization** | Storing each weight in fewer bits by snapping it to a grid |
| **Ternary** | Only three allowed values: −α, 0, +α. Costs log₂(3) = 1.58 bits |
| **PTQ** | Post-Training Quantization — snap the weights of a finished model. Fast |
| **QAT** | Quantization-Aware Training — retrain while simulating the snapping. Slow, but needed below 4 bits |
| **Scale / zero-point** | Extra numbers stored so a quantized weight can be converted back |
| **Group size** | How many weights share one scale. Smaller = better accuracy, bigger file |
| **Calibration** | A small image sample used to choose quantization ranges |
| **Pruning** | Deleting weights. *Structured* removes whole units; *unstructured* scatters zeros |
| **Low-rank / SVD** | Replacing one big matrix with two small ones whose product approximates it |
| **Distillation** | Training a small student model to copy a big teacher's outputs |
| **Entropy** | How unpredictable a set of symbols is. Low entropy = compresses well |
| **Entropy coding** | Giving frequent symbols short codes. **Lossless** — output is bit-identical |
| **Huffman coding** | A simple, optimal-per-symbol entropy coder |
| **Arithmetic coding / CABAC** | A better entropy coder that can beat Huffman by coding whole sequences |
| **Rate–distortion** | The trade-off between file size (rate) and error (distortion). The core idea of all codecs |
| **Lossless** | Decoded output is bit-for-bit identical. Zero accuracy change, guaranteed |
| **ImageNet-C / -R / -A** | Test sets for corruption, artistic renditions, and hard natural images |
| **Calibration (of a model)** | Whether stated confidence matches actual accuracy |
| **Linear probe** | Freeze the model, train only a simple classifier on its features, to test feature quality |
| **Top-1 accuracy** | How often the model's single best guess is right |

---

## The five sentences that matter

1. **96.5% of DeiT-Small lives in the 12 transformer blocks, and 64% is the MLP layers** — but at a 10 MB
   target, the patch embedding, position embeddings and head stop being negligible and become a third of
   your file.
2. **Your budget is 3.63 bits per parameter.** Standard INT4 gives 11.03 MB and misses by 10%, so there is
   no lazy path.
3. **Compressing a ViT under 10 MB is already done** — FTerViT reaches 5.81 MB at 79.64% — so that alone
   is not a contribution.
4. **Entropy coding is a dead end here** — FTerViT's ternary weights are 37.3% zeros, near-uniform, with
   measured entropy 1.580 bits against a 1.585 floor, so an entropy coder buys 0.020 bits over packing.
   But take the **packing** win — 2.110 → **1.600** bits/weight, 5.82 → ~4.4–4.7 MB, lossless,
   off-the-shelf — and spend your research effort elsewhere. Two gaps, stated as two lines.
5. **And every paper in the field reports one number, ImageNet top-1**, while the single study that looked
   deeper found compression *inverts model rankings* — so the loss map is wide open, and it's the half of
   your project that will get cited.
