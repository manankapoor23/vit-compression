# Squeezing an 88 MB Vision Transformer Under 10 MB

## The honest news first

**The squeeze is already published.** FTerViT (2026) explicitly targets "strict memory constraints
(below 10 MB) of microcontroller-class hardware" and hits it:

| Result | Size | ImageNet top-1 |
|---|---|---|
| DeiT-Small, FP32 (your 88 MB starting point) | 88 MB | 79.8% |
| **FTerViT-Small** (ternary, DeiT-III-S @224) | **5.81 MB** | **79.64%** |
| FTerViT W2A8, DeiT-III-S @384 | 6.09 MB | 82.43% (−2.42 pp vs FP32) |
| TerViT, Swin-S | 13.1 MB | >79% |

So "get a ViT under 10 MB with good accuracy" is a solved problem, at ~15× compression with almost no
top-1 loss. If you propose it as your contribution, you will be scooped on day one.

**But look at what those papers report: one number.** ImageNet top-1. That is the entire loss accounting.
**Your second clause — "document exactly what you lost" — is completely unclaimed, and it is the real
project.**

---

## The compression ladder for DeiT-Small (22M parameters)

Size = params × bits ÷ 8.

| Format | Bits/weight | Size | Verdict |
|---|---|---|---|
| FP32 | 32 | 88 MB | Starting point |
| FP16 | 16 | 44 MB | Free |
| INT8 | 8 | 22 MB | Effectively free |
| INT4 | 4 | **11 MB** | **Misses the target by 10%** |
| INT3 | 3 | 8.25 MB | Fits |
| Ternary / INT2 | 2 | 5.5 MB | Fits — FTerViT territory |
| Binary | 1 | 2.75 MB | Collapses |

**The sharpest fact here:** uniform 4-bit lands at 11 MB. It does not fit. Getting under 10 MB forces
you to either go below 4 bits, or mix precisions, or prune ~15% of weights first. There is no easy path.

**And the trap FTerViT identifies:** prior ternary ViTs left the patch embedding (the layer that reads
pixels), the LayerNorm parameters (normalisation constants), and the classifier head at full precision.
In DeiT-Small those are ~679k parameters — **2.7 MB at FP32, which is 34% of an 8 MB file.** At this scale
the parts nobody bothers to quantize dominate the footprint.

---

## What "documenting the loss" actually means

Top-1 accuracy is a single scalar over 50,000 images. It hides almost everything. Measure these instead:

| Dimension | What to test | Why it matters |
|---|---|---|
| **Robustness** | ImageNet-C (blur, noise, weather corruption) | Compressed models typically degrade far more here than top-1 suggests |
| **Out-of-distribution** | ImageNet-R, -A, -Sketch | Tests whether it learned concepts or memorised textures |
| **Calibration** | Expected Calibration Error — does 90% confidence mean 90% correct? | Compression makes models overconfident. Dangerous, and rarely reported |
| **Fine vs coarse** | Dog breeds vs "is it an animal" | Coarse ability survives; fine distinctions die first. Nobody quantifies where the line is |
| **Shape vs texture bias** | Cue-conflict images | Does compression push the model toward shortcut features? |
| **Feature quality** | Linear probe (train a simple classifier on frozen features) on transfer datasets | Whether the *representation* survived, not just the classifier |
| **Attention maps** | Compare before/after | Is it still looking at the object? |

**The dimension that connects to your bigger goal:** if this ViT is meant to become the **eye of a VLM**,
ImageNet top-1 is the wrong metric entirely. What matters is whether its features can still be mapped into
a language model by a connector. A ternary ViT can hold 79.6% top-1 while its features become useless for
grounding language. **Nobody has tested this.** That is a genuinely novel, directly useful finding, and it
is cheap to run: freeze the compressed encoder, train only a small projector, measure captioning or VQA
quality.

---

## The project, restated defensibly

> **Not:** "we compressed a ViT to under 10 MB." (Done — FTerViT, 5.81 MB, 79.64%.)
>
> **Instead:** "we compressed a ViT to under 10 MB by four different routes and measured, across eight
> dimensions, exactly what each route destroys — including whether the result is still usable as a VLM
> encoder."

Four routes to the same file size, so the comparison is fair:

| Route | How it reaches <10 MB |
|---|---|
| A | Uniform INT3 (8.25 MB) |
| B | Ternary, everything included (5.5 MB) |
| C | INT4 + 15% structured pruning (9.4 MB) |
| D | Distil into DeiT-Tiny, then INT8 (5 MB) |

**Route D is your mandatory baseline.** DeiT-Tiny at INT8 is 5 MB and scores 72.2% out of the box, with
no research required. Any compressed DeiT-Small at 10 MB that scores below 72.2% has failed, and a reviewer
will check this first.

**Your contribution is the loss map, not the compression ratio.** Compare all four at equal file size,
report every dimension in the table above, and state plainly which route wins for which use case. That
paper does not exist, and every future ViT-compression paper would cite it.
