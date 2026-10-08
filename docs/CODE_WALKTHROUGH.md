# Code Walkthrough — every script, in plain words

Read this top to bottom once. Then use it as a lookup when someone points at a file and says
"explain this". Every worked example below is **real output** from the project's own functions
(run 2026-10-08), not made up.

---

## 0. The map

```
 src/phase1_audit.py      measure real file size vs the formula          laptop
 src/build_routes.py      make 4 compressed files, each hitting a byte target
 src/train_routes.py      recover accuracy: 3000 training steps          GPU
 src/finalize.py          rescue a run that crashed before exporting     GPU
 src/evaluate_routes.py   accuracy on clean + corrupted images, ECE      GPU
 src/stats.py             significance tests for one seed
 src/aggregate_seeds.py   combine the 3 seeds: mean ± std
 src/offline_analysis.py  seed floor, per-corruption, calibration        laptop
 src/arch_audit_summary.py  re-derive every file size from shapes        laptop
```

Dependency chain: `phase1_audit` ← `build_routes` ← `train_routes` ← `evaluate_routes` / `finalize`.
Each later file **imports** helper functions from the earlier one, so the exact same quantization
code is used everywhere. That is deliberate: one implementation, no drift.

---

## 1. Five ideas you must understand first

### Idea 1 — Group-wise quantization (routes A and C)

A weight is a decimal number like `-0.40`. Storing it in 32 bits is expensive. Quantization stores
a small integer instead, plus one shared **scale** per group of weights.

Real example: six weights, 3 bits, group size 3 (`quantize_groupwise` in
[src/phase1_audit.py:95](../src/phase1_audit.py#L95)):

```
weights            0.12  -0.40   0.05  |  0.31  -0.08   0.22
                   ----- group 1 ----     ----- group 2 -----
scale = max|w| / 3  0.40/3 = 0.1333       0.31/3 = 0.1033
round(w / scale)    1     -3      0     |  3     -1      2      (range -4..3)
stored code (+4)    5      1      4     |  7      3      6      (0..7 fits 3 bits)
decoded = q*scale   0.1333 -0.3999 0.0  |  0.31  -0.1033 0.2067
```

- 3 bits allow 8 levels (−4…3). `qmax = 2**(bits-1) - 1 = 3`.
- Worst rounding error here: 0.05 (the `0.05 → 0.0` weight).
- **Storage:** 6 codes × 3 bits = 18 bits → `pack_codes` writes **3 bytes**. Plus 2 scales × 2 bytes
  (FP16). The scales are what the textbook formula `params × bits ÷ 8` forgets.
- Smaller group → more scales → better accuracy but bigger file. That is the whole group-size trade-off.

### Idea 2 — Ternary + base-3 packing (route B)

Each weight becomes −1, 0 or +1, times one scale per row (`quantize_ternary`,
[src/build_routes.py:38](../src/build_routes.py#L38)). The rule is from Ternary Weight Networks:

```
weights          0.12  -0.40   0.05   0.31  -0.08
threshold δ = 0.7 × mean|w| = 0.1344
ternary          0     -1      0     +1      0        (|w| ≤ δ  ->  0)
scale α = mean of |w| where non-zero = (0.40+0.31)/2 = 0.355
stored code      1      0      1      2      1        (−1→0, 0→1, +1→2)
```

**Base-3 packing** ([src/build_routes.py:57](../src/build_routes.py#L57)): five codes, each 0–2,
become one number `c0 + 3·c1 + 9·c2 + 27·c3 + 81·c4`. Here `1 + 0 + 9 + 54 + 81 = 145` → **one byte**.
Max possible is 242 < 256, so 5 weights always fit in 8 bits = **1.6 bits per weight**.

Ternary is **not** 2-bit: 2-bit has 4 levels, ternary 3. log₂(3) = 1.585 is the theoretical floor;
base-3 packing achieves 1.600.

### Idea 3 — The straight-through estimator (training through rounding)

Rounding has gradient zero almost everywhere, so a model can't learn through it. The trick
(`FakeQuant`, [src/train_routes.py:69](../src/train_routes.py#L69)):

- **Forward pass:** use the quantized weight (so the model *feels* the damage).
- **Backward pass:** pretend rounding was not there — pass the gradient straight through.

Real output: forward `[0.12, -0.40, 0.05] → [0.1333, -0.3999, 0.0]`, gradient `[1.0, 1.0, 1.0]`.
The full-precision "master" weights keep learning; quantization is re-applied every step.
This is **quantization-aware training (QAT)**.

### Idea 4 — Binary search to a byte target

Each route has one knob. `build_route` ([src/build_routes.py:172](../src/build_routes.py#L172))
guesses a knob, **writes the real file**, measures it with `os.path.getsize()`, and halves the search
range — up to 18 times — until the file is within 1% of 9.77 MB.

| Route | Knob | Bigger knob means |
|---|---|---|
| A | group size (searched 32–1024) | fewer scales → smaller file |
| B | how many sensitive layers get INT8 instead of ternary | bigger file |
| C | fraction of MLP channels deleted (0–60%) | smaller file |
| D | how many layers stay FP16 instead of INT8 | bigger file |

Found: A = 36, B = 15, C = 0.24375, D = 38. The 1% tolerance is why the four sizes differ slightly
(9.718–9.796 MB, a 0.80% spread).

### Idea 5 — Paired statistics

All models see the **same images**, so we compare them image by image:

- **McNemar's test** counts only the images where the two models disagree: "A right, B wrong" vs
  "A wrong, B right". If those two counts are very unequal, the models really differ.
- **Bootstrap**: resample the 5,000 images with replacement 10,000 times, recompute the gap each time;
  the middle 95% of results is the confidence interval.
- **Bonferroni**: running 12 tests, so each must pass p < 0.05/12 = 0.0042.
- **Never test over seeds**: only 3 seeds — report mean ± std instead.

---

## 2. Script by script

### `src/phase1_audit.py` — what does the model really weigh?

**Input:** a timm model name. **Output:** real files in `audit_out/`, a size table, optional JSON.

| Function | Line | What it does |
|---|---|---|
| `component_of` | [39](../src/phase1_audit.py#L39) | Labels each tensor: MLP, attention, head, patch embedding… |
| `is_quantizable` | [60](../src/phase1_audit.py#L60) | Only 2-D+ weight matrices get quantized; biases and norms stay FP16 |
| `pack_codes` | [74](../src/phase1_audit.py#L74) | Bit-packs integer codes (e.g. 8 three-bit codes → 3 bytes) |
| `quantize_groupwise` | [95](../src/phase1_audit.py#L95) | Idea 1. Returns codes, scales **and** error |
| `serialize_quantized` | [134](../src/phase1_audit.py#L134) | Writes `[4-byte header length][JSON layer list][raw bytes]` and measures it |

**Why a raw binary file, not `torch.save`?** `torch.save` wraps data in pickle/zip framing. An
early version reported INT8 at 12.20 bits/param instead of 8.36 — 45% wrong. Raw bytes mean
`getsize()` is the true deployable size.

`--random-init` (added 2026-09-25): skips downloading weights, because file size depends only on
tensor shapes. Proven byte-identical to pretrained on 16 files × 2 models.

Full line-by-line version: [phase1_audit_explained.md](phase1_audit_explained.md).

### `src/build_routes.py` — four files, one size

| Function | Line | What it does |
|---|---|---|
| `quantize_ternary` | [38](../src/build_routes.py#L38) | Idea 2, per output row |
| `pack_base3` | [57](../src/build_routes.py#L57) | 5 ternary codes per byte |
| `prune_mlp_channels` | [72](../src/build_routes.py#L72) | Scores each MLP hidden channel by the size (L2 norm) of its incoming + outgoing weights, keeps the top `1 − frac`. Deletes whole rows/columns so the matrix **really** shrinks: 1536 → 1162 channels |
| `write_model` | [101](../src/build_routes.py#L101) | The one file writer every route uses. Asks `policy(name, tensor)` what to do with each tensor: int / ternary / fp16 |
| `sensitivity_order` | [141](../src/build_routes.py#L141) | Ranks layers by weight variance — high variance = most hurt by coarse quantization = protect first. A cheap proxy (limitation stated in results.md) |
| `make_policy` | [152](../src/build_routes.py#L152) | Turns a route + knob into the per-tensor decision |
| `build_route` | [172](../src/build_routes.py#L172) | Idea 4, the binary search |

Route D starts from **DeiT-Tiny**, not DeiT-S (`base = student if r == "D"`, line 243).

**Bug fixed here (line 204–207):** the search ends on a fractional knob (e.g. 35.875). The builder
rounded it to 36 but the trainer truncated to 35 — different files, 1–2% size drift. Now the exact
integer is stored.

*Stale comment, for honesty:* the docstring (line 9) still says route A is "group 256". That was the
plan; the search actually found group 36.

### `src/train_routes.py` — recovering accuracy (GPU)

**Input:** route letter, seed, `models/routes.json`. **Output:** `.ckpt` (checkpoint), `.bin`
(compressed file), `logs/train_<route>_s<seed>.json`.

Flow of `main()` ([line 181](../src/train_routes.py#L181)):

1. Load pretrained DeiT-S (DeiT-Tiny for D). For C, prune first and rebuild smaller MLP layers
   (`_rebuild_mlp`, line 301).
2. Build the **same** policy as `build_routes` (sensitivity from **pretrained** weights).
3. `attach_fakequant` (line 82): replaces each Linear/Conv layer's forward so it uses
   `FakeQuant(weight)` — Idea 3.
4. Loop exactly `--steps` (3000) optimizer steps — **the control**. AdamW, lr 1e-4, cosine schedule,
   batch 256, gradient clipping 1.0, bfloat16.
5. Loss: cross-entropy with label smoothing 0.1. Route D adds a distillation term (lines 263–267):
   `0.5 × CE + 0.5 × KL(student/2 ‖ teacher/2) × 4` — the student copies the softened outputs
   (temperature 2) of a full-precision DeiT-S.
6. Every 500 steps: quick accuracy on 40 batches (the recovery table in results.md §1.7).
7. End: accuracy on 100 batches (25,600 images), then `write_model` exports the `.bin` with the same
   policy, and the size is logged.

GPU-box survival code at the top (lines 28–33): the machine had only 1 GB of shared memory, so data
workers pass tensors as files in local `/tmp`, images stay uint8 until they are on the GPU
(`to_gpu`, line 163), and only 2 workers run.

Data (`loaders`, line 111): ImageNet packed in tar shards (WebDataset) because the network disk read
only 1,494 small files/s. **Bug fixed here:** WebDataset does not understand `*` patterns —
`train-0[0-1]*.tar` silently matched zero files. Now an explicit file list is passed and an empty
list stops the run (line 126).

`--smoke`: 20 steps, used before every real run.

### `src/evaluate_routes.py` — the experiment (GPU)

**Stage 1, `build_cache`** (line 100): takes the first 5,000 validation images, applies each
ImageNet-C corruption **once**, saves `.npy` files. Every model then sees byte-identical images.
`_patch_skimage` (line 61) fixes a library break (`multichannel` argument removed in scikit-image 0.19).

**Stage 2, `load_route`** (line 144): rebuilds a trained model with its fake-quantization attached.
**Bug fixed here (lines 161–165):** `.detach().cpu()` on a tensor already on the CPU returns the
*same* tensor, so loading the checkpoint overwrote the "pretrained" copy, the sensitivity ranking
came from the wrong weights, and route B silently lost 3.9 points. `.clone()` fixes it.

**`score`** (line 180): returns, per image, the predicted class and the confidence (top softmax
probability).

**Stage 3, metrics:** `ece` (line 200) — 15 confidence bins, |accuracy − confidence| per bin, weighted
by bin size. `agreement` (line 211) — how often two models fail together (Jaccard).

Output: `logs/eval_s<seed>.json` with a **`correct` list of True/False for every image** — this is
what makes every later statistic recomputable on a laptop.

`--clean-only`: the 50,000-image clean run, because corrupting 50,000 images needed ~15 GB per worker.

### `src/finalize.py` — crash recovery (GPU)

Loads a checkpoint, evaluates it, exports the `.bin`. **Bug fixed here:** it once ranked layer
sensitivity on *trained* weights, protecting a different set of layers than training used → B read
68.99% instead of 73.25%. Now uses pretrained weights (lines 38–48).

### `src/stats.py` — significance, one seed

McNemar (clean), bootstrap over images (corrupted), Bonferroni over 12 tests, then a rank-inversion
check that reports "real" only if **both** directions are significant. Written after the first
analysis wrongly treated 14 corruptions × 5,000 images as 70,000 independent samples.

### `src/aggregate_seeds.py` — three seeds together

Mean ± std per route; gap ÷ pooled seed std ("how many times bigger than training noise");
whether the inversion holds in **every** seed. Prints `ROBUST` only if 3/3 seeds agree **and** both
gaps exceed 2× noise. Deliberately computes **no** p-value over seeds.

### `src/offline_analysis.py` — the laptop analyses (written 2026-09-25)

Reads only `evidence/logs/`. Its header lists the predictions written **before** it ran (P1 was
wrong). Sections:

1. **Seed floor** (line 81): disagreement between two seeds of the same route vs between routes.
2. **Per corruption** (line 143): 14 corruptions, McNemar per corruption, Bonferroni over 14.
3. **Within DeiT-S** (line 180): difference-in-differences — does B's gap grow from clean to corrupted?
4. **Calibration** (line 233): checks ECE == accuracy − mean confidence (→ uniform underconfidence).
5. **Fingerprint** (line 258): any corruption where the clean order reverses inside DeiT-S? (none)

### `src/arch_audit_summary.py` — size, derived a second way

`predicted_bytes` (line 25) recomputes each file from shapes: `ceil(n·bits/8)` code bytes +
`ceil(n/g) × 2` scale bytes, or `2n` for FP16 tensors. Asserts each measured file differs only by its
header. All 80 pass.

### `src/cpu_latency.py` — the P49 latency axis (laptop)

Builds each route's **architecture** with random weights (latency does not depend on weight values),
then times 50 single-image forward passes after 10 warm-ups and reports the median.
A and B decode to plain DeiT-S, C is DeiT-S with 1162-wide MLPs (`pruned_deit_s`), D is DeiT-Tiny.
Each is timed in FP32 and after `torch.ao.quantization.quantize_dynamic` (INT8 linear layers).
Key lesson it shows: our 3-bit/ternary files have no CPU kernel, so a smaller file is not a faster model.

---

## 3. Honest gaps in the code — know these before anyone finds them

1. **No `.bin` file was loaded back during the original runs.** *(Partly closed 2026-10-08: `src/decode_bin.py` decodes all 21 files and re-encodes them byte-identically; `src/rescore.py` scores the decoded files on a GPU — see `docs/GPU_RUN.md`.)* Accuracy is measured on the training checkpoint with
   fake-quantization applied on the fly; the `.bin` is written by `write_model` with the same
   policy and the same rounding math. They agree by construction, but a decode-and-compare
   round-trip test was never run. Small known differences: during evaluation, biases and norms stay
   full precision while the file stores them as FP16; an all-zero group gets scale 1 in the file and
   a tiny clamp in training. **What would settle it:** a 30-line decoder that rebuilds the weights from
   the `.bin` and runs the same images. Needs the images (GPU box or ImageNet download).
2. **Sensitivity is weight variance**, not Hessian/Fisher. Affects which layers B and D protect.
3. **Checkpoints (`.ckpt`) were not archived**, only the `.bin` files. Re-evaluating needs a decoder
   (point 1) or retraining.
4. Scripts in steps 2–5 use paths relative to the GPU box (`logs/`, `models/`, `data/`); they are
   kept as they ran, not edited after the fact.
5. Training-time accuracy uses 40-batch subsets (noisy); the headline numbers come from
   `evaluate_routes.py` on 50,000 images.

---

## 4. Explain it in 60 seconds (practise this out loud)

> "`phase1_audit` writes real compressed files and shows the textbook size formula is wrong, because
> it forgets the per-group scales. `build_routes` builds four compressed versions — 3-bit, ternary,
> 4-bit plus pruning, and a distilled smaller model — and binary-searches one knob each until every
> file is the same size on disk. `train_routes` recovers accuracy for exactly 3000 steps using a
> straight-through estimator, so the model trains through the rounding. `evaluate_routes` scores
> them on 50,000 clean images and 5,000 images under 14 corruptions, saving right/wrong for every
> image. `stats`, `aggregate_seeds` and `offline_analysis` then do paired tests on those arrays — and
> because they were saved, everything re-runs on my laptop with `./reproduce.sh`."

## 5. Code questions you may get

| Question | Answer |
|---|---|
| Why does the scale use `max|w| / qmax`? | Symmetric quantization: the largest weight in the group maps to the largest code, so nothing clips |
| Why store scales in FP16? | Half the bytes of FP32; we round the scale to FP16 *before* quantizing so the file and the math agree |
| Why per-row scales for ternary but per-group for INT? | Ternary follows the TWN paper (one α per output channel); INT follows GPTQ/AWQ group layout |
| Why prune MLP channels, not attention heads or tokens? | The MLP holds 64% of DeiT-S's parameters. Token pruning changes **zero** bytes on disk |
| Why AdamW, lr 1e-4? | Standard fine-tuning settings for DeiT; the same for every route, which is what matters |
| What is `label_smoothing=0.1`? | Trains towards 90% on the true class instead of 100%; reduces overconfidence (and explains the uniform underconfidence we measured) |
| How do you know the code didn't silently fail? | Size is measured two ways; results were checked against a second evaluation; smoke runs; nine bugs found this way are listed in results.md |
| What is the `* 4.0` in the distillation loss? | Temperature² (2² = 4), the standard Hinton correction so gradients keep their size when outputs are softened |
