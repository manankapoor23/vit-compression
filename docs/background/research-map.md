# Shrinking a VLM's File Size to a Target, Without Losing Quality

**Last updated:** September 2026
**Scope:** reducing the **model's size on disk** (the actual .safetensors / .gguf file you download, and
the memory it occupies when loaded) down to a **target size you choose in advance**, while **keeping
output quality the same**.

**Explicitly out of scope:** making queries faster or cheaper (latency, FLOPs, visual token counts,
KV cache). Those are a different research problem, and mixing them in is what makes most papers in this
area muddy.

---

## 0. Why this scoping decision is the most important thing in the document

Roughly **70% of all "efficient VLM" papers are about visual token compression** (feeding the model fewer
image chunks). **Those papers are irrelevant to you.** Cutting tokens from 576 to 64 makes a query cheaper
but changes the model file by **exactly zero bytes** — the weights are untouched.

| Technique | Shrinks the file on disk? | Shrinks cost per query? |
|---|---|---|
| Visual token compression (fewer image chunks) | **No — 0 bytes** | Yes |
| KV-cache compression (scratch memory while answering) | **No — 0 bytes** | Yes |
| Early exit / layer skipping at runtime | **No — 0 bytes** | Yes |
| **Quantization** (fewer bits per number) | **Yes** | Somewhat |
| **Structured pruning** (deleting parts) | **Yes** | Yes |
| **Low-rank factorization** (replacing big tables with small ones) | **Yes** | Yes |
| **Distillation** (training a smaller copy) | **Yes** | Yes |
| **Lossless weight coding** (compressing the numbers themselves) | **Yes** | No |

**This immediately shrinks your literature to review by about 70%, and it removes the crowded field.**
It also means several of the "gaps" people cite in efficient-VLM surveys do not apply to you at all.

---

## 1. The levers that actually change file size

| Lever | What it does | Typical size cut | Quality cost |
|---|---|---|---|
| **Quantization** (store each weight in fewer bits) | 16-bit → 8 / 4 / 3 / 2 bits per number | **2× to 8×** | None at 8-bit; small at 4-bit; severe below 3-bit |
| **Lossless weight coding** (compress the bit-patterns, like a ZIP for weights) | ~16 bits → ~11 bits per number | **~1.4×** | **Exactly zero — bit-for-bit identical output** |
| **Low-rank factorization** (replace one big table of numbers with two small ones whose product approximates it) | Cuts a layer's parameter count | **1.2× to 2×** | Moderate; needs recovery training |
| **Structured pruning** (delete whole layers / attention heads / neurons) | Removes parameters outright | **1.2× to 2×** | Moderate to large; needs recovery training |
| **Distillation** (train a small student to copy a big teacher) | Builds a genuinely smaller model | **Any factor** | Depends entirely on training budget |

**The dominance fact you must internalise:** quantization (fewer bits per number) is not one lever among
five — it is *the* lever. Going 16-bit → 4-bit is a **4× file reduction with only a small quality drop**.
Pruning 25% of a model gives **1.33×** and usually needs retraining to recover. Quantization gets you
most of the way; everything else fights over the remainder.

---

## 2. The size ladder — what file sizes are actually reachable

Formula: **file size ≈ parameters × bits per parameter ÷ 8**, plus metadata overhead (see §3).

For **Qwen2.5-VL 3B** (~3.8B parameters — 3.8 billion learned numbers):

| Format | Bits per weight | File size | Quality |
|---|---|---|---|
| BF16 (the original download) | 16 | **7.6 GB** | 100% (reference point) |
| Lossless coded (DFloat11-style) | ~11 | **5.3 GB** | **100% — bit-identical** |
| INT8 (8-bit integers) | 8 | **3.8 GB** | ~99.5% — effectively free |
| INT4 (4-bit, standard AWQ/GPTQ) | 4 | **1.9 GB** | ~97–99% |
| INT3 | 3 | **1.4 GB** | ~90–96%, unstable |
| INT2 | 2 | **1.0 GB** | Collapses |

For **LLaVA-1.5 7B** (~7.1B parameters): BF16 = **14.2 GB**, INT8 = **7.1 GB**, INT4 = **3.6 GB**,
INT3 = **2.7 GB**.

**Read this ladder carefully, because it defines your research problem.** The rungs are far apart
(7.6 → 3.8 → 1.9 GB). If your target is **2.6 GB**, no standard format hits it. INT4 undershoots to 1.9 GB
(you wasted quality you didn't need to give up); INT8 overshoots to 3.8 GB (doesn't fit). **The space
between the rungs is where your contribution lives** — and hitting it requires *mixed precision* (different
parts of the model stored at different bit-widths), which is exactly the open problem in §6.

---

## 3. Three things that make the real file bigger than the formula says

Almost nobody reports these. Verifying them on your own models is a genuine, citable contribution and
takes an afternoon.

**(a) Quantization metadata.** Quantization doesn't just store 4-bit numbers — it stores a **scale** (a
multiplier that maps the small integer back to a real value) and often a **zero-point** (an offset) for
every **group** (a block of consecutive weights, typically 32, 64 or 128 of them).

> Overhead in bits per weight = (bits per scale + bits per zero-point) ÷ group size

| Group size | Overhead (FP16 scale + FP16 zero) | As a % of a 4-bit model |
|---|---|---|
| 128 | 0.25 bits/weight | **+6%** |
| 64 | 0.50 bits/weight | **+13%** |
| 32 | 1.00 bits/weight | **+25%** |

A "4-bit" model at group size 32 is really a 5-bit model on disk. Smaller groups give better quality —
so this is a genuine, unreported trade-off between quality and *actual file size*.

**(b) The embedding table** (the lookup table converting words to numbers). Qwen2.5's vocabulary is
151,936 words × 2,048 numbers each = **311M parameters** — 8% of a 3.8B model. Most quantization tools
leave it at 16-bit for quality reasons. At 16 bits that's **622 MB**; if everything else is at 4 bits
(1.75 GB), the embedding table is **26% of your entire file**. Nobody optimises it.

**(c) The vision encoder is often skipped entirely.** Standard tooling (llama.cpp / GGUF) frequently
quantizes the language model and leaves the **vision tower** (the eye) in FP16. On a 3B VLM, a 400M-param
encoder at 16 bits is **800 MB** — potentially **40% of a 2 GB file, left completely uncompressed.**
Check this on your target model before doing anything else. It may be your single biggest easy win.

---

## 4. What has already been done — quantization

| Work | Year | What it does | Covers VLMs? |
|---|---|---|---|
| GPTQ / AWQ | 2023– | The two standard 4-bit post-training (applied after training, no retraining) methods | Applied to VLMs, but designed for text |
| Q-VLM | 2024 | Post-training quantization tuned for large VLMs; accounts for how a rounding error in one layer feeds the next | **Yes** |
| MBQ | CVPR 2025 | **Modality-balanced** — image data and text data need different **calibration** (the sample data used to choose rounding rules) | **Yes** |
| Bi-VLM | 2025 | Pushes below 2 bits per weight; shows where quality collapses | **Yes** |
| Quantization best-practices study | 2026 | Systematic sweep of *which component* to squeeze | **Yes** — see below |
| **LLM-MQ** | 2024 | Assigns 2 / 3 / 4 bits per layer to hit a **target memory budget**, solved as integer programming (a maths optimiser that respects a hard limit) | **No — text only** |
| **GAMMA** | 2026 | **Bit allocation under an arbitrary budget.** Learns a sensitivity ranking once, reuses it for any target. Matches fixed-3-bit quality at **2.5-bit average**; per-budget adaptation drops from hours to minutes | **No — text only (Llama, Qwen 8B–32B)** |
| Layer bit-width allocation under a quality constraint | 2026 | Minimises size **subject to a maximum allowed quality drop** | **No — text only** |

**The finding your whole project rests on:**

> ViT (Vision Transformer — the eye) and LLM (Large Language Model — the brain) show **comparable
> importance to final performance, despite very different parameter counts.**
> — *Towards Understanding Best Practices for Quantization of VLMs* (2026)

In a 7B VLM, the eye is ~300M parameters (**4%** of the file) and the brain is ~6.7B (**96%**). If they
matter *equally* to accuracy, then **bytes are the wrong unit for spending your compression budget.** The
eye should be quantized gently (it's cheap to protect — 4% of the file) and the brain aggressively. A
uniform bit-width across the whole model is provably the wrong answer, and it is what every tool does
today.

**Note what this means against GAMMA.** GAMMA solves bit allocation across the *layers* of a text model,
where all layers are architecturally similar and were trained together. A VLM has **three structurally
different weight groups** (eye, translator, brain) that were trained separately, on different data, with
different normalisation — and whose importance-to-size ratios differ by roughly **25×**. GAMMA's
formulation never encounters this. That is a real, defensible reason for a VLM-specific method, not a
cosmetic "we applied X to Y" extension.

---

## 5. What has already been done — the other levers

### Low-rank factorization (VLM-specific work exists)

| Work | Year | What it does |
|---|---|---|
| **QSVD** | NeurIPS 2025 | SVD (Singular Value Decomposition — a maths technique that splits a big table of numbers into smaller ones) applied jointly to the Query/Key/Value weight matrices of VLMs, **plus** rank allocation across layers, **plus** quantization |
| **LASER** | 2026 | Loss-aware SVD + rank allocation. Tested on **Qwen2-VL 7B and SmolVLM 2B**, where plain SVD degrades severely |
| WSVD | 2026 | Weighted low-rank approximation |

**Important:** QSVD already combines low-rank with quantization for VLMs. If you propose "SVD + quantization
for VLMs" as your novelty, you are scooped. What QSVD and LASER do **not** do: target a **file size**. They
allocate rank by a loss criterion, not to hit 1.5 GB.

### Structured pruning (VLM-specific work exists, and it is recent)

| Work | Year | What it does |
|---|---|---|
| **Structural Pruning of Large VLMs** | IJCV 2026 | The comprehensive study: layerwise vs widthwise pruning, recovery training, data efficiency. **Finding: width pruning beats depth pruning at equal parameter count** (depth pruning cuts latency more, but you don't care about latency) |
| SlimVLM | 2026 | Sensitivity-aware dynamic structured pruning |
| MoPE-CLIP | 2024 | Module-wise pruning error metric; width-then-depth + distillation |
| InfoPrune | 2025 | Information-bottleneck head pruning + low-rank FFN |
| UPop / EfficientVLM | 2023 | The early work: unified progressive pruning; distill-then-prune with different rates per modality |

**The IJCV 2026 study is required reading and partly bad news** — it covers a lot of the obvious pruning
ground. Read it before committing to a pruning-based project. **The good news:** it targets pruning
*ratios*, not file sizes, and it prunes the language backbone rather than allocating across components.

### Lossless weight coding (zero quality loss — and untested on VLMs)

| Work | Year | Result |
|---|---|---|
| **DFloat11** | NeurIPS 2025 | Huffman-codes the exponent bits of BF16 weights (frequent exponents get shorter codes). 16 bits → ~11 bits. **30% smaller, bit-for-bit identical output.** Validated on Llama 3.3, Qwen 3, Mistral 3, FLUX.1, Qwen-Image |
| ZipNN | 2024 | Similar idea, for storage and download only |
| NeuZip | 2024 | Entropy coding with layer-wise decompression |
| Shannon-bound coding | 2026 | Within 0.01–0.1 bits of the theoretical limit; 11× faster than NeuZip/DFloat11 |
| Unweight (Cloudflare) | 2026 | Lossless MLP weight compression |

**This category matters more to you than to anyone else**, because your constraint is "keep the quality
the same" and this is the only family that *guarantees* it. **And no published work has evaluated it on a
VLM.** DFloat11 works because BF16 exponents in LLMs are low-entropy (predictable). Whether a **ViT
encoder's** weights have the same statistics is an open empirical question — vision transformers are
trained differently, with different normalisation and initialisation. They might compress better. They
might compress worse. **Nobody has checked, and checking is a week of work.**

---

## 6. The gaps, re-ranked for a disk-size project

| # | Gap | How open | Cost | Scoop risk |
|---|---|---|---|---|
| 1 | **Component-aware bit allocation for VLMs under a file-size budget** — how many bits for eye vs translator vs brain to hit exactly N GB | **Wide open** | Medium | Low–medium |
| 2 | **Report and optimise real file size**, including scale metadata, embeddings, and the un-quantized vision tower | **Wide open** | **Low** | Low |
| 3 | **Lossless coding on VLMs** — do ViT weights compress like LLM weights? | **Wide open** | **Low** | Medium |
| 4 | **Quality-constrained formulation**: minimise size subject to ≤X% drop on *every* ability, not on the average | Open | Low–medium | Medium |
| 5 | **Combining all four levers under one size budget** (bits + rank + pruning + lossless coding) | Open | High | Low |
| 6 | Compressing the **embedding table** in a VLM | Mostly open | Low | Low |
| 7 | Structured pruning of VLMs | **Mostly closed** (IJCV 2026) | High | High |
| 8 | Low-rank (SVD) for VLMs | **Closed** (QSVD, LASER) | Medium | **Very high** |
| 9 | Applying AWQ/GPTQ to a new VLM | **Closed** | Low | **Very high** |
| 10 | Anything about visual tokens | **Irrelevant to you** | — | — |

### The three that are genuinely yours

**Gap 1 — component-aware bit allocation to a file-size target.**
Everything needed for this exists in pieces and nobody has assembled it: GAMMA proves budget-targeted bit
allocation works (text-only, average-bit-width budget). The 2026 quantization study proves the eye and
brain have *wildly* different importance-per-byte. QSVD proves the levers combine for VLMs. **Nobody has
written the allocator that takes "1.5 GB" and outputs a per-component bit assignment for a VLM.** This is
your core method, and the argument for why it must be VLM-specific is strong (§4).

**Gap 2 — measure the actual file.**
Every paper reports average bit-width or parameter count. **Report gigabytes.** Include quantization scale
overhead (+6% to +25%, §3a), the embedding table (up to 26% of a 4-bit file, §3b), and the vision tower
that standard tooling leaves at FP16 (up to 40% of the file, §3c). This is cheap, unglamorous, and
immediately useful — and it is the empirical foundation Gap 1 needs, so **do it first regardless.**

**Gap 3 — lossless coding on VLMs.**
DFloat11 gives 30% off with *provably identical output*. It has never been run on a VLM. Measure the
exponent entropy (how predictable the bit patterns are) of a ViT encoder versus an LLM. If they differ,
that is a publishable finding on its own and it changes how the size ladder in §2 is built. **One week
of work, near-zero risk of being wrong.**

### One thing to be honest with yourself about

**The remaining headroom is smaller than it looks.** 16-bit → 4-bit already gives you 4×, and that is a
solved, one-command operation. Going 4-bit → 3-bit gains another 25%. So a paper claiming "we made VLMs
much smaller" will not be believed, because the big win is already available to everyone.

**Frame your contribution correctly:** it is not "smaller than before." It is **"exactly the size you
specified, with the quality loss placed where it hurts least."** That is a deployment contribution — it
matters because real hardware has hard limits (8 GB of phone RAM, a 16 GB GPU, a 2 GB app-store cap) and
current tools can only offer you the coarse rungs in §2. Say that plainly in your abstract.

---

## 7. The project

> **Given a hard file-size target for a VLM, decide how many bits each component (eye / translator /
> brain / embeddings) should be stored in, so that the resulting file hits the target exactly and quality
> stays within a stated bound.**

| Phase | What you do | Output | Effort |
|---|---|---|---|
| **1. Audit** | Take Qwen2.5-VL 3B and LLaVA-1.5 7B. Break the file down byte by byte: eye, translator, brain, embeddings, scale metadata. Report the real GB at every standard format. | The measurement nobody publishes (Gap 2) | **Low** |
| **2. Sensitivity** | Quantize **one component at a time** to 8 / 4 / 3 / 2 bits, hold the rest at 16. Measure quality per ability (OCR, counting, spatial, hallucination) — not just the average. | The importance-per-byte curve for each component (Gap 1's input) | **Low** |
| **3. Lossless check** | Run DFloat11-style exponent-entropy analysis on the eye and the brain separately. Do they compress differently? | A standalone finding (Gap 3) | **Low** |
| **4. Allocate** | Solve: minimise quality loss subject to total bytes ≤ target. Integer programming (like LLM-MQ) is enough — you have ~4 components, not 300 layers, so the search space is tiny. | The method | **Medium** |
| **5. Prove** | Hit three awkward targets that no standard format reaches (e.g. 2.6 GB, 1.5 GB, 900 MB). Compare against: uniform quantization rounded to the nearest rung, and natively-small models at the same file size. | The paper | **Medium** |

**Note how tractable Phase 4 is.** Text-model bit allocation is hard because there are hundreds of layers.
You have **four component groups**. You could nearly solve it by exhaustive search. The difficulty in your
project is not the optimisation — it is Phase 2, measuring sensitivity correctly. Put your effort there.

**Non-obvious decisions to get right:**

| Decision | Do this | Not this |
|---|---|---|
| Budget unit | **Measured gigabytes of the actual saved file** | Average bit-width, or parameter count |
| Quality definition | Worst-case across abilities (OCR, counting, hallucination) | The average benchmark score |
| Baselines | Uniform quantization at the nearest rung, **and** natively-small models (SmolVLM, PaliGemma) at the same file size | Only the uncompressed original |
| Include the eye | **Always quantize and report it** | Silently leaving it at FP16 like the standard tools do |
| Report | Bytes per component, before and after | One aggregate number |

---

## 8. What to avoid

| Don't | Why |
|---|---|
| Anything involving visual token counts | Doesn't change the file size by one byte |
| SVD / low-rank for VLMs as your main novelty | QSVD (NeurIPS 2025) and LASER (2026) already own this |
| Structured pruning of VLMs as your main novelty | The IJCV 2026 study covers the ground comprehensively |
| Applying AWQ to a new VLM and reporting numbers | Engineering, not research |
| Claiming "much smaller than before" | 4-bit already gives 4× for free. Claim **precision of targeting**, not magnitude |
| Reporting average benchmark scores only | Averages hide OCR collapse — which is exactly what compression breaks first |

---

## 9. Reading list, in order

| # | Work | Why |
|---|---|---|
| 1 | *Towards Understanding Best Practices for Quantization of VLMs* (2026) | The eye-vs-brain importance finding your project rests on |
| 2 | **GAMMA: Global Bit Allocation under Arbitrary Budgets** (2026) | The text-only version of your method. Read it closely and position against it |
| 3 | LLM-MQ (2024) | The simplest budget-targeted formulation; a good starting template |
| 4 | **DFloat11** (NeurIPS 2025) | Lossless, 30%, bit-identical — your zero-quality-loss lever |
| 5 | **QSVD** (NeurIPS 2025) + **LASER** (2026) | VLM-specific weight compression. Know exactly what they claim so you don't repeat it |
| 6 | **Structural Pruning of Large VLMs** (IJCV 2026) | The pruning ground already covered |
| 7 | MBQ (CVPR 2025) | Why image and text data need different calibration |
| 8 | Bi-VLM (2025) | Where quality collapses at ultra-low bits |
| 9 | Layer bit-width allocation under a quality-degradation constraint (2026) | The quality-constrained formulation, for text |
| 10 | *Scaling down, Powering up* — small VLM survey | Your natively-small baselines |

**Skip entirely:** the multimodal token compression survey, VTC-Bench, SmartVL, FastV, VisionZip,
LLaVA-Mini, SparseVLM, and every other token-compression paper. They are excellent work on a different
problem.

---

## 10. Glossary

| Term | Plain meaning |
|---|---|
| **VLM** | Vision Language Model — show it a picture, ask it questions |
| **LLM** | Large Language Model — the text-only "brain" inside a VLM |
| **ViT** | Vision Transformer — the "eye"; turns image patches into numbers |
| **Encoder / vision tower** | The eye. Converts pixels into numbers |
| **Connector / projector** | The small translator between eye and brain |
| **Parameter / weight** | One learned number inside the model. "7B" = 7 billion of them |
| **Bit / byte** | A bit is one binary digit; a byte is 8 bits |
| **Precision / bit-width** | How many bits each number is stored in |
| **BF16 / FP16** | 16-bit number formats — the standard for an uncompressed model |
| **INT8 / INT4** | 8-bit and 4-bit whole-number formats used after quantization |
| **Quantization** | Storing each weight in fewer bits |
| **Mixed precision** | Different parts of the model stored at different bit-widths |
| **Scale / zero-point** | Extra numbers stored alongside quantized weights so they can be converted back |
| **Group size** | How many weights share one scale. Smaller = better quality, bigger file |
| **Calibration** | A small data sample used to choose the rounding rules |
| **Post-training (PTQ)** | Compression applied after training finishes; no retraining |
| **Lossless** | The output is bit-for-bit identical to the original. Zero quality loss, guaranteed |
| **Entropy** | How predictable a set of numbers is. Low entropy = compresses well |
| **Huffman coding** | Giving frequent patterns short codes and rare ones long codes |
| **Pruning** | Deleting parts of the model outright |
| **Structured pruning** | Deleting whole units (a layer, a head), not scattered individual numbers |
| **Width vs depth pruning** | Making layers narrower vs deleting whole layers |
| **Low-rank / SVD** | Replacing one big table of numbers with two small ones that approximate it |
| **Rank** | How small those two replacement tables are. Lower rank = smaller, less accurate |
| **Distillation** | Training a small "student" to copy a big "teacher" |
| **Recovery training** | Brief retraining after compression to win back lost quality |
| **Embedding table** | The lookup table mapping vocabulary words to numbers. Often surprisingly large |
| **Integer programming** | A maths optimiser that finds the best answer subject to hard limits |
| **Sensitivity** | How much quality you lose when you compress a specific part |
| **Budget** | A limit you must not exceed — here, gigabytes on disk |
| **Baseline** | The existing result you must beat |
| **Benchmark** | A standard test set everyone reports scores on |
| **OCR** | Optical Character Recognition — reading text inside an image |
| **Hallucination** | The model confidently describing something not in the image |
| **Open / closed** | Nobody has done it yet / thoroughly done already |
| **Scoop risk** | The chance someone else publishes your idea first |
| **GGUF / safetensors** | The two common file formats for saved models |
| **FLOPs / latency / KV cache** | Query-speed concepts — **all irrelevant to file size** |

---

## 11. One-paragraph summary

Only five techniques actually shrink a VLM's file: quantization, lossless weight coding, low-rank
factorization, structured pruning, and distillation — and the enormous visual-token-compression literature
changes the file size by zero bytes, so ignore it. Of the five, quantization dominates: 16-bit → 4-bit is
4× for almost free, which means the honest framing of your contribution is **precision of targeting**, not
magnitude of compression. The genuinely open problem is that a VLM's three weight groups — eye, translator,
brain — have importance-to-size ratios differing by roughly 25×, yet every existing tool quantizes them
uniformly, and every budget-targeted bit allocator (GAMMA, LLM-MQ) is text-only. Add to that three
unmeasured facts: quantization scale metadata adds 6–25% to the real file, the embedding table can be a
quarter of a 4-bit model, and standard tooling often leaves the vision encoder entirely uncompressed.
**Audit the real bytes first, measure per-component sensitivity second, then build the allocator that hits
an exact gigabyte target** — and test whether lossless coding, which guarantees identical output, behaves
differently on a vision encoder than on a language model, because nobody has checked.
