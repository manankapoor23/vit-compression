# Vision Language Models (VLMs), Explained Plainly

A VLM is a model you can show a picture to and talk to about it in the same breath.
Ask "what's the total on this receipt?", attach a photo, and get "₹1,240" back.

This doc covers: what's inside one, how it actually processes an image, real examples
at both ends of the size range, and the several different things people mean when they
say a model is "big" or "small."

---

## 1. The one-sentence version

**A VLM is a normal text model with a converter bolted onto the front that turns pixels
into things the text model can read.**

That's the whole trick. The language model was never retrofitted to "see." Instead, the
image is chopped up, each piece is turned into the same kind of number-list that words
get turned into, and those pieces are handed to the language model as if someone had
typed them.

---

## 2. The three parts

Every mainstream VLM has the same three pieces:

```
   IMAGE                                              TEXT PROMPT
     |                                                     |
     v                                                     |
+-------------+     +-------------+                        |
|   PART 1    |     |   PART 2    |                        |
|   The Eye   | --> | The Adapter | -------+               |
| (vision     |     | (projector) |        |               |
|  encoder)   |     |             |        v               v
+-------------+     +-------------+     +---------------------+
                                        |       PART 3        |
                                        |     The Reader      |
                                        | (the language model)|
                                        +---------------------+
                                                   |
                                                   v
                                              ANSWER (text)
```

### Part 1 — The eye (the *vision encoder*)

Takes the raw image and describes it in numbers.

It does this by cutting the picture into a grid of small squares, typically **14×14
pixels** each, called **patches**. A 336×336 photo becomes a 24×24 grid = **576
patches**. Each patch is turned into a list of numbers (say 1,024 of them) that captures
"what's here and how does it relate to everything else in the picture."

That list of numbers is a **vector**. Think of it as a coordinate: similar-looking things
land near each other. A patch of grass and another patch of grass get near-identical
coordinates.

The eye is almost never trained from scratch. Nearly everyone reuses an off-the-shelf one:

| Vision encoder | Size | Used by |
|---|---|---|
| CLIP ViT-L/14-336 | ~300M params | LLaVA-1.5, Phi-3.5-vision |
| SigLIP-SO400M | ~400M params | PaliGemma, SmolVLM-2.2B, Gemma 3 |
| SigLIP-base | ~93M params | SmolVLM-256M / 500M |
| InternViT-6B | ~6B params | InternVL series |

Note the range: **93 million to 6 billion parameters just for the eye.** This matters
later.

### Part 2 — The adapter (the *projector* or *connector*)

The eye speaks in vectors of one length (1,024 numbers). The language model expects
vectors of a different length (4,096 numbers). The adapter is a translator between them.

In the simplest and most common design it is literally **two small matrix multiplications**
— a couple of million parameters, the smallest part of the whole system. LLaVA proved in
2023 that this tiny piece is enough.

There are three adapter styles in use:

| Style | What it does | Cost | Examples |
|---|---|---|---|
| **MLP projector** | Passes every patch through as a token, in-line with the text | Simple, but many tokens | LLaVA, Qwen-VL, PaliGemma, most open models |
| **Resampler / Q-Former** | Squeezes all patches down to a fixed small number (e.g. 32 or 64) | Cheap, loses fine detail | BLIP-2, MiniCPM-V |
| **Cross-attention** | Image stays outside the text stream; text layers "look over" at it | Text speed unaffected, but adds big layers | Flamingo, Llama 3.2 Vision |

### Part 3 — The reader (the *language model*)

An ordinary LLM — Llama, Qwen, Gemma, Mistral, Phi. It does all the reasoning, all the
answering, all the writing. Once the image has been converted, it genuinely cannot tell
which of its inputs came from a keyboard and which came from a camera. They're all just
vectors in a row.

---

## 3. Walk through one actual question

**Setup:** LLaVA-1.5-7B. You upload a 336×336 photo of a dog and ask *"What breed is this?"*

1. **Resize** — image is squashed/padded to exactly 336×336.
2. **Cut** — into 14×14 pixel patches → a 24×24 grid → **576 patches**.
3. **Encode** — CLIP ViT-L/14 turns each patch into a vector of **1,024 numbers**. You now
   have a 576 × 1,024 block of numbers.
4. **Adapt** — the projector converts each one to **4,096 numbers**, the width the 7B
   language model uses. Still 576 items.
5. **Assemble** — the input to the language model is now:

   ```
   [576 image vectors] + ["What", " breed", " is", " this", "?"]
   ```

   576 image tokens + 5 text tokens = **581 tokens**.
6. **Generate** — the language model reads all 581 and writes out one word at a time:
   "This appears to be a Golden Retriever..."

**The thing to notice:** that one small photo cost **576 tokens** — roughly 430 English
words, about a page and a half of text. The image dwarfs the question. This is the single
most important fact about VLM efficiency, and section 7 comes back to it.

---

## 4. How they get trained

Three stages, in order. Nobody trains all three parts at once from zero.

**Stage 1 — Train the eye (usually already done for you).**
CLIP/SigLIP are trained on hundreds of millions of image–caption pairs scraped from the
web, learning only one job: put an image and its correct caption at the same coordinates.
Most VLM teams download this and never touch it.

**Stage 2 — Alignment.** Freeze the eye. Freeze the language model. Train **only the
adapter**, on simple image–caption pairs. This teaches the translator its vocabulary.
LLaVA used 558,000 pairs for this — cheap, a few GPU-hours.

**Stage 3 — Instruction tuning.** Unfreeze the language model (and often the eye too).
Train on real tasks: questions about images, reading text in photos, describing charts,
locating objects. LLaVA-1.5 used 665,000 examples. Big commercial models use tens of
millions, plus human feedback on top.

Stage 3 is where a model learns to *answer*, not just *describe*.

---

## 5. Big VLMs — real examples

### Closed models (weights not released, sizes not published)

| Model | Maker | Published parameter count |
|---|---|---|
| GPT-4o / GPT-5 | OpenAI | Never disclosed |
| Gemini 2.5 Pro | Google | Never disclosed |
| Claude (Opus / Sonnet) | Anthropic | Never disclosed |

For these, size is inferred indirectly from price and speed. Treat any specific number
you see quoted for them as a rumour. What *is* public for GPT-4o is how it bills images,
which tells you the token cost: **85 tokens base + 170 tokens per 512×512 tile**. A
1024×1024 image = 4 tiles = 85 + 680 = **765 tokens**.

### Open models (you can download the weights and count them yourself)

| Model | Total params | The eye | The reader | Notes |
|---|---|---|---|---|
| **Llama 3.2 90B Vision** | 90B | ViT-H (~630M) | Llama 3.1 70B | ~20B of that is cross-attention layers added *into* the text model |
| **Qwen2.5-VL 72B** | 72B | ~670M ViT | Qwen2.5 72B | Handles native resolution — token count scales with image size |
| **InternVL 2.5 78B** | 78B | InternViT-6B | Qwen2.5 72B | Unusually huge eye |
| **Pixtral Large** | 124B | 1B encoder | Mistral Large 2 (123B) | |
| **Molmo 72B** | 72B | CLIP ViT-L | Qwen2 72B | Trained on human-spoken image descriptions |
| **LLaVA-1.5 13B** | 13B | CLIP ViT-L (300M) | Vicuna 13B | The classic reference design |

**What "90B" actually contains.** Llama 3.2 90B Vision = Llama 3.1 **70B** text model +
**~20B** of new vision machinery. The vision *encoder* itself is only ~630M of that; the
other ~19B is cross-attention layers threaded through the language model. So the label on
the box includes the eye and the plumbing, not just the reader.

---

## 6. Small VLMs — real examples

| Model | Total params | The eye | The reader | Runs on |
|---|---|---|---|---|
| **Florence-2-base** | 0.23B | DaViT | small encoder-decoder | Phone / CPU |
| **SmolVLM-256M** | 0.26B | SigLIP-base (93M) | SmolLM2-135M | Phone / browser |
| **SmolVLM-500M** | 0.5B | SigLIP-base (93M) | SmolLM2-360M | Phone |
| **Moondream 2** | 1.9B | SigLIP | Phi-1.5 (1.3B) | Laptop, Raspberry Pi |
| **SmolVLM-2.2B** | 2.2B | SigLIP-SO400M (400M) | SmolLM2-1.7B | Laptop |
| **Qwen2.5-VL 3B** | 3.8B | ~670M ViT | Qwen2.5 3B | Laptop / 8GB GPU |
| **PaliGemma 3B** | 2.9B | SigLIP-SO400M (400M) | Gemma 2B | Laptop |
| **Phi-3.5-vision** | 4.2B | CLIP ViT-L (300M) | Phi-3.5-mini (3.8B) | Laptop / 8GB GPU |

**Notice the proportions flip.** In PaliGemma 3B, the eye (400M) is **14%** of the model.
In SmolVLM-256M, the eye (93M) is **36%** of the model. In Llama 3.2 90B it's **0.7%**.
The smaller the VLM, the more of it is eye. That has a direct consequence for compression:
shrinking the language model gets you less and less as you go down.

---

## 7. How size is measured — six different numbers

This is where "how big is it?" stops having one answer. Six different measurements are all
called "size," and they don't track each other.

### 7.1 Parameter count

The number of learned numbers in the model. Reported in **M** (millions) or **B**
(billions). This is what the name usually refers to: "7B" = 7 billion parameters.

**Total vs active.** Mixture-of-Experts models hold many parameters but only use a fraction
per token. Such a model may be "8×7B total, 13B active" — it takes 8×7B of memory but costs
roughly 13B of compute. Always ask which number you're being told.

### 7.2 Memory footprint = parameters × bytes per parameter

Parameter count alone doesn't tell you if it fits on your machine. You need the
**precision** — how many bytes each number is stored in.

| Precision | Bytes per parameter | Typical use |
|---|---|---|
| FP32 | 4 | Training, rarely for running |
| FP16 / BF16 | 2 | The standard for running models |
| INT8 | 1 | Compressed, near-lossless |
| INT4 | 0.5 | Compressed, small quality drop |

Multiply them:

| Model size | FP32 | FP16 | INT8 | INT4 |
|---|---|---|---|---|
| 256M | 1.0 GB | 0.5 GB | 0.26 GB | 0.13 GB |
| 2B | 8 GB | 4 GB | 2 GB | 1 GB |
| 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 72B | 288 GB | 144 GB | 72 GB | 36 GB |
| 90B | 360 GB | 180 GB | 90 GB | 45 GB |

Then add **20–30% on top** for the working scratch space the model needs while running
(activations and the KV cache). A 7B model at FP16 needs ~14 GB of weights and realistically
wants a **16–24 GB** GPU.

**Practical read:** a 7B VLM at INT4 (3.5 GB) fits comfortably on a consumer laptop GPU. A
72B VLM at INT4 (36 GB) needs a data-center card or two. A 90B VLM at FP16 (180 GB) needs
a multi-GPU server.

### 7.3 Visual tokens per image — the hidden size

**The measurement people forget, and the one that dominates VLM cost.**

Two models with identical parameter counts can differ 10× in cost per image, because they
disagree on how many tokens a picture is worth. And attention cost grows with the *square*
of token count — doubling tokens roughly quadruples that part of the work.

| Model / setting | Tokens for one image |
|---|---|
| PaliGemma @ 224px | **256** |
| LLaVA-1.5 @ 336px | **576** |
| Qwen2.5-VL @ 1024×1024 | **~1,330** (scales with resolution: `(H/28) × (W/28)`) |
| GPT-4o @ 1024×1024, high detail | **765** (85 + 4 tiles × 170) |
| LLaVA-NeXT, high-res multi-crop | **up to ~2,880** |
| SmolVLM (aggressive compression) | **~64–81 per tile** |

Read that last row against the second: **SmolVLM spends roughly one-eighth the tokens
LLaVA-1.5 does.** That compression is a bigger part of why SmolVLM is fast than its
parameter count is.

### 7.4 Context window

How many tokens the model can hold at once — text and image tokens together. A 128,000-token
window sounds enormous until you price it in images: at 576 tokens each, that's ~220 images
and no room to think. At 2,880 tokens each (high-res mode), it's **44 images**.

Context window is what determines "can it read a 60-page PDF?" — and image tokens are what
eat it.

### 7.5 Compute per answer (FLOPs) and latency

The actual work done. Driven by parameters × tokens, so it compounds: a big model reading
many image tokens is expensive twice over. What you feel in practice:

- **Time to first token** — dominated by the eye plus reading all those image tokens.
- **Tokens per second** — dominated by language model size and memory bandwidth.

A 2B VLM on a laptop typically answers in **under a second**. A 72B VLM on a single server
GPU typically takes **several seconds**.

### 7.6 Disk size of the download

What you actually pull down. Roughly equal to 7.2's precision math, because checkpoints are
usually shipped in BF16. A "7B" model is a **~14 GB** download. Quantized community
re-uploads of the same model are 4–8 GB.

### Summary: the six numbers side by side

| Measure | Unit | Governs |
|---|---|---|
| Parameters | B / M | Reasoning ability, and the headline number |
| Precision | bytes/param | Memory, and small quality trade |
| Memory footprint | GB | **Whether it runs on your hardware at all** |
| Visual tokens/image | tokens | **Cost and speed per image** |
| Context window | tokens | How many images/pages at once |
| Latency | ms, tok/s | Whether it feels usable |

If you take one thing from this section: **parameters tell you if it's smart, tokens-per-image
tell you what it costs.** They are independent knobs, and most people only ever look at the
first.

---

## 8. What you actually lose by going small

Concretely, in order of how quickly it degrades:

1. **Reading small text (OCR).** The first casualty. A 2B model reads a road sign fine and
   mangles a paragraph of 8pt terms-and-conditions. This is usually a *resolution and
   token-budget* limit rather than a brain limit — the patches are too coarse to hold the
   letters.
2. **Dense charts and tables.** Small models read the title and the axis labels, then invent
   the numbers.
3. **Counting.** "How many people are in this photo?" is unreliable below ~7B, and shaky
   above it.
4. **Multi-step reasoning about what's seen.** "Is the person on the left taller than the
   sign behind them?" needs the reader, and the reader is what got cut.
5. **Multiple images at once.** Comparing two images, or following a sequence of frames, is
   largely a big-model capability.
6. **Confident wrong answers.** Small VLMs hallucinate objects that aren't present at a
   noticeably higher rate. They rarely say "I can't tell."

What small models are genuinely good at: describing a scene, classifying, spotting an
obvious object, reading large clear text, tagging photos, simple yes/no questions about an
image, and doing it on a phone, offline, for free, in under a second.

---

## 9. Making them smaller (compression)

Five levers, roughly in order of value-for-effort:

**1. Quantization — fewer bits per number.** Store weights as INT8 or INT4 instead of 16-bit.
INT8 is close to free in quality. INT4 costs a little accuracy for **4× less memory**. This
is the standard first move and usually the only one needed.

*VLM-specific caveat:* quantize the language model hard, but treat the vision encoder
gently. The encoder is a small share of the bytes (0.7% of Llama 3.2 90B, 14% of PaliGemma
3B) so crushing it saves almost nothing — while OCR and fine-detail accuracy are highly
sensitive to encoder precision. Bad trade.

**2. Cutting visual tokens.** The lever unique to VLMs, and often the biggest win, because
it attacks the quadratic term. Techniques: pixel-shuffle (fold a 2×2 group of patches into
one token), pooling, a resampler with a fixed small output, or dropping patches that carry
little information. Going 576 → 144 tokens cuts image cost ~4× with modest quality loss.
Going 576 → 64 is aggressive, and is roughly what SmolVLM does.

**3. Lower input resolution.** Fewer pixels → fewer patches → fewer tokens, quadratically.
Cheap and effective, and it destroys OCR. Fine for scene description, wrong for documents.

**4. Distillation — train a small model to imitate a big one.** The small model learns from
the big model's outputs rather than from raw labels, and lands better than training it
alone. Expensive up front, permanent payoff. This is how most good small VLMs were made.

**5. Pruning — delete weights that contribute little.** Structured pruning (removing whole
attention heads or layers) gives real speedups; unstructured pruning gives sparsity that
most hardware can't exploit. Generally the least reliable lever of the five.

These stack. INT4 (4× smaller) plus 4× token reduction is a **16× cut in the practical cost
of answering a question about an image**, with the quality hit concentrated in fine detail.

---

## 10. Which one should you use

| Situation | Pick | Why |
|---|---|---|
| On a phone, offline | SmolVLM-256M/500M, Florence-2 | Under 1 GB at INT8 |
| On a laptop, no GPU | Moondream 2, SmolVLM-2.2B | ~1–2 GB at INT4, ~1s answers |
| One consumer GPU (8–24 GB) | Qwen2.5-VL 7B, Phi-3.5-vision | Best quality that fits |
| Reading documents, invoices, forms | Qwen2.5-VL (any size) | Native high resolution — resolution beats parameters for OCR |
| Hardest visual reasoning | GPT-5, Gemini 2.5 Pro, Claude Opus, or Qwen2.5-VL 72B if it must be local | |
| Millions of images, cost-sensitive | Small model + heavy token reduction | Token count, not parameters, is your bill |

---

## 11. Glossary

| Term | Plain meaning |
|---|---|
| **Token** | One chunk of input. For text, roughly a word-piece. For images, one patch (or a merged group of patches). |
| **Patch** | A small square of the image, typically 14×14 pixels. |
| **Vector / embedding** | A list of numbers standing for one token. Similar things get similar lists. |
| **Vision encoder (ViT)** | The eye. Turns patches into vectors. |
| **Projector / connector** | The adapter that reshapes the eye's vectors to fit the language model. |
| **Parameter** | One learned number inside the model. Models have billions. |
| **Precision** | How many bytes each parameter is stored in. Fewer bytes = smaller and faster, slightly less accurate. |
| **Quantization** | Converting parameters to fewer bytes each. |
| **Context window** | The maximum number of tokens the model can hold at one time. |
| **KV cache** | Scratch memory the model keeps while writing an answer. Grows with token count. |
| **Distillation** | Training a small model to copy a big model's behaviour. |
| **MoE (Mixture of Experts)** | A design where only part of the model runs per token — big in memory, cheaper in compute. |

---

## The four sentences worth remembering

1. A VLM is a text model with a converter in front that turns image patches into tokens.
2. "Size" means at least six different things; **memory footprint** decides if it runs, and
   **tokens per image** decides what it costs.
3. The name on the box (7B, 90B) counts the eye and the plumbing too, not just the reader —
   and in small models the eye can be a third of the total.
4. Going small costs you fine text, chart numbers, counting, and honesty about uncertainty;
   it costs you very little on "describe this scene."
