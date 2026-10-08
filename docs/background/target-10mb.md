# Target: A Vision Language Model Under 10 MB

## Where the target sits

| Model | Params | Disk size |
|---|---|---|
| DeiT-Small (vision only, no language) | 22M | 88 MB (FP32) |
| SmolVLM-256M (smallest published VLM) | 256M | 512 MB (BF16), 128 MB (INT4) |
| **This project** | **~20M** | **<10 MB** |

## What 10 MB buys

10 MB = 83.9 million bits.

| Precision | Params affordable |
|---|---|
| INT8 | 10M |
| INT4 | 21M |
| INT2 | 42M |

## The key consequence

**Compressing an existing VLM cannot reach 10 MB.** SmolVLM-256M at 4-bit is 128 MB — 13× over budget.
At 2-bit, where quality collapses, it is 64 MB. No compression path from any published VLM reaches your
target.

This is therefore not a compression project but **co-design**: design a tiny architecture, distill (train
it to imitate a big model) into it, then quantize.

## A budget that fits

| Component | Params | Bits | Size |
|---|---|---|---|
| Vision encoder — the "eye" (DeiT-Tiny scale) | 5M | 4 | 2.50 MB |
| Connector — the translator | 0.3M | 4 | 0.15 MB |
| Language model body — the "brain" | 10M | 4 | 5.00 MB |
| Embedding table (8k vocab × 256 dims) | 2.1M | 8 | 2.10 MB |
| **Total** | **17.4M** | | **9.75 MB** |

## The non-obvious constraint

**Vocabulary size decides whether this is possible.** A standard 32k-word vocabulary at 256 dimensions is
8.4M parameters — **4.2 MB at 4-bit, nearly half the budget**, spent before any reasoning capacity exists.
Shrinking the vocabulary to ~8k, tying input and output embeddings (one shared table), or using byte-level
tokens is mandatory, not optional. Nobody discusses this because at 3B parameters it is a rounding error.

## Be honest about capability

A 10 MB VLM will not do general visual question answering. It can plausibly do **one narrow task well**:
one fixed domain, one document type, yes/no questions about a scene. Frame it accordingly — *"the smallest
VLM that still does X"* is publishable; *"a general VLM in 10 MB"* is not.

## Where the novelty is

No published VLM exists below ~230M parameters. **The 10–50M range is empty.** Establishing what
capability survives there, and which vocabulary / encoder / brain split maximises it, is unclaimed
territory. Your contribution is the frontier itself, not a compression ratio.
