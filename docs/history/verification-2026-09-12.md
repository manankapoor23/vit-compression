# Verification Audit — 2026-09-12

> **CORRECTION, 2026-09-25 — read before using §2.3.** The "correctable error in Kuzmin et al."
> below does not hold. In Kuzmin's own frame (overhead as a share of an FP16 weight) their
> comparison survives every configuration we built — route A's group-36 scales are 2.78% of FP16,
> under pruning's 6.25%. The "~1.0 bits/param at group 32 = 25% of a 4-bit model" row silently
> assumed **asymmetric** scales we never implemented; symmetric is **0.500 bits = 12.5%**. What
> survives is a denominator argument (at a byte budget the scales are 12.6% of route A's file) —
> see `results.md` §3.1. The "35–42% unshared failures" in §2.2 also lacked a seed floor; see
> `results.md` §1.3.

Run while the 12 seeded training jobs were executing. Everything below is either checked or
explicitly marked unchecked.

---

## 1. Arithmetic — 35/35 PASS ✅

Every derived number in `results.md` was recomputed from raw measurements: parameter shares,
budget in bits/param, bits/param from measured sizes, overhead percentages, leftover shares,
scale cost, all four ternary entropy figures, the Mix-QViT reconstruction, the FTerViT packing
win, both size spreads, all four corruption drops, and the ECE and gap ratios.

**No discrepancies.** Largest deviation was 0.08% (attention share, 32.182 vs 32.2 as printed).

---

## 2. Two claims must be weakened ⚠️

### 2.1 "Nobody counts quantization metadata" — TOO STRONG

**False for the LLM quantization literature:**

| Work | What it does |
|---|---|
| **SpQR** | Explicitly accounts for metadata: bilevel quantization, 3-bit scales and zeros at group 16, **0.625 bits/weight** of grouping overhead |
| **BayesQ** | Tracks theoretical vs measured bits; reported average **includes scales, codebooks and headers**, to a ±0.02 bit tolerance |
| Cluster-based quantization studies | Report **0.64–10.33%** metadata overhead for OPT at 4-bit |

**Still true for the ViT quantization papers we checked:** LampQ, MixFrag and Mix-QViT report
average bit-width or `Σ|w|·b`, with no metadata term.

**Corrected wording:** *"Metadata accounting is established in the LLM quantization literature
(SpQR, BayesQ) but is absent from the ViT mixed-precision papers we surveyed, which specify
budgets as average bit-width."*

### 2.2 "Nobody compares compression methods at an equal budget" — PARTLY FALSE ✅ RESOLVED

**[Kuzmin, Nagel, van Baalen, Behboodi, Blankevoort — *Pruning vs Quantization: Which is Better?*
NeurIPS 2023, Qualcomm AI Research](https://arxiv.org/abs/2307.02973).** Full text read from
`kuzmin.txt`. All four unknowns resolved:

| Question | Answer | Effect on us |
|---|---|---|
| How is it equalised? | **Theoretical compression ratio.** *"we used a theoretical compression ratio for quantization, which depends on the bitwidth"* (§6). Table 1 says "equal model size" but it is nominal. | **Our measured-bytes claim survives** |
| Is metadata counted? | **No — explicitly excluded.** *"assume no overhead on storing the sparsity mask"* (×2). Quantization overhead dismissed as *"a single scaling factor per tensor (or channel)"* | **Survives, and see §2.3 below** |
| Beyond accuracy? | **No.** Table 1 reports accuracy / mIoU / mAP / perplexity only. No robustness, corruption, calibration or per-example analysis | **All of Phase 3 is untouched** |
| Vision transformers? | **Yes — ViT**, 81.3% baseline; 3-bit quant = 78.4% | Sanity check: their −2.9 pts at 3-bit vs our −2.7 pts. Consistent. |

**WE MUST CONCEDE: the equal-fine-tuning-budget control is theirs.** *"For a fair comparison, we
used the same amount of epochs of fine-tuning for each method."* **Cite Kuzmin et al. for that
design choice. Do not present it as ours.**

**They hand us two openings, in their own words:**

1. *"we do not study combinations of pruning and quantization apart from analyzing the inherent
   sparsity. **We leave this for future work.**"* — **Route C is exactly that combination.**
2. *"we solely focused in our comparison on **unstructured pruning**"*, and they explicitly decline
   to make claims for structured pruning where pruning wins. **Ours is structured.**

**And one unexpected ally (Appendix I):** *"fine-tuning after pruning tends to recover the original
representation, while quantization-aware training leads to learning **completely new
representations**."* This is the qualitative version of our per-example divergence result.
**Cite it as supporting evidence** — our 35–42% unshared failures is its quantitative measurement.

### 2.3 ~~NEW FINDING — a correctable error in Kuzmin et al.~~ — WITHDRAWN 2026-09-25, see banner

They dismiss quantization's storage overhead as *"a single scaling factor per tensor (or channel)"*.
**That is true for per-tensor scaling and false for group-wise scaling**, which is what AWQ, GPTQ
and every modern quantizer actually use.

| Scale granularity | Overhead (measured, DeiT-S) |
|---|---|
| Per-tensor (their assumption) | negligible |
| **Group 128** | **0.120 bits/param** |
| **Group 32** | **~1.0 bits/param = 25% of a 4-bit model** |

Their conclusion — *pruning's mask overhead is real, quantization's is not* — **does not hold under
group-wise quantization.** This is a concrete, measured correction to a NeurIPS paper, and it
strengthens the case for reporting measured bytes.

## 3. What survives re-verification ✅

| Claim | Status after reading Kuzmin |
|---|---|
| Equalised on **measured serialized bytes** | **Survives** — Kuzmin uses theoretical ratio, excludes metadata |
| **Four routes**, incl. distillation and prune+quant | **Survives** — Kuzmin does two; flags combinations as future work |
| **Structured** pruning | **Survives** — Kuzmin is unstructured only, declines claims for structured |
| Robustness + calibration + per-example agreement | **Survives** — Kuzmin reports accuracy only |
| **Rank inversion under corruption** | **Survives** — invisible to a single-metric study |
| **Group-wise metadata cost is not negligible** | **NEW** — corrects an explicit claim in Kuzmin §6 |
| Overhead scales with bit-width AND model size | **Survives** — no prior work found |
| ~~Equal fine-tuning budget as a control~~ | **CONCEDED to Kuzmin et al. 2023. Cite them.** |

## 4. Downgraded to supporting detail, not a finding ⬇️

**"Ternary recovers 3.7× faster per step."** The qualitative direction is already known — ternary
accumulates more quantization error and *"requires a prolonged training budget to achieve
convergence"*. Our quantitative crossover estimate (~step 4400) was not found elsewhere, but it is
extrapolated, not measured.

**Report as an observation with the extrapolation flagged. Do not headline it.**

---

## 5. Net effect on the paper

**Unchanged:** the core result. Four routes, equal measured bytes, equal training budget, and a
ranking that inverts under corruption.

**Narrowed:** the framing. Not *"the field does not measure properly"* — that is false for LLM
quantization work and unfair to papers doing their own job correctly. Instead:

> Compression comparisons are typically equalised by compression rate or bit-width. We equalise on
> **measured serialized size**, compare **four routes rather than two**, hold the **recovery budget
> constant**, and evaluate **robustness, calibration and example-level behaviour** alongside
> accuracy. Under that design the ranking by clean accuracy inverts.

**Every qualifier is load-bearing.** Drop any one and prior work covers it.

---

## 6. Still unverified — the honest list

- [ ] Kuzmin et al. 2307.02973 full text — **blocking**
- [ ] Whether the ViT quantization papers report metadata anywhere outside their main tables
- [ ] Whether any paper reports a corruption-induced rank inversion (searched, none found — absence
      of evidence)
- [ ] Hendrycks & Dietterich and Guo et al. were verified this session; their *application to
      compressed models* was confirmed to exist (Hooker 2019, Hidden Cost CVPRW 2024)
