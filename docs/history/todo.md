# TODO

**Rule:** one job at a time, foreground. Kill and verify with `pgrep` before starting the next.
Three zombie nohup loops competing for one MIG slice cost ~2 hours on 2026-09-11.

---

## Now — finish Phase 2

- [ ] Run route D: `python -u src/train_routes.py --route D --steps 3000 --distill 2>&1 | tee logs/train_D.log`
- [ ] Finalize B: `python -u src/finalize.py --route B 2>&1 | tee logs/final_B.log`
- [ ] Check all four `.bin` files are within ±1% of 9.77 MB. A drifted to 9.753 from a build of 9.718 — if routes drift differently, the equal-bytes claim weakens.

---

## Fix the four weaknesses

These come from `4-model-routes.md` §"Where we DO NOT have an edge". Two are cheap and worth
doing; two are limitations to state, not fix.

### W1 — Route A's group size was found by search, not designed ✅ DO IT

**Problem:** binary search landed on group 35. Nobody chose it. **1.22 MB of the 9.77 MB file
(12.6%) is scales.** We cannot claim anything about Route A's efficiency without knowing whether
that is a good operating point or an accident.

**Fix:** sweep Route A at group = 32 / 64 / 128 / 256 / 512, each re-tuned to 9.77 MB, each
trained for the same 3000 steps. Plot accuracy against group size.

**Cost:** ~1 hour of GPU (5 runs × ~12 min).
**Why it's worth it:** it is a *result*, not just a fix — "at a fixed byte budget, spending 12% of
the file on scales beats spending 3%" (or doesn't). Nobody has published that trade-off in
megabytes.

### W2 — Sensitivity ranking is weight variance ✅ DO IT

**Problem:** routes B and D decide which layers to protect using `weight.var()`. That is a crude
proxy. Proper methods use Hessian or Fisher information (how much the *loss* changes, not how much
the weights vary). **If any result turns on which layers were protected, this is the first thing a
reviewer attacks.**

**Fix:** replace `sensitivity_order()` in `src/build_routes.py` with Fisher information — run one
calibration batch, backprop, and rank layers by the mean squared gradient per parameter. Then
rebuild B and D and retrain.

**Cost:** ~2 hours (1 to implement, 1 to rebuild and retrain B and D).
**Why it's worth it:** cheap, and it removes the single most obvious attack on routes B and D.

### W3 — RTN instead of GPTQ/AWQ ❌ STATE, DON'T FIX

**Problem:** we round each weight to the nearest rung independently. GPTQ pushes each rounding
error into the weights not yet quantized so errors cancel; AWQ scales the ~1% of channels that
matter most before quantizing. **A specialist paper gets better accuracy at the same bit-width.**

**Why not fix it:** all four routes use the same quantizer, so the *comparison* stays fair — which
is the contribution. Implementing GPTQ is 1–2 days and improves every route roughly equally.

**What to write instead:** "All routes use round-to-nearest quantization. Calibration-based
methods (GPTQ, AWQ) would raise absolute accuracy across all routes; we use a single shared
quantizer so that differences between routes reflect the compression strategy rather than the
quantizer."

### W4 — Weight-only, 20% ImageNet, 3000 steps ❌ STATE, DON'T FIX

**Problem, three parts:**

1. We quantize **weights only** — activations stay bfloat16. The 3-bit ViT papers quantize both
   (W3A3, W3A8), which is substantially harder.
2. We train on 20% of ImageNet (200 of 1024 shards).
3. Nothing converged at 3000 steps. A was still climbing; B was climbing 3.7× faster.

**Why not fix it:** each fix is a different, larger project. Fixing (1) means solving activation
quantization. Fixing (2) needs 150 GB and days of training. Fixing (3) abandons the equal-budget
control, which *is* the experimental design.

**What to write instead, and never break this rule:**

> Our 77.14% must NEVER be compared against I&S-ViT's 55.78% (W3A3) or CPT-V's 61.16% (W3A8).
> Different problem. **Every comparison is route-vs-route, never route-vs-published.**

And state the budget finding as a result, not an apology: *"at a fixed 3000-step recovery budget,
ternary recovers 3.7× faster per step but starts 8 points lower; the ranking is budget-dependent
and we report the crossover."*

---

## Phase 3 — Evaluate (the actual paper)

- [ ] ImageNet-C corruptions, generated on the fly from val with `imagecorruptions`
- [ ] Calibration: Expected Calibration Error per route
- [ ] Per-example failure agreement: do the four routes fail on the *same* images
- [ ] Secondary: frozen-encoder + projector probe, with the Cambrian-1 dynamic-range caveat stated

---

## Phase 4 — Write up

- [ ] Breadth for SCIE: audit DeiT-T/S/B, Swin-T, ViT-B (the audit script already does this —
      one afternoon; journals reject single-model studies)
- [ ] Proposed reporting protocol: measured bytes, group size, scale policy, which layers stay
      un-quantized. Turns "the formula is wrong" into a methodological contribution.
- [ ] 40–60 references
- [ ] Code release

---

## Follow-up experiments worth running

- [ ] **Budget sweep for A and B at 6000 steps.** B was gaining +3.3 per 1000 steps against A's
      +0.9 and would cross around step 4400. Turns "B is worse" into "B is worse *at this budget*,
      and here is the crossover." Cost: ~40 min.
- [ ] Confirm the shm fix holds across a full unattended run before trusting `nohup` again.
