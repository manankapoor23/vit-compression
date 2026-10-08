# Progress

**Updated:** 2026-09-25

```
  PHASE 1  AUDIT          ████████████████████  DONE  (now 5 architectures)
  PHASE 2  BUILD ROUTES   ████████████████████  DONE  (12 models: 4 routes x 3 seeds)
  PHASE 3  EVALUATE       ████████████████████  DONE  (+ offline re-analysis, 2026-09-25)
  PHASE 4  WRITE UP       ░░░░░░░░░░░░░░░░░░░░  not started

  Experiments needed for the claim:  COMPLETE
  Evidence off the GPU box:          COMPLETE  (evidence/, 259 MB, verified to the byte)
  Coursework deliverable:            ~80% done  [estimate: only writing remains]
  Publishable paper:                 ~55% done  [estimate: writing + one optional GPU run]
```

**Right now:** every experiment is finished and every headline number has been recomputed on the
Mac from the raw per-image arrays. The claims were re-sized after a literature check — see the top of
`results.md` (repo root), which is the authority for every number. This file tracks state only.

**GPU access:** gone as of 2026-09-25. Nothing on the critical path needs it.

---

## Phase 1 — Audit ✅ DONE

- `phase1_audit.py` — writes real files, measures with `os.path.getsize()`. Raw binary container.
  New `--random-init` flag: sizes depend only on shapes (verified byte-identical to pretrained).
- **Five architectures** (`evidence/audit_multiarch/`): DeiT-Tiny, DeiT-S, Swin-T, DeiT-B, ViT-L.
  INT2 overhead +53.5% → +8.5% as width grows; overhead rises as bits fall in 5/5 models; all 80
  quantized files rebuilt from shapes by `src/arch_audit_summary.py`.
- Literature: LampQ, MixFrag, Mix-QViT, and a 2026 *Scientific Reports* ViT paper all use the
  nominal formula (the last two reconstruct to 0.03 and 0.003 MB). LLM quantization already counts
  metadata (M²XFP, SpQR) — so this is a cross-subfield gap, not a discovery.

## Phase 2 — Build the four routes ✅ DONE

| # | Route | Knob (from the logs) | Exported | Hit target |
|---|---|---|---|---|
| A | Uniform INT3 | group 36 | **9.718 MB** | 0.000 MB |
| B | Ternary + base-3 packing | 15 layers at INT8 | **9.785 MB** | 0.000 MB |
| C | INT4 all + structured prune | 24.375% of MLP channels | **9.796 MB** | 0.000 MB |
| D | Distill into DeiT-Tiny, then INT8 | 38 layers at FP16 | **9.765 MB** | 0.000 MB |

Spread 0.80%. Identical 3000-step budget (Kuzmin et al.'s control — cited as theirs). All 17
seeded and sweep model files verified against their logs to the byte. The knobs previously written
here (35 / 14 / 24.37% / 37) were pre-fix values — corrected 2026-09-25.

## Phase 3 — Evaluate ✅ DONE

| Result | Status after the 2026-09-25 checks |
|---|---|
| Clean A 76.49 > C 75.68 > B 72.40 > D 71.41 (n=50,000) | Recomputed from raw arrays, exact |
| Corrupted A 54.99 > C 53.49 > D 48.01 > B 44.24 (n=5,000 × 14) | Recomputed from raw arrays, exact |
| **Corruption fingerprints** — ternary loses 17.7–20.2 pts on noise vs 5.9–6.4 on digital, within DeiT-S | **NEW, lead claim** |
| **Routes vs seeds** — route disagreement is 1.2–2.0× a measured seed floor, all CIs exclude 0 | **NEW**, replaces "33–43% unshared" |
| B↔D flip — clean favours B, corruption favours D on 7/14 and B on 2/14 | Survives, **at the decision level only**: it needs D's architecture change |
| Within DeiT-S, does any corruption reverse the clean order? | **No** (42 tests). Consistent with *Accuracy on the Line* |
| Calibration: D's ECE 2.8× higher | **Demoted** — a uniform underconfidence offset, removable by one temperature |
| Group sweep: returns collapse 9.6× | Supporting; the size side is a serializer check |

Scripts: `src/stats.py`, `src/aggregate_seeds.py`, `src/offline_analysis.py` (new).

## Phase 4 — Write up ⬜ NOT STARTED

The claim paragraph is at the bottom of `results.md`. **Estimate: 1–2 weeks.**

---

## What's left

| # | Item | Time | Verdict |
|---|---|---|---|
| 1 | Write the paper / report | 1–2 weeks | Do it |
| 2 | Score the 5 archived group-sweep models under corruption → a same-architecture accuracy line | ~1 hr GPU [estimate] | **Only if GPU access returns.** Settles the one open question (is ternary *off* the line?) |
| 3 | Read Lost in Pruning and 2405.20876 in full (currently cited from summaries) | ~40 min | Do before citing |
| 4 | W2 — Fisher sensitivity instead of weight variance | ~2 hr GPU | Optional. State as a limitation |
| 5 | W3 (RTN vs GPTQ), W4 (weight-only, 20% data) | — | **State as limitations, don't fix** |

---

## How close is this to publication?

| Venue | Verdict |
|---|---|
| Coursework (R4 rubric) | **Solid.** Controlled comparison, honest negative results, causal checks |
| SCIE journal (*Scientific Reports*, *IEEE Access* tier) | **Plausible**, framed as a controlled empirical benchmark with the protocol |
| CVPR / NeurIPS main track | **No.** A new configuration, not a new phenomenon or method |

**What cannot be overturned by training randomness:** the five-architecture size audit, the
literature reconstructions, the reporting protocol.

**What a reviewer will push on, and the answer:** *"rank flips on synthetic corruptions are expected"*
→ cite Miller; the flip is stated at the decision level, and the within-architecture fingerprint is
separate. *"Seeds alone cause disagreement"* → measured; routes exceed it 1.2–2.0×. *"ECE is fixable
by temperature"* → agreed, reported as a deployment note.

---

## Next action

Open `results.md` and read the first section, *The research problem, resolved* (about 5 minutes).
It is the whole project in one page, and it is what the write-up follows.
