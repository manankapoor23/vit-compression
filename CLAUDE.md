# CLAUDE.md

## How to respond

- Lead with the action or the number. No preamble, no recap, no "let me know if".
- **Numbered steps** for anything multi-step. One bounded action per step.
- **Specific time estimates.** "20 min to write, 4 hours to run" — never "a bit of work".
- **Simple language.** Put the plain meaning in brackets next to any technical term: "Pareto curve (best accuracy at each cost level)". Assume no jargon is known.
- **Be deterministic.** Concrete numbers with the config that produced them. No vague hedging.
- **One concrete next action** at the end. Under two minutes to start.
- **Errors:** state cause and fix. No "uh oh".

# THE RULE: cross-verify everything

Every number, formula, statistic, and claim about model behaviour gets checked before it is
stated. Not estimated, not recalled, not eyeballed. **Computed, and checked a second way.**

This is not a preference. Four separate times on this project, an unverified number sent the work
in the wrong direction — an assumed sparsity figure made entropy coding the plan's headline, a
mischaracterised paper nearly lost a live research angle, a silent glob bug would have trained on
an empty dataset, and three wrong statistics were reported as findings.

**If it cannot be verified right now, say "not yet verified" and state what would settle it.**
Never fill the gap with a plausible number.

---

## 1. Arithmetic and units

- **Compute it, show it.** Run the calculation in a script rather than doing it in prose.
- **Check the second way.** Mix-QViT's reported 11.4 MB was confirmed by rebuilding it from
  `params × bits ÷ 8` plus 8-bit first/last layers — matching to 0.03 MB. One derivation is a
  guess; two that agree is a check.
- **State the expected magnitude first**, so a wrong answer is obvious. "INT8 should land near
  8.2 bits/param" is what caught a 45% serialization bug.
- **Units kill.** bits vs bytes; MB (10⁶) vs MiB (2²⁰); per-parameter vs total; code bits vs code
  bits + metadata. Quoting the packing win as 24% instead of 18.9% came from comparing a total
  against a code-bits-only figure.

## 2. Formulas

Check the definition against the source before implementing. Entropy, ECE, Jaccard, mCE,
compression ratio — all have conventions that differ between papers.

- log₂(3) = **1.585** is the entropy *floor* for a uniform ternary source. Base-3 packing achieves
  **1.600** (= 8/5). **They are different numbers.** Do not state a floor as an achieved value.
- Ternary is **3 levels at 1.585 bits**. INT2 is **4 levels at 2.000 bits**. Not the same thing.

## 3. Statistics

**A z-score or p-value is a claim. Compute it with the right test or do not state it.**

1. **Independent samples?** 14 corruptions of the same 5,000 images are NOT 70,000 samples.
   Effective n is the number of **images**. Getting this wrong inflated z by √14 ≈ 3.7×.
2. **Paired data?** If every model saw identical inputs, use **McNemar's exact test** for binary
   outcomes, or bootstrap over the shared units. An unpaired test is wrong in either direction.
3. **How many tests?** 6 pairs × 2 conditions = 12. Bonferroni: p < 0.05/12 = 0.0042.
4. **Report the interval.** "+3.05 pts, 95% CI [+2.4, +3.7]" is a result. "Significant" is an
   opinion.
5. **Say which way an error would cut, before running the test.** Pairing increases power, so
   "indistinguishable" claims are the fragile ones.

Use `src/stats.py`. Never estimate a p-value in prose.

## 4. Claims about the literature

Open the primary source and quote it. A search summary is not a source.

- "The literature reports `params × bits ÷ 8`" held for **LampQ, MixFrag, Mix-QViT** (quotes in
  the plan) but was **false for FTerViT**, which reports a real flash footprint. The unverified
  version would have been refuted in a viva.
- Distinguish **weight-only** from **weight+activation** quantization. Our 77.14% must never be
  compared with I&S-ViT's 55.78% (W3A3) — different problem.

## 5. Code

Review line by line before it runs. GPU and download time are expensive.

1. **Silent failures.** `webdataset` does brace expansion, not shell globs — `train-0[0-1]*.tar`
   matched **zero shards with no error**.
2. **Aliasing.** `.detach().cpu()` on a CPU tensor returns the **same object**; a later
   `load_state_dict` overwrites it in place. That silently cost route B 3.9 points.
3. **Library API guesses.** Wrap version-dependent calls in try/except and say which are guessed.
4. **A `--smoke` path** on every long-running script: ~20 steps, 60 seconds, run before every
   real run.
5. **Say which parts are verified and which are assumptions**, every time.

## 6. Before sending

State, explicitly: what was computed, what was checked a second way, and what remains an
assumption. If a number appears without one of those three labels, it should not be there.

## The project

Take DeiT-Small (88.20 MB, 22,050,664 params). Compress it under **10 MB on disk** by four
routes that all land at the **same measured byte size**. Document what each route costs.

**Budget: 3.63 bits per parameter.**

**In scope:** anything that changes bytes in the saved file — quantization, pruning, low-rank,
distillation, packing.

**CPU latency is IN scope** — problem statement **P49** requires a Pareto table of accuracy × size ×
CPU latency (`src/cpu_latency.py`). Still out of scope: token pruning, KV cache, early exit —
they change the file by zero bytes.

**Don't drift.** The spine does not change. If a "better idea" appears, it is a Phase 4 result or
a separate project, not a replacement.

## Where work happens

- **Only `/workspace/vit-compression`** on the GPU box. Never touch anything else in `/workspace`
  (shared machine).
- Mac side: this repository.
- Claude has **no shell on the GPU box**. Give commands to paste; never claim to have run them.

## The machine (verified 2026-09-11)

| Fact | Value | Consequence |
|---|---|---|
| GPU | H100 80GB, **MIG 3g.40gb slice** | ~A100-class. No multi-GPU. |
| Disk | `/workspace` NFS, 191 GB free | Shared machine. Don't fill it. |
| Small-file read | **1494 files/sec** | Too slow for loose JPEGs. Use WebDataset tar shards. |
| CPU | 224 cores | Start `num_workers=16`, tune up. |
| RAM limit | 90 GB (cgroup) | No full-ImageNet RAM disk. |
| `/dev/shm` | **1 GB, cannot remount** (no CAP_SYS_ADMIN) | Set `mp.set_sharing_strategy('file_system')`, keep loader output **uint8**, normalise on GPU. |
| Env | torch 2.4.0 NGC, timm 1.0.29 | Already installed. No pip needed. |

Long jobs: `nohup`, not a notebook cell. Browser disconnect kills the kernel.

**GPU access ended 2026-09-25.** Evidence is in `evidence/`. The Mac has no system numpy/torch;
use a venv (`python3 -m venv`, `pip install numpy scipy torch timm`).

## Data plan

- ImageNet val (6.3 GB) + **25% class-balanced train subset** (~37 GB). Full train (150 GB) does
  not fit on the shared disk.
- **Generate ImageNet-C corruptions on the fly** from val with `imagecorruptions`. Saves 75 GB.
  Numbers won't match published mCE — fine, every route sees identical corruptions.
- Absolute accuracy will sit below the published 79.8% because recovery is on a fraction.
  **All comparisons are route-vs-route, never route-vs-published.**

## Measured numbers — use these, don't recompute

DeiT-Small, 22,050,664 params. Written to disk, read back with `os.path.getsize()`.

| Format | Group | Nominal | Measured | Fits 10 MB? |
|---|---|---|---|---|
| FP32 | — | 88.20 | 88.24 | no |
| INT8 | 256 | 22.05 | 23.04 | no |
| INT4 | 256 | 11.03 | **12.42** | **no — 24% over** |
| INT3 | 256 | 8.27 | **9.77** | yes |
| INT3 | 128 | 8.27 | 9.94 | yes, by 65 KB only |
| INT2 | 256 | 5.51 | 7.12 | yes |

- Formula error grows as bits fall: **+4.5% at INT8 → +29.1% at INT2**.
- And as models shrink: DeiT-Tiny is **+8.0% at INT8 → +53.5% at INT2**.
- "INT4" at group 32 is really **4.93 bits/param**.
- Leftovers (patch_embed + head + norms + biases, 817k params, 1.63 MB FP16) are 7.0% of an INT8
  file but **22.4% of an INT2 file**.
- Scale cost: **0.120 bits/param at group 128** (symmetric, FP16 scales). Asymmetric roughly doubles it.

**Baseline to beat: DeiT-Tiny INT8 = 6.22 MB at 72.2% ImageNet top-1** (measured; nominal says 5.72).

## The four routes

| # | Route | Measured feasibility |
|---|---|---|
| A | Uniform INT3 | 9.77 MB at g=256. Avoid g=128 (65 KB margin). |
| B | Ternary + base-3 packing | ~4.7 MB. Needs its own quantizer — **INT2 (4 levels) ≠ ternary (3 levels)**. |
| C | INT4 + structured pruning | INT4 alone can't fit. Needs full quantization **and ~14% pruning**. |
| D | Distill into DeiT-Tiny, then INT8 | 6.22 MB, 72.2%. The floor. |

**As built (post-fix, 3 seeds each):** A group 36 → 9.718 MB · B 15 layers INT8 → 9.785 MB ·
C 24.375% MLP channels pruned → 9.796 MB · D 38 layers FP16 → 9.765 MB. Spread 0.80%. The table
above is the Phase-1 feasibility plan, not what was built. Authority: `results.md` §1.1.

Control: **equal fine-tuning budget across routes**, defined as a fixed number of optimizer steps
(Kuzmin et al. 2023's control — cite it as theirs).
Evaluate: ImageNet-C, calibration (ECE), per-example failure agreement.

## Dead ends — do not revisit

| Idea | Why it's dead |
|---|---|
| Entropy coding ternary weights | FTerViT is 37.3% zeros, near-uniform. Entropy 1.580 vs 1.585 floor. Gains **0.020 bits/param**. |
| Compression *ordering* as the research question | Four papers since 2024. Crowded, and not about file size. |
| "Does compression hurt robustness" standalone | Saturated since Hooker 2019 (PIEs). |
| Compressed ViT as VLM encoder **as the spine** | Cambrian-1 showed class-supervised encoders are the weakest category — DeiT-S starts near the floor, so no dynamic range. Keep it as a secondary experiment with that caveat stated. |
| Quoting the packing win as 24% | It's **18.9%**. FTerViT's 5.81 MB = 5.51 MB codes + 0.30 MB FP32 extras; packing only touches the codes. |


## Files

| File | What |
|---|---|
| `README.md` | Map of the repo + how to explain it. Start here. |
| `docs/vit-compression-plan.md` | The plan. Scope lock at the top. |
| `src/phase1_audit.py` | The size audit. Raw binary container, not torch.save. |
| `docs/phase1_audit_explained.md` | Line-by-line walkthrough + viva answers. |
| `evidence/audit_deit/audit.json` / `audit_tiny.json` | Measured results (pretrained weights). |
| `results.md` | **Authority for every number.** Top section = the resolved claim set. |
| `docs/progress.md` | State only. |
| `proposal/` | UCS532 proposal (HTML source + PDF). |
| `docs/history/` | Superseded docs, kept so retractions are traceable. |
| `evidence/` | Everything pulled off the GPU box 2026-09-25: raw logs with per-image `correct` arrays, all 21 model files, analysis outputs, 5-architecture audit. |
| `src/offline_analysis.py` | Seed floor, per-corruption fingerprints, within-architecture tests, calibration offset. Runs on the Mac. |
| `src/arch_audit_summary.py` | Rebuilds every audited file size from tensor shapes (second derivation). |
| `docs/background/` | Early background notes. Superseded by the plan. |
