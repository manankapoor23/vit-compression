#!/usr/bin/env python3
"""
Offline analyses that answer the three reviewer attacks found in the 2026-09-25 literature check.
Runs on the saved per-image arrays in evidence/logs/ -- no GPU, no images, no models.

  ATTACK 1  Underspecification (D'Amour et al. 2020): models that differ only in SEED already
            disagree on many images and on corruptions. Is route-vs-route divergence bigger than
            seed-vs-seed divergence?  -> Section 1.
  ATTACK 2  "Accuracy on the Line" (Miller et al. 2021): synthetic corruptions are where the
            clean->corrupted correlation is known to be weaker, and route D is a different
            architecture (DeiT-Tiny). Is the B-vs-D inversion broad or driven by a few
            corruptions, and does anything survive WITHIN one architecture (A, B, C are all
            DeiT-S)?  -> Sections 2 and 3.
  ATTACK 3  Calibration differences are removable by temperature scaling (Guo et al. 2017).
            Is our ECE gap a uniform confidence offset?  -> Section 4.

Predictions, written BEFORE running (CLAUDE.md rule 1 and 3.5):
  P1  Within-route (seed) correctness disagreement on clean: 3-6%. Cross-route: 9-14%.
      If seed disagreement >= cross-route, the "routes fail on different images" claim is dead.
      Fragile direction: a LOW seed floor is what we need, so a high one hurts us.
  P2  D beats B in >= 8 of 14 corruptions; the noise family shows the largest D-minus-B gap.
  P3  Within DeiT-S, B's deficit grows under corruption (difference-in-differences < 0 with a CI
      excluding zero). Whether it grows MORE than the A-C local slope predicts: expect
      inconclusive, because the A-C clean gap is only ~0.8 points.
  P4  ECE == accuracy - mean confidence to within rounding, in every condition -> every bin is
      underconfident -> a single temperature would remove most of it.

Statistics (CLAUDE.md section 3): images are the unit; every model saw identical images, so
per-image bootstrap (paired) or McNemar exact. Bonferroni stated per family of tests.
No p-value is ever computed over seeds -- seeds give mean +/- sd and sign consistency only.

Usage:  python src/offline_analysis.py [--logs evidence/logs] [--out evidence/analysis]
"""
import argparse, itertools, json, os
import numpy as np
from scipy.stats import binomtest, norm

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default="evidence/logs")
ap.add_argument("--out", default="evidence/analysis")
ap.add_argument("--boots", type=int, default=10000)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(0)
ROUTES, SEEDS = "ABCD", (0, 1, 2)
DEIT_S = "ABC"                          # D is DeiT-Tiny: a different architecture
FAMILY = {"gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
          "defocus_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur",
          "snow": "weather", "frost": "weather", "fog": "weather", "brightness": "weather",
          "contrast": "digital", "elastic_transform": "digital", "pixelate": "digital",
          "jpeg_compression": "digital"}
OUT = {}

def load(name):
    return json.load(open(os.path.join(args.logs, name)))

clean50 = {s: load(f"evalclean_s{s}.json")["results"] for s in SEEDS}      # n = 50,000
corr5 = {s: load(f"eval_s{s}.json") for s in SEEDS}                         # n = 5,000 x 15
CONDS = [c for c in corr5[0]["conditions"] if c != "clean"]
assert len(CONDS) == 14, CONDS

def v50(r, s):  return np.array(clean50[s][r]["clean"]["correct"], dtype=bool)
def v5(r, s, c): return np.array(corr5[s]["results"][r][c]["correct"], dtype=bool)

def boot_ci(per_unit, alpha=0.05, B=None):
    """Percentile CI for the MEAN of a per-image quantity, resampling images."""
    B = B or args.boots
    x = np.asarray(per_unit, dtype=np.float64); n = len(x)
    means = np.empty(B)
    for i in range(0, B, 200):                       # chunked: 200 x 50,000 ~ 80 MB of indices
        k = min(200, B - i)
        means[i:i + k] = x[rng.integers(0, n, size=(k, n))].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)

def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p

# ============================================================================ 1. SEED vs ROUTE
print("=" * 92)
print("1. DO ROUTES DISAGREE MORE THAN SEEDS?   (control for underspecification)")
print("=" * 92)
print("Disagreement = fraction of images where exactly one of the two models is correct.")
print("Within-route pairs use different seeds; cross-route pairs are restricted to DIFFERENT seeds")
print("too, so both sides differ in seed and only the cross side also differs in route.\n")

def disagree_vec(a, b): return (a != b).astype(np.float64)
def jaccard(a, b):
    bw = (~a & ~b).sum(); un = (~a | ~b).sum()
    return float(bw / un) if un else float("nan")

sec1 = {}
for label, getv, n_img in (("clean  (n=50,000)", lambda r, s: v50(r, s), 50000),
                           ("corrupt (n=5,000 x 14)", None, 5000)):
    is_clean = getv is not None
    # per-image disagreement vectors
    def dvec(r1, s1, r2, s2):
        if is_clean:
            return disagree_vec(v50(r1, s1), v50(r2, s2))
        return np.mean([disagree_vec(v5(r1, s1, c), v5(r2, s2, c)) for c in CONDS], axis=0)
    def jac(r1, s1, r2, s2):
        if is_clean:
            return jaccard(v50(r1, s1), v50(r2, s2))
        a = np.concatenate([v5(r1, s1, c) for c in CONDS])
        b = np.concatenate([v5(r2, s2, c) for c in CONDS])
        return jaccard(a, b)

    within = {r: [dvec(r, s1, r, s2) for s1, s2 in itertools.combinations(SEEDS, 2)]
              for r in ROUTES}
    within_j = {r: [jac(r, s1, r, s2) for s1, s2 in itertools.combinations(SEEDS, 2)]
                for r in ROUTES}
    print(f"--- {label}")
    print(f"  {'route':<6}{'seed-vs-seed disagreement':>28}{'seed Jaccard':>16}")
    for r in ROUTES:
        m = [100 * x.mean() for x in within[r]]
        print(f"  {r:<6}{np.mean(m):>20.2f}%  [{min(m):.2f}-{max(m):.2f}]"
              f"{np.mean(within_j[r]):>12.3f}")
    print(f"\n  {'pair':<6}{'cross-route':>12}{'seed floor':>12}{'excess':>9}"
          f"{'95% CI of excess':>22}{'ratio':>8}{'cross J':>9}")
    rows = {}
    for x, y in itertools.combinations(ROUTES, 2):
        cross = [dvec(x, s1, y, s2) for s1 in SEEDS for s2 in SEEDS if s1 != s2]
        cross_j = [jac(x, s1, y, s2) for s1 in SEEDS for s2 in SEEDS if s1 != s2]
        floor = np.mean(within[x] + within[y], axis=0)       # per-image seed floor, both routes
        excess_img = np.mean(cross, axis=0) - floor
        m, lo, hi = boot_ci(excess_img, B=2000 if is_clean else args.boots)
        c_mean, f_mean = 100 * np.mean(cross), 100 * floor.mean()
        rows[f"{x}-{y}"] = dict(cross=c_mean, floor=f_mean, excess=100 * m,
                                ci=[100 * lo, 100 * hi], ratio=c_mean / f_mean,
                                cross_jaccard=float(np.mean(cross_j)))
        print(f"  {x}-{y:<4}{c_mean:>11.2f}%{f_mean:>11.2f}%{100*m:>+8.2f}"
              f"{f'[{100*lo:+.2f}, {100*hi:+.2f}]':>22}{c_mean/f_mean:>7.1f}x"
              f"{np.mean(cross_j):>9.3f}")
    sec1[label.split()[0]] = dict(within={r: 100 * float(np.mean([x.mean() for x in within[r]]))
                                          for r in ROUTES},
                                  within_jaccard={r: float(np.mean(within_j[r])) for r in ROUTES},
                                  pairs=rows)
    print()
OUT["1_seed_vs_route"] = sec1

# ============================================================== 2. INVERSION, CORRUPTION BY CORRUPTION
print("=" * 92)
print("2. IS THE B-vs-D INVERSION BROAD, OR DRIVEN BY A FEW CORRUPTIONS?")
print("=" * 92)
ALPHA14 = 0.05 / 14
print(f"McNemar exact per corruption per seed; Bonferroni over 14 corruptions: p < {ALPHA14:.4f}\n")
print(f"  {'corruption':<20}{'family':<9}" + "".join(f"{r:>8}" for r in ROUTES)
      + f"{'D-B':>8}{'signs':>7}{'sig/3':>7}")
sec2 = {}
for c in CONDS:
    acc = {r: [100 * v5(r, s, c).mean() for s in SEEDS] for r in ROUTES}
    gaps = [acc["D"][i] - acc["B"][i] for i in range(3)]
    sig = sum(1 for s in SEEDS if mcnemar(v5("D", s, c), v5("B", s, c))[2] < ALPHA14
              and gaps[s] > 0)
    signs = sum(g > 0 for g in gaps)
    name = c.replace("_s3", "")
    sec2[name] = dict(family=FAMILY[name], acc={r: float(np.mean(acc[r])) for r in ROUTES},
                      d_minus_b=float(np.mean(gaps)), d_ahead_seeds=signs, significant_seeds=sig)
    print(f"  {name:<20}{FAMILY[name]:<9}" + "".join(f"{np.mean(acc[r]):>8.2f}" for r in ROUTES)
          + f"{np.mean(gaps):>+8.2f}{signs:>5}/3{sig:>5}/3")
n_d_ahead = sum(1 for v in sec2.values() if v["d_ahead_seeds"] == 3)
n_d_sig = sum(1 for v in sec2.values() if v["significant_seeds"] == 3)
n_b_ahead = sum(1 for v in sec2.values() if v["d_ahead_seeds"] == 0)
print(f"\n  D ahead of B in all 3 seeds: {n_d_ahead}/14 corruptions "
      f"(significant in all 3 seeds: {n_d_sig}/14). B ahead in all 3 seeds: {n_b_ahead}/14.")
print("\n  By family (mean over corruptions and seeds):")
fam_out = {}
for f in ("noise", "blur", "weather", "digital"):
    cs = [k for k, v in sec2.items() if v["family"] == f]
    fam_out[f] = {r: float(np.mean([sec2[k]["acc"][r] for k in cs])) for r in ROUTES}
    fam_out[f]["d_minus_b"] = float(np.mean([sec2[k]["d_minus_b"] for k in cs]))
    print(f"    {f:<8} (n={len(cs)})  " + "  ".join(f"{r} {fam_out[f][r]:.2f}" for r in ROUTES)
          + f"   D-B {fam_out[f]['d_minus_b']:+.2f}")
OUT["2_per_corruption"] = dict(per_corruption=sec2, by_family=fam_out,
                               d_ahead_all_seeds=n_d_ahead, d_sig_all_seeds=n_d_sig,
                               b_ahead_all_seeds=n_b_ahead)

# ====================================================== 3. WITHIN ONE ARCHITECTURE: GAP AMPLIFICATION
print("\n" + "=" * 92)
print("3. WITHIN DeiT-S ONLY (A, B, C): DOES B'S DEFICIT GROW UNDER CORRUPTION?")
print("=" * 92)
print("Same 5,000 images for clean and corrupted, so the difference-in-differences is paired:")
print("  DiD = (x - y)_corrupted - (x - y)_clean, per image, bootstrapped over images.")
print("Amplification = corrupted gap / clean gap. Bonferroni over 3 pairs x 3 seeds = 9 tests.\n")
ALPHA9 = 0.05 / 9
sec3 = {}
for x, y in (("A", "B"), ("C", "B"), ("A", "C"), ("A", "D"), ("C", "D"), ("D", "B")):
    res = []
    for s in SEEDS:
        dc = v5(x, s, "clean").astype(float) - v5(y, s, "clean").astype(float)
        dk = np.mean([v5(x, s, c).astype(float) - v5(y, s, c).astype(float) for c in CONDS], 0)
        m, lo, hi = boot_ci(dk - dc, alpha=ALPHA9)
        res.append(dict(clean_gap=100 * dc.mean(), corr_gap=100 * dk.mean(),
                        did=100 * m, ci=[100 * lo, 100 * hi],
                        amp=float(dk.mean() / dc.mean()) if dc.mean() else float("nan")))
    sec3[f"{x}-{y}"] = res
    tag = "within DeiT-S" if x in DEIT_S and y in DEIT_S else "crosses architecture"
    print(f"  {x} vs {y}  ({tag})")
    for s, r_ in zip(SEEDS, res):
        excl = "excludes 0" if (r_["ci"][0] > 0 or r_["ci"][1] < 0) else "includes 0"
        print(f"    seed {s}: clean gap {r_['clean_gap']:+6.2f}  corrupted gap {r_['corr_gap']:+6.2f}"
              f"  DiD {r_['did']:+6.2f}  CI [{r_['ci'][0]:+.2f}, {r_['ci'][1]:+.2f}] {excl}"
              f"  amp {r_['amp']:.2f}x")

# Off-the-line test: does B's amplification exceed the A-C local slope? Bootstrap jointly.
print("\n  Off-the-line check, probit scale (Miller et al. fit lines in probit space).")
print("  Line through A and C (same architecture, same quantizer family), predict B, report residual.")
print("  Joint image bootstrap; 2 reference points only -> expect a wide interval.")
sec3_line = []
for s in SEEDS:
    cl = {r: v5(r, s, "clean").astype(float) for r in DEIT_S}
    co = {r: np.mean([v5(r, s, c).astype(float) for c in CONDS], 0) for r in DEIT_S}
    def resid(idx):
        P = lambda a: norm.ppf(np.clip(a, 1e-4, 1 - 1e-4))
        xa, xc, xb = (P(cl[r][idx].mean()) for r in "ACB")
        ya, yc, yb = (P(co[r][idx].mean()) for r in "ACB")
        slope = (ya - yc) / (xa - xc) if xa != xc else np.nan
        pred_b = ya + slope * (xb - xa)
        return 100 * (co["B"][idx].mean() - norm.cdf(pred_b)), slope
    full = resid(np.arange(5000))
    bs = [resid(rng.integers(0, 5000, 5000)) for _ in range(2000)]
    rs = np.array([b[0] for b in bs]); sl = np.array([b[1] for b in bs])
    lo, hi = np.nanpercentile(rs, [2.5, 97.5])
    slo, shi = np.nanpercentile(sl, [2.5, 97.5])
    sec3_line.append(dict(residual=float(full[0]), ci=[float(lo), float(hi)],
                          slope=float(full[1]), slope_ci=[float(slo), float(shi)]))
    print(f"    seed {s}: B residual {full[0]:+.2f} pts  95% CI [{lo:+.2f}, {hi:+.2f}]"
          f"   probit slope {full[1]:.2f} [{slo:.2f}, {shi:.2f}]")
OUT["3_within_architecture"] = dict(did=sec3, off_the_line=sec3_line)

# ============================================================================ 4. CALIBRATION
print("\n" + "=" * 92)
print("4. IS THE CALIBRATION GAP A UNIFORM OFFSET?  (ECE vs accuracy - mean confidence)")
print("=" * 92)
print("If ECE == acc - conf, every confidence bin is underconfident (same sign), i.e. a global")
print("shift that one temperature parameter largely removes (Guo et al. 2017).\n")
sec4 = {}
print(f"  {'route':<6}{'ECE clean50k':>14}{'acc-conf':>10}{'|diff|':>9}"
      f"{'ECE corr (mean14)':>19}{'acc-conf':>10}{'max |diff| over 15 conds':>27}")
for r in ROUTES:
    e50 = np.mean([clean50[s][r]["clean"]["ece"] for s in SEEDS])
    g50 = np.mean([clean50[s][r]["clean"]["top1"] - clean50[s][r]["clean"]["mean_conf"] for s in SEEDS])
    ek = np.mean([corr5[s]["results"][r][c]["ece"] for s in SEEDS for c in CONDS])
    gk = np.mean([corr5[s]["results"][r][c]["top1"] - corr5[s]["results"][r][c]["mean_conf"]
                  for s in SEEDS for c in CONDS])
    worst = max(abs(corr5[s]["results"][r][c]["ece"]
                    - abs(corr5[s]["results"][r][c]["top1"] - corr5[s]["results"][r][c]["mean_conf"]))
                for s in SEEDS for c in ["clean"] + CONDS)
    sec4[r] = dict(ece_clean=float(e50), gap_clean=float(g50), ece_corr=float(ek),
                   gap_corr=float(gk), worst_abs_diff=float(worst))
    print(f"  {r:<6}{e50:>14.3f}{g50:>+10.3f}{abs(e50-abs(g50)):>9.4f}{ek:>19.3f}{gk:>+10.3f}{worst:>27.4f}")
print("  (acc-conf > 0 means UNDERconfident. Training recipe: label_smoothing=0.1 on all routes;")
print("   route D adds KL to the teacher at temperature 2 -- see src/train_routes.py:262-267.)")
OUT["4_calibration"] = sec4

# ================================================ 5. WITHIN DeiT-S: CORRUPTION-SPECIFIC FINGERPRINTS
print("\n" + "=" * 92)
print("5. WITHIN DeiT-S: DOES ANY CORRUPTION REVERSE THE CLEAN ORDER?  (the Miller-proof test)")
print("=" * 92)
print("A, B, C share architecture, pretrained weights, byte budget and step budget. On a per-")
print("corruption accuracy-on-the-line, the lower-clean model must stay lower on EVERY corruption.")
ALPHA42 = 0.05 / 42
print(f"3 pairs x 14 corruptions = 42 tests. McNemar exact, Bonferroni p < {ALPHA42:.5f}, per seed.\n")
sec5 = {}
for x, y in (("A", "B"), ("C", "B"), ("A", "C")):      # x has the higher clean accuracy
    rows = {}
    for c in CONDS:
        g, sig_rev, sig_fwd = [], 0, 0
        for s in SEEDS:
            a_, b_ = v5(x, s, c), v5(y, s, c)
            n01, n10, p = mcnemar(a_, b_)
            g.append(100 * (a_.mean() - b_.mean()))
            if p < ALPHA42: sig_rev += g[-1] < 0; sig_fwd += g[-1] > 0
        rows[c.replace("_s3", "")] = dict(gap=float(np.mean(g)), reversed_seeds=int(sum(v < 0 for v in g)),
                                          sig_reversed_seeds=int(sig_rev), sig_forward_seeds=int(sig_fwd))
    sec5[f"{x}-{y}"] = rows
    rev = [k for k, v in rows.items() if v["reversed_seeds"] == 3]
    srev = [k for k, v in rows.items() if v["sig_reversed_seeds"] == 3]
    gaps = sorted(rows.items(), key=lambda kv: kv[1]["gap"])
    print(f"  {x} over {y}: clean says {x} ahead. Per-corruption gap ranges "
          f"{gaps[0][1]['gap']:+.2f} ({gaps[0][0]}) to {gaps[-1][1]['gap']:+.2f} ({gaps[-1][0]})")
    print(f"      reversed in 3/3 seeds: {rev or 'none'}   | significant reversal 3/3: {srev or 'none'}")

# Family contrast. Families are the standard ImageNet-C taxonomy, fixed in FAMILY before any
# result was seen. The noise-vs-digital CONTRAST was chosen after seeing section 2: exploratory.
print("\n  Family contrast of the ternary deficit (families pre-specified; contrast chosen post hoc):")
print("  per image: mean_noise(x - B) - mean_digital(x - B), bootstrapped over the 5,000 images.")
noise = [c for c in CONDS if FAMILY[c.replace('_s3', '')] == "noise"]
digit = [c for c in CONDS if FAMILY[c.replace('_s3', '')] == "digital"]
fam5 = {}
for x in ("A", "C"):
    out_s = []
    for s in SEEDS:
        gn = np.mean([v5(x, s, c).astype(float) - v5("B", s, c).astype(float) for c in noise], 0)
        gd = np.mean([v5(x, s, c).astype(float) - v5("B", s, c).astype(float) for c in digit], 0)
        m, lo, hi = boot_ci(gn - gd, alpha=0.05 / 6)
        out_s.append(dict(noise_gap=100 * gn.mean(), digital_gap=100 * gd.mean(),
                          contrast=100 * m, ci=[100 * lo, 100 * hi]))
        print(f"    {x}-B seed {s}: noise gap {100*gn.mean():+6.2f}  digital gap {100*gd.mean():+6.2f}"
              f"  difference {100*m:+6.2f}  CI(Bonf 6) [{100*lo:+.2f}, {100*hi:+.2f}]")
    fam5[f"{x}-B"] = out_s
OUT["5_within_deit_s_fingerprint"] = dict(per_corruption=sec5, family_contrast=fam5)

json.dump(OUT, open(os.path.join(args.out, "offline_analysis.json"), "w"), indent=2,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"\nSaved {os.path.join(args.out, 'offline_analysis.json')}")
