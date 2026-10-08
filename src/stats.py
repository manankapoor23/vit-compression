#!/usr/bin/env python3
"""
Proper significance testing for logs/eval.json.

Fixes three errors in the quick analysis:

  1. Corrupted samples are NOT independent -- 14 corruptions of the SAME images.
     Effective n is the number of IMAGES (5000), not image-corruption pairs
     (70000). We bootstrap over images, which respects that automatically.
  2. The data is PAIRED (every model saw identical images), so clean accuracy
     is compared with McNemar's exact test, not a two-proportion z-test.
  3. 6 pairs x 2 conditions = 12 tests, so p-values are Bonferroni-corrected.

Usage:  python src/stats.py
"""
import itertools, json, sys
import numpy as np
from scipy.stats import binomtest

J = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "logs/eval.json"))
res, routes = J["results"], sorted(J["results"].keys())
conds = J["conditions"]
corr_conds = [c for c in conds if c != "clean"]
N_TESTS = len(list(itertools.combinations(routes, 2))) * 2
ALPHA = 0.05

def vec(r, c): return np.array(res[r][c]["correct"], dtype=bool)

print(f"{len(routes)} models | {len(vec(routes[0],'clean'))} images | "
      f"{len(corr_conds)} corruptions")
print(f"Bonferroni: {N_TESTS} tests, so significance requires p < {ALPHA/N_TESTS:.4f}\n")

# ---- clean accuracy: McNemar exact (paired) -------------------------------
print("CLEAN ACCURACY -- McNemar exact test (paired)")
print(f"{'pair':<8}{'acc1':>8}{'acc2':>8}{'b':>7}{'c':>7}{'p':>12}  verdict")
for x, y in itertools.combinations(routes, 2):
    a, b_ = vec(x, "clean"), vec(y, "clean")
    b = int((a & ~b_).sum())        # x right, y wrong
    c = int((~a & b_).sum())        # x wrong, y right
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    sig = "DIFFERENT" if p < ALPHA / N_TESTS else "not distinguishable"
    print(f"{x}v{y:<6}{100*a.mean():>7.2f}%{100*b_.mean():>7.2f}%{b:>7}{c:>7}{p:>12.2e}  {sig}")

# ---- corruption robustness: bootstrap over IMAGES --------------------------
if not corr_conds:
    print("\nNo corruption conditions in this file (clean-only run).")
    print("Clean McNemar results above are complete; skipping the bootstrap and")
    print("the rank-inversion check, which both need corrupted conditions.")
    raise SystemExit(0)

print(f"\nCORRUPTION ROBUSTNESS -- bootstrap over images (10,000 resamples)")
print("per-image accuracy averaged across corruptions, so images are the unit")
print(f"{'pair':<8}{'mean1':>8}{'mean2':>8}{'diff':>8}{'95% CI':>18}  verdict")
per_img = {r: np.stack([vec(r, c) for c in corr_conds]).mean(0) for r in routes}
rng = np.random.default_rng(0)
n = len(per_img[routes[0]])
idx = rng.integers(0, n, size=(10000, n))
for x, y in itertools.combinations(routes, 2):
    d = per_img[x] - per_img[y]
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [100*ALPHA/N_TESTS/2, 100*(1-ALPHA/N_TESTS/2)])
    sig = "DIFFERENT" if (lo > 0 or hi < 0) else "not distinguishable"
    print(f"{x}v{y:<6}{100*per_img[x].mean():>7.2f}%{100*per_img[y].mean():>7.2f}%"
          f"{100*d.mean():>+7.2f}%{f'[{100*lo:+.2f},{100*hi:+.2f}]':>18}  {sig}")

# ---- the headline: does the ranking actually invert? -----------------------
print("\nRANK INVERSION CHECK")
clean = {r: vec(r, "clean").mean() for r in routes}
rob = {r: per_img[r].mean() for r in routes}
rc, rr = sorted(routes, key=lambda r: -clean[r]), sorted(routes, key=lambda r: -rob[r])
print(f"  by clean     : {' > '.join(rc)}")
print(f"  by corrupted : {' > '.join(rr)}")
swaps = [(x, y) for x, y in itertools.combinations(routes, 2)
         if (clean[x] > clean[y]) != (rob[x] > rob[y])]
if not swaps:
    print("  NO inversion -- the rankings agree. The headline claim does not hold.")
for x, y in swaps:
    a, b_ = vec(x, "clean"), vec(y, "clean")
    bb, cc = int((a & ~b_).sum()), int((~a & b_).sum())
    pc = binomtest(bb, bb + cc, 0.5).pvalue if (bb + cc) else 1.0
    d = per_img[x] - per_img[y]; boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [100*ALPHA/N_TESTS/2, 100*(1-ALPHA/N_TESTS/2)])
    both = pc < ALPHA/N_TESTS and (lo > 0 or hi < 0)
    print(f"  {x} vs {y}: clean p={pc:.2e} | corrupted CI [{100*lo:+.2f},{100*hi:+.2f}]")
    print(f"    -> {'BOTH significant: the inversion is real' if both else 'NOT both significant: state it as a trend, not a result'}")
