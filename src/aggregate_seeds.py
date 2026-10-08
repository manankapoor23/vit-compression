#!/usr/bin/env python3
"""
Does the rank inversion survive training randomness?

Every earlier statistic treated the four models as FIXED and bootstrapped over
images. But each model is one draw from a stochastic training process. If
seed-to-seed variance is comparable to the B-vs-D gap, the inversion is noise.

This script answers that, and nothing else.

Usage:  python src/aggregate_seeds.py --seeds 0 1 2
"""
import argparse, glob, itertools, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--logs", default="logs")
a = ap.parse_args()

runs = {}
for s in a.seeds:
    f = f"{a.logs}/eval_s{s}.json"
    if not os.path.exists(f):
        raise SystemExit(f"missing {f} -- run evaluate_routes.py --seed {s}")
    runs[s] = json.load(open(f))

routes = sorted(runs[a.seeds[0]]["results"].keys())
conds = runs[a.seeds[0]]["conditions"]
corr = [c for c in conds if c != "clean"]

def clean(s, r): return runs[s]["results"][r]["clean"]["top1"]
def rob(s, r):   return float(np.mean([runs[s]["results"][r][c]["top1"] for c in corr]))

C = {r: np.array([clean(s, r) for s in a.seeds]) for r in routes}
R = {r: np.array([rob(s, r) for s in a.seeds]) for r in routes}
n = len(a.seeds)

print(f"{n} seeds: {a.seeds}   |   {len(corr)} corruptions\n")
print("PER-ROUTE, mean +/- std across seeds")
print(f"{'route':<7}{'clean':>18}{'corrupted':>18}{'drop':>9}")
for r in routes:
    print(f"{r:<7}{C[r].mean():>11.2f} +/-{C[r].std(ddof=1):>5.2f}"
          f"{R[r].mean():>11.2f} +/-{R[r].std(ddof=1):>5.2f}"
          f"{C[r].mean()-R[r].mean():>9.2f}")

# The decisive comparison: effect size vs training noise.
print("\nEFFECT SIZE vs SEED NOISE  (gap / pooled std; >2 means the gap dominates noise)")
print(f"{'pair':<8}{'clean gap':>12}{'ratio':>8}{'signs':>8}   {'corrupt gap':>12}{'ratio':>8}{'signs':>8}")
def pooled(x, y): return float(np.sqrt((x.std(ddof=1)**2 + y.std(ddof=1)**2) / 2)) or 1e-9
rows = {}
for x, y in itertools.combinations(routes, 2):
    dc, dr = C[x] - C[y], R[x] - R[y]
    rc, rr = abs(dc.mean()) / pooled(C[x], C[y]), abs(dr.mean()) / pooled(R[x], R[y])
    sc = f"{int((dc>0).sum())}/{n}"; sr = f"{int((dr>0).sum())}/{n}"
    rows[(x, y)] = (dc, dr, rc, rr)
    print(f"{x}v{y:<6}{dc.mean():>+11.2f}{rc:>8.1f}{sc:>8}   "
          f"{dr.mean():>+11.2f}{rr:>8.1f}{sr:>8}")

print("\nRANK INVERSION -- does it hold in EVERY seed?")
per_seed = []
for s in a.seeds:
    oc = sorted(routes, key=lambda r: -clean(s, r))
    orr = sorted(routes, key=lambda r: -rob(s, r))
    per_seed.append((oc, orr))
    print(f"  seed {s}: clean {' > '.join(oc)}   |   corrupted {' > '.join(orr)}")

swaps = [(x, y) for x, y in itertools.combinations(routes, 2)
         if (C[x].mean() > C[y].mean()) != (R[x].mean() > R[y].mean())]
print()
if not swaps:
    print("  NO inversion on the seed means. The headline claim does NOT survive.")
for x, y in swaps:
    dc, dr, rc, rr = rows[(x, y)]
    consistent = sum(1 for oc, orr in per_seed
                     if (oc.index(x) < oc.index(y)) != (orr.index(x) < orr.index(y)))
    verdict = ("ROBUST" if consistent == n and rc > 2 and rr > 2 else
               "FRAGILE -- report as a trend, not a result")
    print(f"  {x} vs {y}: inverts in {consistent}/{n} seeds | "
          f"clean gap {dc.mean():+.2f} ({rc:.1f}x noise) | "
          f"corrupted gap {dr.mean():+.2f} ({rr:.1f}x noise)")
    print(f"    -> {verdict}")

print(f"\nCAVEAT: {n} seeds is too few for a significance test ON seed variance.")
print("Report mean +/- std and sign consistency. Do NOT quote a p-value over seeds.")
if n < 5:
    print("If any ratio above is under 3, run more seeds before claiming robustness.")
