#!/usr/bin/env python3
"""Ternary (B) vs distilled DeiT-Tiny (D): the clean/damaged reversal, per seed (report Sec. 5.4).

Reads evidence/rescore/rescore.json (every model re-scored from its .bin file).
Clean:   B - D accuracy and McNemar's exact test (binomial test on the discordant images).
Damaged: D - B accuracy averaged over the 14 corruptions; each image's score is its mean
         correctness over the 14 corruptions, so the 5,000 images (not 70,000 pairs) are the unit.
         Image bootstrap, 10,000 resamples, seed 0: the plain 95% interval and the
         Bonferroni-corrected interval for 6 tests (1 - 0.05/6 = 99.17%).
"""
import json
import numpy as np
from scipy.stats import binomtest

R = json.load(open("evidence/rescore/rescore.json"))["results"]
CORR = [k for k in R["A-s0"] if k.endswith("_s3")]
ALPHA = 0.05 / 6  # 6 tests: clean and damaged, 3 seeds


def vec(key, cond):
    return np.array(R[key][cond]["correct"], dtype=bool)


print(f"{len(CORR)} corruptions; Bonferroni bar p < {ALPHA:.4f}, corrected interval {100 * (1 - ALPHA):.2f}%")
for s in (0, 1, 2):
    b, d = vec(f"B-s{s}", "clean"), vec(f"D-s{s}", "clean")
    only_b, only_d = int(np.sum(b & ~d)), int(np.sum(d & ~b))
    p = binomtest(only_b, only_b + only_d, 0.5).pvalue
    clean = 100 * (b.mean() - d.mean())

    db = np.stack([vec(f"D-s{s}", c).astype(float) - vec(f"B-s{s}", c) for c in CORR]).mean(0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, db.size, size=(10000, db.size))
    boots = 100 * db[idx].mean(1)
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    loB, hiB = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    print(f"seed {s}: clean B-D {clean:+.2f} (McNemar p={p:.1e}) | damaged D-B {100 * db.mean():+.2f} "
          f"95% [{lo95:.2f}, {hi95:.2f}]  99.17% [{loB:.2f}, {hiB:.2f}]")
