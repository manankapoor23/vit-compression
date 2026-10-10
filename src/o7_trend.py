#!/usr/bin/env python3
"""O7: is ternary (route B) below the clean-to-damaged trend line of same-shape models?

Trend line: 7 DeiT-Small models (route A at group sizes 32-512, plus routes A and C of the same seed),
fitted on probit axes (Miller et al. 2021), clean vs damaged accuracy on the same 5,000 photos.
Residual = ternary's damaged accuracy minus the line's prediction at ternary's clean accuracy.
Pre-registered rule: "below the line" only if the 95% bootstrap range excludes zero.

Second check: Bonferroni-strict range for 3 seeds (98.3%), and the same fit on plain (non-probit) axes.
Input: evidence/rescore/rescore.json. Output: evidence/rescore/o6_line.json.
"""
import json
import numpy as np
from scipy.stats import norm

R = json.load(open("evidence/rescore/rescore.json"))["results"]
NOISE = ["gaussian_noise_s3", "shot_noise_s3", "impulse_noise_s3"]
ALL14 = [k for k in R["A-s0"] if k.endswith("_s3")]
SWEEP = [f"A-g{g}-s0" for g in (32, 64, 128, 256, 512)]
N = 5000


def probit(a):
    return norm.ppf(np.clip(a, 1e-4, 1 - 1e-4))


def clean(k):
    return np.array(R[k]["clean"]["correct"][:N], float)     # same 5,000 photos as the damage set


def damaged(k, conds):
    return np.mean([np.array(R[k][c]["correct"], float) for c in conds], 0)


def residual(X, Y, ref, target, idx, axis=probit, inverse=norm.cdf):
    x = np.array([axis(X[k][idx].mean()) for k in ref])
    y = np.array([axis(Y[k][idx].mean()) for k in ref])
    slope, icept = np.polyfit(x, y, 1)
    predicted = inverse(icept + slope * axis(X[target][idx].mean()))
    return 100 * (Y[target][idx].mean() - predicted)


def main():
    rng = np.random.default_rng(0)
    out = {}
    for fam, conds in (("noise", NOISE), ("all 14", ALL14)):
        for s in (0, 1, 2):
            ref = SWEEP + [f"A-s{s}", f"C-s{s}"]
            keys = ref + [f"B-s{s}"]
            X = {k: clean(k) for k in keys}; Y = {k: damaged(k, conds) for k in keys}
            full = residual(X, Y, ref, f"B-s{s}", np.arange(N))
            boots = [residual(X, Y, ref, f"B-s{s}", rng.integers(0, N, N)) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out[f"{fam}_s{s}"] = dict(residual=full, ci=[lo, hi])
            verdict = "excludes 0" if hi < 0 or lo > 0 else "includes 0"
            print(f"{fam:6} seed {s}: ternary {full:+.2f} pts from the line, 95% range [{lo:+.2f}, {hi:+.2f}]  {verdict}")
    json.dump(out, open("evidence/rescore/o6_line.json", "w"), indent=1)

    print("\nSecond check on noise: 98.3% range (Bonferroni, 3 seeds), probit axes and plain axes")
    rng = np.random.default_rng(1)
    plain = lambda a: a
    for s in (0, 1, 2):
        ref = SWEEP + [f"A-s{s}", f"C-s{s}"]
        keys = ref + [f"B-s{s}"]
        X = {k: clean(k) for k in keys}; Y = {k: damaged(k, NOISE) for k in keys}
        lin = [residual(X, Y, ref, f"B-s{s}", rng.integers(0, N, N)) for _ in range(3000)]
        lo, hi = np.percentile(lin, [0.833, 99.167])
        raw = residual(X, Y, ref, f"B-s{s}", np.arange(N), axis=plain, inverse=plain)
        rb = [residual(X, Y, ref, f"B-s{s}", rng.integers(0, N, N), axis=plain, inverse=plain) for _ in range(3000)]
        rlo, rhi = np.percentile(rb, [0.833, 99.167])
        print(f"seed {s}: probit 98.3% [{lo:+.2f}, {hi:+.2f}]  |  plain axes {raw:+.2f}, 98.3% [{rlo:+.2f}, {rhi:+.2f}]")


if __name__ == "__main__":
    main()
