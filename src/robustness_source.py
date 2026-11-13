#!/usr/bin/env python3
"""Where does route D's lead on damaged photos come from: the smaller network, or distillation?
Compares uncompressed DeiT-Tiny (timm weights) with route D (DeiT-Tiny distilled from DeiT-S, then 8-bit)
and route B (ternary DeiT-S). Input: evidence/rescore/rescore.json (same photos and damage for every model)."""
import json
import numpy as np

R = json.load(open("evidence/rescore/rescore.json"))["results"]
DMG = [k for k in R["A-s0"] if k.endswith("_s3")]
FAM = {"noise": ["gaussian_noise", "shot_noise", "impulse_noise"], "blur": ["defocus_blur", "motion_blur", "zoom_blur"],
       "weather": ["snow", "frost", "fog", "brightness"], "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]}
MODELS = {"DeiT-S 32-bit": ["deit_s_fp32"], "DeiT-Tiny 32-bit": ["deit_tiny_fp32"],
          "D distilled+8bit": [f"D-s{s}" for s in (0, 1, 2)], "B ternary": [f"B-s{s}" for s in (0, 1, 2)],
          "A 3-bit": [f"A-s{s}" for s in (0, 1, 2)]}


def summary(keys):
    clean = np.mean([R[k]["clean"]["top1"] for k in keys])
    damaged = np.mean([np.mean([R[k][c]["top1"] for c in DMG]) for k in keys])
    family = {f: np.mean([np.mean([R[k][f"{c}_s3"]["top1"] for c in cs]) for k in keys]) for f, cs in FAM.items()}
    return clean, damaged, family


out = {}
print(f"{'model':<18}{'clean':>7}{'damaged':>9}{'drop':>7}" + "".join(f"{f:>9}" for f in FAM))
for name, keys in MODELS.items():
    c, d, fam = summary(keys); out[name] = (c, d)
    print(f"{name:<18}{c:>7.2f}{d:>9.2f}{c - d:>7.2f}" + "".join(f"{fam[f]:>9.2f}" for f in FAM))
tiny, dd, b = out["DeiT-Tiny 32-bit"], out["D distilled+8bit"], out["B ternary"]
print(f"\nuncompressed DeiT-Tiny vs B on damaged photos: {tiny[1] - b[1]:+.2f} pts")
print(f"route D vs B on damaged photos:                {dd[1] - b[1]:+.2f} pts")
print(f"distillation + 8-bit changed DeiT-Tiny by:     {dd[1] - tiny[1]:+.2f} pts damaged, {dd[0] - tiny[0]:+.2f} pts clean")
print("-> D's lead over B on damaged photos comes from the smaller network, not from distillation.")
