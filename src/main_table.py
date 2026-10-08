#!/usr/bin/env python3
"""Main result tables for P49, built only from measured files.

Table 1: the four routes against the uncompressed model (size x accuracy x CPU time).
Table 2: what each compression stage buys and costs.

Sizes:    os.path.getsize of the saved files (evidence/models, evidence/audit_deit).
Accuracy: evidence/rescore/rescore.json (Kaggle T4, every model decoded from its .bin).
CPU time: evidence/rescore/cpu_latency_x86.json and evidence/latency/cpu_latency.json (M2),
          fp32, 4 threads, batch 1, median of 50 runs.
"""
import json, os
import numpy as np

R = json.load(open("evidence/rescore/rescore.json"))["results"]
CORR = [k for k in R["A-s0"] if k.endswith("_s3")]
X86 = json.load(open("evidence/rescore/cpu_latency_x86.json"))["rows"]
M2 = json.load(open("evidence/latency/cpu_latency.json"))["rows"]
AUDIT_S = {(r["format"], r["group"]): r["bytes"] for r in json.load(open("evidence/audit_deit/audit.json"))["sections"]["sizes"]}
AUDIT_T = {(r["format"], r["group"]): r["bytes"] for r in json.load(open("evidence/audit_deit/audit_tiny.json"))["sections"]["sizes"]}
MB = 1e6


def ms(rows, prefix):
    return [r["median_ms"] for r in rows if r["model"].startswith(prefix) and r["kind"] == "fp32" and r["threads"] == 4][0]


def size(name):
    return os.path.getsize(f"evidence/models/ChotaDeiT-{name}.bin") / MB


def acc(key):
    clean = R[key]["clean"]["top1"]
    damaged = float(np.mean([R[key][c]["top1"] for c in CORR]))
    return clean, damaged


def seeds(route):
    c, d = zip(*(acc(f"{route}-s{s}") for s in (0, 1, 2)))
    return np.mean(c), min(c), max(c), np.mean(d)


base_c, base_d = acc("deit_s_fp32")
base_mb = AUDIT_S[("FP32", "-")] / MB

print("Table 1. Four routes at ~9.8 MB vs the uncompressed model "
      "(clean: 50,000 images; damaged: mean of 14 corruptions, severity 3, 5,000 images)")
print(f"{'model':<26}{'file MB':>8}{'clean %':>9}{'3-seed range':>15}{'loss':>7}{'damaged %':>11}"
      f"{'x86 ms':>8}{'M2 ms':>7}")
print(f"{'DeiT-S, uncompressed':<26}{base_mb:8.2f}{base_c:9.2f}{'':>15}{'':>7}{base_d:11.2f}"
      f"{ms(X86, 'DeiT-S baseline'):8.1f}{ms(M2, 'DeiT-S baseline'):7.1f}")
LAT = {"A": "A / B", "B": "A / B", "C": "C:", "D": "D:"}
NAME = {"A": "A  3-bit", "B": "B  ternary", "C": "C  4-bit + pruning", "D": "D  distilled DeiT-Tiny"}
for r in "ABCD":
    m, lo, hi, dmg = seeds(r)
    print(f"{NAME[r]:<26}{size(r + '-s0'):8.3f}{m:9.2f}{f'[{lo:.2f}, {hi:.2f}]':>15}{base_c - m:7.2f}{dmg:11.2f}"
          f"{ms(X86, LAT[r]):8.1f}{ms(M2, LAT[r]):7.1f}")

print("\nTable 2. What each stage buys and costs (seed 0 where a route has seeds)")
print(f"{'stage':<52}{'file MB':>8}{'clean %':>9}{'damaged %':>11}")
print(f"{'DeiT-S, 32-bit':<52}{base_mb:8.2f}{base_c:9.2f}{base_d:11.2f}")
print("route A: shrink every weight to 3 bits (group size = weights sharing one scale)")
for g in (512, 256, 128, 64):
    c, d = acc(f"A-g{g}-s0")
    print(f"{f'  3-bit, group {g}':<52}{size(f'A-g{g}-s0'):8.3f}{c:9.2f}{d:11.2f}")
c, d = acc("A-s0")
print(f"{'  3-bit, group 36  (route A as submitted)':<52}{size('A-s0'):8.3f}{c:9.2f}{d:11.2f}")
c, d = acc("A-g32-s0")
print(f"{'  3-bit, group 32  (above the matched ~9.77 MB)':<52}{size('A-g32-s0'):8.3f}{c:9.2f}{d:11.2f}")
print("route D: swap to a smaller network, then shrink it")
c, d = acc("deit_tiny_fp32")
print(f"{'  DeiT-Tiny, 32-bit (timm weights)':<52}{AUDIT_T[('FP32', '-')] / MB:8.2f}{c:9.2f}{d:11.2f}")
c, d = acc("D-s0")
print(f"{'  + distilled from DeiT-S, 8-bit (38 layers 16-bit)':<52}{size('D-s0'):8.3f}{c:9.2f}{d:11.2f}")
print("routes B and C: only the final model was saved; their intermediate stages were not measured.")
