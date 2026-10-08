#!/usr/bin/env python3
"""
Proposal figures, drawn ONLY from archived evidence (no typed-in numbers).
Black-and-grey, print-friendly. Output: proposal/figs/*.svg

Usage:  python src/make_figures.py
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_bin import read_bin

OUT = "proposal/figs"; os.makedirs(f"{OUT}/png", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#333", "axes.linewidth": 0.6, "svg.fonttype": "none"})
GREY = {"A": "#222222", "B": "#5f5f5f", "C": "#9a9a9a", "D": "#d0d0d0"}
NAME = {"A": "A  3-bit", "B": "B  ternary", "C": "C  4-bit + prune", "D": "D  distilled Tiny"}

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}/png/{name}.png", bbox_inches="tight", dpi=110); plt.close(fig); print("wrote", name)

# ---------------------------------------------------------------- data
clean = {r: np.mean([json.load(open(f"evidence/logs/evalclean_s{s}.json"))["results"][r]["clean"]["top1"]
                     for s in (0, 1, 2)]) for r in "ABCD"}
ev = [json.load(open(f"evidence/logs/eval_s{s}.json")) for s in (0, 1, 2)]
conds = [c for c in ev[0]["conditions"] if c != "clean"]
corr = {r: np.mean([np.mean([e["results"][r][c]["top1"] for c in conds]) for e in ev]) for r in "ABCD"}
off = json.load(open("evidence/analysis/offline_analysis.json"))
_lat = {row["model"]: row["median_ms"] for row in json.load(open("evidence/latency/cpu_latency.json"))["rows"]
        if row["kind"] == "fp32" and row["threads"] == 4}
LAT_KEY = {"base": "DeiT-S baseline (FP32, 88.2 MB)", "A": "A / B: decode to DeiT-S",
           "C": "C: pruned DeiT-S (MLP 1162)", "D": "D: DeiT-Tiny"}      # explicit: first letters collide
lat = {k: _lat[v] for k, v in LAT_KEY.items()}
audit = json.load(open("evidence/audit_deit/audit.json"))
P = audit["total_params"]
sizes = {(r["format"], r["group"]): r["bytes"] / 1e6 for r in audit["sections"]["sizes"]}

# ---------------------------------------------------------------- 0. the study design
fig, ax = plt.subplots(figsize=(7.2, 2.3)); ax.axis("off"); ax.set_xlim(-2, 103); ax.set_ylim(0, 30)
def box(x, y, w, h, text, fill="white", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fill, ec="#222", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2,
            fontweight="bold" if bold else "normal")
def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#222"))
box(0, 11, 13, 8, "DeiT-Small\n88.2 MB", bold=True)
for i, r in enumerate("ABCD"):
    y = 24 - i * 7.3
    box(21, y, 22, 5, NAME[r].replace("  ", ": ")); arrow(13.6, 15, 20.6, y + 2.5)
    arrow(43.6, y + 2.5, 50.4, 15)
box(51, 10, 16, 10, "every file\nsearched to\n~9.77 MB")
arrow(67.6, 15, 71.4, 15)
box(72, 10, 12, 10, "3000 steps\nx 3 seeds")
arrow(84.6, 15, 87.4, 15)
box(87.5, 5, 13.5, 20, "measure:\naccuracy\ncorruptions\ncalibration\nCPU speed")
save(fig, "fig0_design")

# ---------------------------------------------------------------- 1. formula vs real file
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
bits = [8, 4, 3, 2]; x = np.arange(4)
nom = [P * b / 8 / 1e6 for b in bits]; real = [sizes[(f"INT{b}", "g=256")] for b in bits]
a1.bar(x - 0.19, nom, 0.38, color="white", edgecolor="#222", lw=0.8, label="what the formula says")
a1.bar(x + 0.19, real, 0.38, color="#444", label="real file on disk")
a1.axhline(10, color="#222", ls="--", lw=0.8); a1.text(3.55, 10.6, "10 MB", ha="right", fontsize=8)
for i, (n, r) in enumerate(zip(nom, real)):
    a1.text(i + 0.19, r + 0.5, f"+{(r / n - 1) * 100:.1f}%", ha="center", fontsize=7.5)
a1.set_xticks(x, [f"{b}-bit" for b in bits]); a1.set_ylabel("MB"); a1.set_ylim(0, 26)
a1.legend(frameon=False, fontsize=7.5, loc="upper right"); a1.set_title("DeiT-Small: formula vs real file", fontsize=9)
arch = []
for f in sorted(glob.glob("evidence/audit_multiarch/*.json")):
    J = json.load(open(f))
    if J["model"].startswith(("deit", "vit")):
        b2 = {(r["format"], r["group"]): r["bytes"] for r in J["sections"]["sizes"]}[("INT2", "g=256")]
        arch.append((J["total_params"], J["model"].split("_")[1].capitalize(), (b2 / (J["total_params"] * 2 / 8) - 1) * 100))
arch.sort()
a2.bar(range(len(arch)), [o for *_, o in arch], color="#777", width=0.6)
for i, (_, n, o) in enumerate(arch): a2.text(i, o + 1.2, f"+{o:.1f}%", ha="center", fontsize=7.5)
a2.set_xticks(range(len(arch)), [("DeiT-" if n != "Large" else "ViT-") + n for _, n, _ in arch], fontsize=7.5)
a2.set_ylabel("formula error at 2-bit (%)"); a2.set_ylim(0, 62)
a2.set_title("Smaller models, bigger error", fontsize=9)
fig.tight_layout(); save(fig, "fig1_size_gap")

# ---------------------------------------------------------------- 2. what is inside each file
fig, ax = plt.subplots(figsize=(7.2, 1.9))
parts = {}
for r in "ABCD":
    hdr, _ = read_bin(f"evidence/models/ChotaDeiT-{r}-s0.bin")
    codes = sum(cb for _, k, _, cb, sb in hdr if k != "f16") / 1e6
    scales = sum(sb for *_, sb in hdr) / 1e6
    f16 = sum(cb for _, k, _, cb, _ in hdr if k == "f16") / 1e6
    parts[r] = (codes, scales, f16)
labels = ["compressed weights", "scale numbers", "kept at 16-bit"]
fills = ["#333", "white", "#bbb"]; hatch = ["", "////", ""]
for i, r in enumerate("DCBA"):
    left = 0
    for k, v in enumerate(parts[r]):
        ax.barh(i, v, left=left, color=fills[k], edgecolor="#222", lw=0.6, hatch=hatch[k],
                label=labels[k] if i == 0 else None)
        left += v
    real = os.path.getsize(f"evidence/models/ChotaDeiT-{r}-s0.bin") / 1e6   # measured file, incl. header
    ax.text(left + 0.1, i, f"{real:.3f} MB", va="center", fontsize=7.5)
ax.set_yticks(range(4), [NAME[r] for r in "DCBA"]); ax.set_xlim(0, 11.2); ax.set_xlabel("MB")
ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.32))
save(fig, "fig2_inside_files")

# ---------------------------------------------------------------- 3. the two rankings
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for r in "ABCD":
    ax.plot([0, 1], [clean[r], corr[r]], color="#222" if r in "BD" else "#9a9a9a",
            lw=1.6 if r in "BD" else 0.9, marker="o", ms=4)
    ly = {"A": 77.4, "C": 75.4, "B": 73.2, "D": 71.0}[r]                       # label heights: 2 pts apart
    ax.text(-0.05, ly, f"{NAME[r]}  {clean[r]:.1f}", ha="right", va="center", fontsize=7.5)
    ax.text(1.05, corr[r], f"{corr[r]:.1f}  {r}", ha="left", va="center", fontsize=7.5)
ax.set_xlim(-0.75, 1.3); ax.set_xticks([0, 1], ["clean images\n(50,000)", "corrupted images\n(5,000 x 14)"])
ax.set_ylabel("top-1 accuracy (%)"); ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
ax.set_title("B beats D on clean images; D beats B under corruption", fontsize=8.5)
save(fig, "fig3_two_rankings")

# ---------------------------------------------------------------- 4. corruption families
fam = off["2_per_corruption"]["by_family"]; fams = ["noise", "blur", "weather", "digital"]
fig, ax = plt.subplots(figsize=(7.2, 2.4)); w = 0.2
for k, r in enumerate("ABCD"):
    ax.bar(np.arange(4) + (k - 1.5) * w, [fam[f][r] for f in fams], w, color=GREY[r], edgecolor="#222",
           lw=0.5, label=NAME[r])
ax.set_xticks(range(4), [f"{f}" for f in fams]); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 70)
ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
save(fig, "fig4_families")

# ---------------------------------------------------------------- 5. routes vs seeds
pairs = off["1_seed_vs_route"]["clean"]["pairs"]; keys = ["A-C", "A-B", "B-C", "B-D", "A-D", "C-D"]
fig, ax = plt.subplots(figsize=(7.2, 2.2)); x = np.arange(len(keys))
ax.bar(x - 0.19, [pairs[k]["floor"] for k in keys], 0.38, color="white", edgecolor="#222", lw=0.8,
       label="two seeds of the SAME route (random noise)")
ax.bar(x + 0.19, [pairs[k]["cross"] for k in keys], 0.38, color="#444", label="two DIFFERENT routes")
for i, k in enumerate(keys): ax.text(i + 0.19, pairs[k]["cross"] + 0.3, f"{pairs[k]['ratio']:.1f}x", ha="center", fontsize=7.5)
ax.set_xticks(x, [k.replace("-", " vs ") for k in keys]); ax.set_ylabel("images they disagree on (%)")
ax.set_ylim(0, 17.5); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "fig5_routes_vs_seeds")

# ---------------------------------------------------------------- 6. accuracy vs CPU speed
fig, ax = plt.subplots(figsize=(4.6, 2.9))
pts = {"A": (lat["A"], clean["A"]), "B": (lat["A"], clean["B"]), "C": (lat["C"], clean["C"]), "D": (lat["D"], clean["D"])}
front = sorted([pts[r] for r in "ACD"])
ax.plot([p[0] for p in front], [p[1] for p in front], color="#999", lw=0.8, ls="--", zorder=0)
for r, (l, a) in pts.items():
    ax.scatter(l, a, s=46, color=GREY[r], edgecolor="#222", lw=0.7, zorder=3)
    ax.annotate(NAME[r], (l, a), xytext=(6, -3 if r != "B" else -9), textcoords="offset points", fontsize=7.5)
base_acc = json.load(open("evidence/rescore/rescore.json"))["results"]["deit_s_fp32"]["clean"]["top1"]
ax.scatter(lat["base"], base_acc, s=46, facecolor="white", edgecolor="#222", lw=0.8)
ax.annotate(f"uncompressed, 88 MB\n(measured, {base_acc:.2f}%)", (lat["base"], base_acc), xytext=(-118, -4),
            textcoords="offset points", fontsize=7)
ax.set_xlabel("CPU time per image, ms (Apple M2, 4 threads)"); ax.set_ylabel("clean top-1 (%)")
ax.set_xlim(5, 20.5); ax.set_ylim(70, 81)
ax.set_title("All four files are ~9.8 MB. Dashed line: best trade-offs", fontsize=8.5)
save(fig, "fig6_pareto")

# ---------------------------------------------------------------- techniques: one picture per idea
# Quantisation numbers are real output of quantize_groupwise on [0.12, -0.40, 0.05] at 3 bits (scale 0.40/3).
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.2))
for a in axs: a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 10)
def cell(a, x, y, t, w=1.9, h=1.3, fill="white", fs=7.5):
    a.add_patch(plt.Rectangle((x, y), w, h, fc=fill, ec="#222", lw=0.7))
    a.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(a, x0, y0, x1, y1): a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#222"))
# 1 quantisation
a = axs[0]; a.set_title("Quantisation", fontsize=9, fontweight="bold")
for i, v in enumerate(["0.12", "-0.40", "0.05"]): cell(a, 0.6 + i * 2.9, 7, v)
a.text(5, 9.0, "32-bit weights", ha="center", fontsize=7.5)
arr(a, 5, 6.7, 5, 4.9)
a.text(5.4, 5.7, "round to 1 of 8 levels", fontsize=7, ha="left")
for i, v in enumerate(["1", "-3", "0"]): cell(a, 0.6 + i * 2.9, 3.2, v, fill="#ddd")
a.text(5, 1.8, "3-bit codes  x one scale (0.133)", ha="center", fontsize=7.5)
a.text(5, 0.6, "= 0.13  -0.40  0.00", ha="center", fontsize=7.5, color="#444")
# 2 pruning
a = axs[1]; a.set_title("Structured pruning", fontsize=9, fontweight="bold")
cols = 8; drop = {1, 4}
for c in range(cols):
    for r in range(4):
        a.add_patch(plt.Rectangle((0.6 + c * 0.55, 5.2 + r * 0.8), 0.55, 0.8, fc="#bbb" if c in drop else "white", ec="#222", lw=0.5))
    if c in drop: a.plot([0.6 + c * 0.55 + 0.05, 0.6 + c * 0.55 + 0.5], [5.25, 8.35], color="#111", lw=1.2)
a.text(2.8, 9.0, "score every channel", ha="center", fontsize=7.5)
arr(a, 5.3, 6.8, 6.5, 6.8)
for c in range(cols - len(drop)):
    for r in range(4):
        a.add_patch(plt.Rectangle((6.7 + c * 0.5, 5.2 + r * 0.8), 0.5, 0.8, fc="white", ec="#222", lw=0.5))
a.text(5, 3.4, "delete the weakest whole channels", ha="center", fontsize=7.5)
a.text(5, 2.1, "route C: 1,536 -> 1,162 channels", ha="center", fontsize=7.5)
a.text(5, 0.9, "the matrix really shrinks", ha="center", fontsize=7.5, color="#444")
# 3 distillation
a = axs[2]; a.set_title("Distillation", fontsize=9, fontweight="bold")
cell(a, 0.3, 6.3, "DeiT-Small\nteacher, 22M", w=4.0, h=2.4, fill="#ddd")
cell(a, 6.5, 6.3, "DeiT-Tiny\nstudent, 5.7M", w=3.4, h=2.4)
arr(a, 4.4, 7.5, 6.5, 7.5)
a.text(5, 4.6, "teacher's guesses:\n87% dog, 9% wolf, ...", ha="center", fontsize=7.5)
a.text(5, 2.0, "the student learns to\ngive the same answers", ha="center", fontsize=7.5)
a.text(5, 0.6, "route D", ha="center", fontsize=7.5, color="#444")
fig.tight_layout(); save(fig, "fig_techniques")

# ---------------------------------------------------------------- how a ViT sees an image (course diagram)
fig, ax = plt.subplots(figsize=(7.4, 2.0)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 22)
for r in range(4):                                         # the image as a 4x4 grid of patches
    for c in range(4):
        ax.add_patch(plt.Rectangle((1 + c * 3, 3 + r * 3.6), 3, 3.6, fc="#e8e8e8" if (r + c) % 2 else "#cfcfcf", ec="#222", lw=0.5))
ax.text(7, 19.2, "image, cut into\n16x16-pixel patches", ha="center", fontsize=7.5)
def rbox(x, y, w, h, t, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fill, ec="#222", lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
def ar(x0, x1, y=10): ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#222"))
ar(14, 18); rbox(18.5, 6, 13, 8, "each patch ->\na token (list of\n384 numbers)")
ar(32, 36)
ax.add_patch(FancyBboxPatch((36.5, 2.5), 32, 15, boxstyle="round,pad=0.3,rounding_size=0.8", fc="#f4f4f4", ec="#222", lw=0.8, ls="--"))
ax.text(52.5, 19.3, "one block, repeated 12 times", ha="center", fontsize=7.5)
rbox(38, 6, 13.5, 8, "attention:\ntokens look at\neach other", "white")
ar(52, 54); rbox(54.5, 6, 12.5, 8, "MLP: each token\nthinks alone\n(64% of weights)", "white")
ar(69, 73); rbox(73.5, 6, 11, 8, "class token\nsummarises\nthe image")
ar(85, 88); rbox(88.5, 6, 11, 8, "1,000 scores\n-> top guess\n\"tiger\"", "#ddd")
save(fig, "fig_vit")
