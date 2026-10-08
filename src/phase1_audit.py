#!/usr/bin/env python3
"""
Phase 1 audit — where do DeiT-Small's bytes actually live?

Answers three questions that the ViT compression literature reports incorrectly
or not at all:

  1. Which components hold the parameters? (MLP vs attention vs the "small" bits)
  2. What does a quantized model ACTUALLY weigh on disk, including the per-group
     scale metadata that everyone omits from `params x bits / 8`?
  3. If you follow standard tooling and leave patch-embed / pos-embed / head at
     FP16, how much of your 10 MB budget do those leftovers eat?

No GPU. No training. No ImageNet. Accuracy evaluation is Phase 2.

Usage:
    python src/phase1_audit.py                          # DeiT-Small, default sweep
    python src/phase1_audit.py --model deit_tiny_patch16_224
    python src/phase1_audit.py --groups 32 64 128 256
    python src/phase1_audit.py --outdir ./audit_out --json audit.json
"""

import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
import torch

MB = 1_000_000  # decimal MB, matching how model cards quote sizes


# ----------------------------------------------------------------------------
# component attribution
# ----------------------------------------------------------------------------

def component_of(name: str) -> str:
    """Bucket a parameter name into an architectural component."""
    if name.startswith("patch_embed"):
        return "patch_embed"
    if name.startswith("pos_embed") or name.startswith("cls_token"):
        return "pos_embed+cls"
    if name.startswith("head"):
        return "head"
    if name.startswith("blocks"):
        if ".attn." in name:
            return "blocks.attn"
        if ".mlp." in name:
            return "blocks.mlp"
        if ".norm" in name:
            return "blocks.norm"
        return "blocks.other"
    if name.startswith("norm"):
        return "final_norm"
    return "other"


def is_quantizable(name: str, tensor: torch.Tensor) -> bool:
    """Standard practice: quantize 2D+ weight matrices, leave norms/biases alone."""
    return tensor.dim() >= 2 and name.endswith("weight")


# ----------------------------------------------------------------------------
# exact bit packing
# ----------------------------------------------------------------------------

def packed_bytes(n_values: int, bits: int) -> int:
    """Exact bytes to store n_values codes at `bits` each, bit-packed."""
    return math.ceil(n_values * bits / 8)


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    """Bit-pack integer codes (0 <= code < 2**bits) into a dense byte string."""
    assert 1 <= bits <= 8
    c = codes.astype(np.uint8).reshape(-1, 1)
    bitmat = np.unpackbits(c, axis=1)[:, 8 - bits:]      # low `bits` bits of each
    flat = bitmat.reshape(-1)
    pad = (-len(flat)) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(flat).tobytes()


def base3_bytes(n_values: int) -> int:
    """Ternary base-3 packing: 5 values per byte, since 3**5 = 243 < 256."""
    return math.ceil(n_values / 5)


# ----------------------------------------------------------------------------
# group-wise symmetric quantization
# ----------------------------------------------------------------------------

def quantize_groupwise(w: torch.Tensor, bits: int, group_size: int):
    """
    Symmetric group-wise quantization, the AWQ/GPTQ-style layout.

    Returns (codes, scales, max_abs_err). Codes are unsigned ints in
    [0, 2**bits - 1]; one FP16 scale per group, written to the file --
    a quantized model without its scales is not a model.
    """
    flat = w.detach().float().reshape(-1)
    n = flat.numel()
    g = min(group_size, n)
    pad = (-n) % g
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    groups = flat.reshape(-1, g)

    qmax = 2 ** (bits - 1) - 1                 # symmetric signed range
    scale = groups.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    scale = scale.half().float()               # scales are stored as FP16

    q = torch.clamp(torch.round(groups / scale), -qmax - 1, qmax)
    err = (q * scale - groups).abs().max().item()

    codes = (q + (qmax + 1)).to(torch.uint8).reshape(-1)[:n].numpy()
    return codes, scale.reshape(-1), err        # scales ARE stored now


# ----------------------------------------------------------------------------
# serialization scenarios
# ----------------------------------------------------------------------------

def serialize_float(state, dtype, path):
    """Save every tensor at a single float dtype and measure the real file."""
    out = {k: v.to(dtype) for k, v in state.items()}
    torch.save(out, path)
    return os.path.getsize(path)


def serialize_quantized(state, bits, group_size, path, quantize_all=False):
    """
    Quantize weight matrices to `bits`; everything else stays FP16.

    RAW binary container: 4-byte header length, JSON header, then payload.
    No pickle, no zip. os.path.getsize() is therefore exactly the deployable
    payload. The earlier torch.save version inflated every quantized file by
    4-10 MB of framing -- precisely the class of error this audit exists to find.

    quantize_all=False -> standard tooling: patch_embed and head stay FP16
    quantize_all=True  -> FTerViT-style fully quantized (norms/biases still FP16)
    """
    header, payload = [], bytearray()
    per_component = OrderedDict()
    totals = {"codes": 0, "scales": 0, "fp16_leftover": 0}
    worst_err = 0.0

    for name, w in state.items():
        comp = component_of(name)
        rec = per_component.setdefault(comp, {"params": 0, "bytes": 0, "quantized": 0})
        rec["params"] += w.numel()

        quantize_this = is_quantizable(name, w)
        if not quantize_all and comp in ("patch_embed", "head"):
            quantize_this = False

        if quantize_this:
            codes, scales, err = quantize_groupwise(w, bits, group_size)
            worst_err = max(worst_err, err)
            cbuf = pack_codes(codes, bits)
            sbuf = scales.numpy().astype(np.float16).tobytes()
            payload += cbuf; payload += sbuf
            totals["codes"] += len(cbuf); totals["scales"] += len(sbuf)
            rec["bytes"] += len(cbuf) + len(sbuf); rec["quantized"] += w.numel()
            header.append([name, "q", list(w.shape), len(cbuf), len(sbuf)])
        else:
            fbuf = w.half().numpy().tobytes()
            payload += fbuf
            totals["fp16_leftover"] += len(fbuf)
            rec["bytes"] += len(fbuf)
            header.append([name, "f", list(w.shape), len(fbuf), 0])

    hdr = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(len(hdr).to_bytes(4, "little")); f.write(hdr); f.write(payload)

    exact = sum(totals.values())
    size = os.path.getsize(path)
    return {"file_bytes": size, "exact_bytes": exact, "header_bytes": size - exact,
            "breakdown": totals, "per_component": per_component,
            "max_abs_weight_err": worst_err}


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------

def bar(frac, width=32):
    return "#" * max(1, round(frac * width)) if frac > 0 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--budget-mb", type=float, default=10.0)
    ap.add_argument("--groups", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2])
    ap.add_argument("--outdir", default="audit_out")
    ap.add_argument("--json", default=None)
    ap.add_argument("--random-init", action="store_true",
                    help="skip the weight download. File sizes depend only on tensor shapes; "
                         "verified byte-identical to pretrained on DeiT-S and DeiT-Tiny "
                         "(2026-09-25). Do NOT use for max_abs_weight_err.")
    args = ap.parse_args()

    try:
        import timm
    except ImportError:
        raise SystemExit("Needs timm:  pip install timm torch numpy")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading {args.model} ...")
    model = timm.create_model(args.model, pretrained=not args.random_init)
    model.eval()
    state = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())

    total_params = sum(t.numel() for t in state.values())
    budget_bytes = args.budget_mb * MB
    report = {"model": args.model, "total_params": total_params,
              "budget_mb": args.budget_mb, "sections": {}}

    # ---- 1. parameter breakdown -------------------------------------------
    print("\n" + "=" * 78)
    print(f"1. WHERE THE PARAMETERS ARE  —  {args.model}")
    print("=" * 78)
    comp = OrderedDict()
    for name, t in state.items():
        comp[component_of(name)] = comp.get(component_of(name), 0) + t.numel()
    comp = OrderedDict(sorted(comp.items(), key=lambda kv: -kv[1]))

    print(f"{'component':<18}{'params':>12}{'share':>9}{'FP32':>10}   distribution")
    for k, v in comp.items():
        frac = v / total_params
        print(f"{k:<18}{v:>12,}{frac:>8.1%}{v*4/MB:>9.2f}MB   {bar(frac)}")
    print(f"{'TOTAL':<18}{total_params:>12,}{1.0:>8.1%}{total_params*4/MB:>9.2f}MB")
    report["sections"]["params"] = comp

    # the number the whole project rests on
    avg_bits = budget_bytes * 8 / total_params
    print(f"\n  >> BUDGET: {args.budget_mb:g} MB / {total_params:,} params "
          f"= {avg_bits:.2f} bits per parameter")

    # ---- 2. real file sizes ------------------------------------------------
    print("\n" + "=" * 78)
    print("2. WHAT IT ACTUALLY WEIGHS ON DISK  (measured, not params x bits / 8)")
    print("=" * 78)
    rows = []
    for label, dt in [("FP32", torch.float32), ("FP16", torch.float16)]:
        p = os.path.join(args.outdir, f"{label.lower()}.pt")
        b = serialize_float(state, dt, p)
        rows.append((label, "-", b, b * 8 / total_params, None))

    for bits in args.bits:
        for g in args.groups:
            p = os.path.join(args.outdir, f"int{bits}_g{g}.bin")
            r = serialize_quantized(state, bits, g, p)
            rows.append((f"INT{bits}", f"g={g}", r["file_bytes"],
                         r["file_bytes"] * 8 / total_params, r))

    print(f"{'format':<8}{'group':>7}{'nominal':>11}{'ACTUAL':>10}"
          f"{'bits/param':>12}{'overhead':>10}   fits {args.budget_mb:g}MB?")
    print("  (nominal = params x bits / 8 -- the number the literature prints)")
    for label, grp, size, bpp, r in rows:
        nominal = (total_params * int(label[3:]) / 8 / MB) if label.startswith("INT") \
                  else (total_params * (32 if label == "FP32" else 16) / 8 / MB)
        over = (size / MB) / nominal - 1
        fits = "YES" if size <= budget_bytes else "no"
        print(f"{label:<8}{grp:>7}{nominal:>10.2f}M{size/MB:>9.2f}M"
              f"{bpp:>12.2f}{over:>9.1%}   {fits}")
    report["sections"]["sizes"] = [
        {"format": l, "group": g, "bytes": s, "bits_per_param": b} for l, g, s, b, _ in rows
    ]

    # ---- 3. the leftovers --------------------------------------------------
    print("\n" + "=" * 78)
    print("3. THE LEFTOVERS  —  what standard tooling refuses to quantize")
    print("=" * 78)
    g = args.groups[len(args.groups) // 2]
    for bits in args.bits:
        std = serialize_quantized(state, bits, g, os.path.join(args.outdir, f"std_{bits}.bin"))
        allq = serialize_quantized(state, bits, g, os.path.join(args.outdir, f"all_{bits}.bin"),
                                   quantize_all=True)
        lo = std["breakdown"]["fp16_leftover"]
        print(f"\n  INT{bits} (group {g}):")
        print(f"    standard (patch_embed + head left FP16) : {std['file_bytes']/MB:6.2f} MB")
        print(f"      of which un-quantized FP16 leftovers  : {lo/MB:6.2f} MB "
              f"({lo/std['file_bytes']:.1%} of the file)")
        print(f"    fully quantized                         : {allq['file_bytes']/MB:6.2f} MB "
              f"({1 - allq['file_bytes']/std['file_bytes']:+.1%})")
        print(f"    scale metadata cost                     : "
              f"{std['breakdown']['scales']/MB:6.2f} MB "
              f"({std['breakdown']['scales']*8/total_params:.3f} bits/param)")
    report["sections"]["leftovers_group"] = g

    # ---- 4. ternary projection --------------------------------------------
    print("\n" + "=" * 78)
    print("4. TERNARY PACKING PROJECTION  (the methods-line win)")
    print("=" * 78)
    n = total_params
    for label, bpw, sz in [
        ("naive 2-bit packing (FTerViT ships ~2.11)", 2.0, n * 2 / 8),
        ("base-3 packing, 5 weights / byte", 8 / 5, base3_bytes(n)),
        ("Shannon floor, log2(3), uniform", math.log2(3), n * math.log2(3) / 8),
        ("entropy @ 37.3% zeros (FTerViT reported)", 1.580, n * 1.580 / 8),
    ]:
        print(f"  {label:<45}{bpw:5.3f} bits  ->{sz/MB:6.2f} MB")
    print("\n  TWO DISTINCT GAPS — keep them separate:")
    print(f"    packing  2.110 -> 1.600 = {2.110-1.600:.3f} bits/param  <- LARGE. your methods line.")
    print(f"    entropy  1.600 -> 1.580 = {1.600-1.580:.3f} bits/param  <- negligible. the dead end.")
    print("  base-3 packing achieves 1.600 (=8/5), NOT the 1.585 log2(3) floor.")
    print("  Caveat: 2.110 includes FTerViT's scale overhead; packing touches only the")
    print("  code bits, so the realistic win is ~19-24%. Decompose before quoting it.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nWrote {args.json}")
    print(f"Serialized models in {args.outdir}/  —  verify with: ls -l {args.outdir}")


if __name__ == "__main__":
    main()
