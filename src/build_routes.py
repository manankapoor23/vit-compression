#!/usr/bin/env python3
"""
Build ChotaDeiT-A / B / C / D  —  four routes to the SAME serialized size.

The whole point: every route is tuned until its measured file size matches the
target, so any difference in behaviour later cannot be explained by one model
simply being bigger. Sizes are MEASURED with os.path.getsize(), never computed.

  A  uniform INT3, group 256                  (natural size, sets the target)
  B  ternary blocks + base-3 packing          (undershoots; spends surplus on precision)
  C  INT4 fully quantized + structured prune  (overshoots; prunes down)
  D  distilled DeiT-Tiny + mixed INT8/FP16    (undershoots; spends surplus on precision)

Each route has ONE tunable knob, binary-searched to hit the target.

Usage:
    python build_routes.py --routes A B C D
    python build_routes.py --target-mb 9.77 --tol 0.01
"""

import argparse, json, math, os, sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1_audit import (MB, component_of, is_quantizable, pack_codes,
                          quantize_groupwise, base3_bytes)

OUT = "models"


# ---------------------------------------------------------------------------
# ternary quantization  (TWN-style threshold)  -- NOT the same as INT2
# ---------------------------------------------------------------------------

def quantize_ternary(w: torch.Tensor):
    """
    Weights -> {-1, 0, +1} with one FP16 scale per output channel.

    Threshold delta = 0.7 * mean(|w|) is the Ternary Weight Networks rule.
    Returns (codes in {0,1,2}, per-row scales, max_abs_err, zero_fraction).
    """
    x = w.detach().float().reshape(w.shape[0], -1)
    delta = 0.7 * x.abs().mean(dim=1, keepdim=True)
    tern = torch.zeros_like(x)
    tern[x > delta] = 1.0
    tern[x < -delta] = -1.0
    nz = tern.abs().sum(dim=1, keepdim=True).clamp(min=1)
    alpha = ((x.abs() * tern.abs()).sum(dim=1, keepdim=True) / nz).half().float()
    err = (tern * alpha - x).abs().max().item()
    zero_frac = (tern == 0).float().mean().item()
    return (tern + 1).to(torch.uint8).reshape(-1).numpy(), alpha.reshape(-1), err, zero_frac


def pack_base3(codes: np.ndarray) -> bytes:
    """5 ternary codes per byte: 3**5 = 243 <= 255.  Exactly 1.600 bits/weight."""
    c = codes.astype(np.uint16)
    pad = (-len(c)) % 5
    if pad:
        c = np.concatenate([c, np.zeros(pad, dtype=np.uint16)])
    g = c.reshape(-1, 5)
    packed = g[:, 0] + 3*g[:, 1] + 9*g[:, 2] + 27*g[:, 3] + 81*g[:, 4]
    return packed.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# structured pruning: drop MLP hidden channels
# ---------------------------------------------------------------------------

def prune_mlp_channels(state: OrderedDict, frac: float) -> OrderedDict:
    """
    Remove the lowest-importance `frac` of hidden channels from every MLP.

    Importance = L2 norm of the channel's incoming (fc1) and outgoing (fc2)
    weights. This genuinely shrinks the matrices -- unlike unstructured
    pruning, which needs a sparse format before the file gets smaller.
    """
    if frac <= 0:
        return state
    out = OrderedDict(state)
    blocks = sorted({k.split(".")[1] for k in state if k.startswith("blocks.")}, key=int)
    for b in blocks:
        fc1w, fc1b = f"blocks.{b}.mlp.fc1.weight", f"blocks.{b}.mlp.fc1.bias"
        fc2w = f"blocks.{b}.mlp.fc2.weight"
        if fc1w not in out:
            continue
        imp = out[fc1w].norm(dim=1) + out[fc2w].norm(dim=0)
        keep = torch.topk(imp, max(1, int(round(imp.numel() * (1 - frac))))).indices.sort().values
        out[fc1w] = out[fc1w][keep].contiguous()
        out[fc1b] = out[fc1b][keep].contiguous()
        out[fc2w] = out[fc2w][:, keep].contiguous()
    return out


# ---------------------------------------------------------------------------
# serialization: one raw-binary writer shared by every route
# ---------------------------------------------------------------------------

def write_model(state, path, policy):
    """
    policy(name, tensor) -> ("int", bits, group) | ("ternary", None, None) | ("fp16", ...)

    Raw container: [4-byte header len][JSON header][payload]. getsize() is the payload.
    """
    header, payload = [], bytearray()
    stats = {"codes": 0, "scales": 0, "fp16": 0, "zero_fracs": []}
    worst = 0.0

    for name, w in state.items():
        kind, bits, group = policy(name, w)
        if kind == "int":
            codes, scales, err = quantize_groupwise(w, bits, group)
            cb, sb = pack_codes(codes, bits), scales.numpy().astype(np.float16).tobytes()
            payload += cb; payload += sb
            stats["codes"] += len(cb); stats["scales"] += len(sb); worst = max(worst, err)
            header.append([name, f"i{bits}", list(w.shape), len(cb), len(sb)])
        elif kind == "ternary":
            codes, scales, err, zf = quantize_ternary(w)
            cb, sb = pack_base3(codes), scales.numpy().astype(np.float16).tobytes()
            payload += cb; payload += sb
            stats["codes"] += len(cb); stats["scales"] += len(sb)
            stats["zero_fracs"].append(zf); worst = max(worst, err)
            header.append([name, "t3", list(w.shape), len(cb), len(sb)])
        else:
            fb = w.half().numpy().tobytes()
            payload += fb; stats["fp16"] += len(fb)
            header.append([name, "f16", list(w.shape), len(fb), 0])

    hdr = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(len(hdr).to_bytes(4, "little")); f.write(hdr); f.write(payload)
    return {"bytes": os.path.getsize(path), "stats": stats, "max_err": worst}


# ---------------------------------------------------------------------------
# route policies -- each has ONE knob, binary-searched to hit the target
# ---------------------------------------------------------------------------

def sensitivity_order(state):
    """Layers ranked most-sensitive-first. Proxy: weight-variance per parameter.

    High-variance layers lose the most under coarse quantization, so they are
    the ones worth spending surplus budget on. Cheap proxy -- swap for a
    Hessian/Fisher measure later if the results turn on it.
    """
    cand = [(n, w) for n, w in state.items() if is_quantizable(n, w)]
    return [n for n, _ in sorted(cand, key=lambda kv: -kv[1].float().var().item())]


def make_policy(route, state, knob, order):
    """knob meaning: A=group size, B/D=#layers promoted to higher precision, C unused."""
    protect = set(order[:int(knob)]) if route in ("B", "D") else set()

    def policy(name, w):
        comp = component_of(name)
        if not is_quantizable(name, w):
            return ("fp16", None, None)
        if route == "A":
            return ("int", 3, int(knob))
        if route == "B":
            return ("int", 8, 128) if name in protect else ("ternary", None, None)
        if route == "C":
            return ("int", 4, 128)                      # everything, incl. patch_embed + head
        if route == "D":
            return ("fp16", None, None) if name in protect else ("int", 8, 128)
        raise ValueError(route)
    return policy


def build_route(route, base_state, target_bytes, tol, outdir):
    """Binary-search the knob until measured size is within tol of the target."""
    path = os.path.join(outdir, f"ChotaDeiT-{route}.bin")
    order = sensitivity_order(base_state)

    if route == "C":
        lo, hi = 0.0, 0.60                              # prune fraction
    elif route == "A":
        lo, hi = 32, 1024                               # group size (bigger = smaller file)
    else:
        lo, hi = 0, len(order)                          # #layers promoted

    best = None
    for it in range(18):
        mid = (lo + hi) / 2
        if route == "C":
            st = prune_mlp_channels(base_state, mid)
            res = write_model(st, path, make_policy(route, st, 0, order))
        else:
            k = int(round(mid))
            res = write_model(base_state, path, make_policy(route, base_state, k, order))
        err = res["bytes"] / target_bytes - 1
        best = (mid, res)
        if abs(err) <= tol:
            break
        # bigger knob -> smaller file for A and C; bigger -> larger file for B and D
        if route in ("A", "C"):
            lo, hi = (mid, hi) if err > 0 else (lo, mid)
        else:
            lo, hi = (lo, mid) if err > 0 else (mid, hi)

    knob, res = best
    # Store the EXACT integer the policy will use. Storing the raw float let
    # build (int(round(x))) and train (int(x)) disagree by one layer, which
    # silently drifted every export by 1-2%.
    knob = knob if route == "C" else int(round(knob))
    return {"route": route, "path": path, "knob": knob,
            "bytes": res["bytes"], "mb": res["bytes"] / MB,
            "delta_pct": (res["bytes"] / target_bytes - 1) * 100,
            "max_abs_err": res["max_err"],
            "mean_zero_frac": (float(np.mean(res["stats"]["zero_fracs"]))
                               if res["stats"]["zero_fracs"] else None),
            "codes_mb": res["stats"]["codes"] / MB,
            "scales_mb": res["stats"]["scales"] / MB,
            "fp16_mb": res["stats"]["fp16"] / MB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--student", default="deit_tiny_patch16_224", help="route D backbone")
    ap.add_argument("--routes", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--target-mb", type=float, default=9.77)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()

    import timm
    os.makedirs(args.outdir, exist_ok=True)
    target = args.target_mb * MB

    def load(n):
        m = timm.create_model(n, pretrained=True).eval()
        return OrderedDict((k, v.detach().cpu()) for k, v in m.state_dict().items())

    teacher = load(args.model)
    student = load(args.student)
    print(f"target {args.target_mb:.2f} MB  (+/- {args.tol*100:.0f}%)\n")

    results = []
    for r in args.routes:
        base = student if r == "D" else teacher
        res = build_route(r, base, target, args.tol, args.outdir)
        results.append(res)
        knob = {"A": "group size", "B": "#layers at INT8",
                "C": "prune fraction", "D": "#layers at FP16"}[r]
        print(f"ChotaDeiT-{r}: {res['mb']:6.3f} MB  ({res['delta_pct']:+5.2f}%)   "
              f"{knob} = {res['knob']}")
        print(f"              codes {res['codes_mb']:.2f} | scales {res['scales_mb']:.2f} "
              f"| fp16 {res['fp16_mb']:.2f} MB | max err {res['max_abs_err']:.4f}"
              + (f" | zeros {res['mean_zero_frac']:.1%}" if res['mean_zero_frac'] else ""))

    with open(os.path.join(args.outdir, "routes.json"), "w") as f:
        json.dump({"target_mb": args.target_mb, "routes": results}, f, indent=2)
    print(f"\nWrote {args.outdir}/routes.json")
    print("NOTE: sizes only. Accuracy needs recovery training -- next script.")


if __name__ == "__main__":
    main()
