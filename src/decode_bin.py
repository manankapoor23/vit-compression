#!/usr/bin/env python3
"""
Decode a ChotaDeiT .bin file back into a runnable PyTorch model.

Closes the gap "no .bin file was ever loaded back". The file format (written by
build_routes.write_model) is:

    [4-byte little-endian header length][JSON header][payload]
    header = list of [tensor name, kind, shape, code bytes, scale bytes]
    kind   = "i<bits>" (group-wise symmetric int), "t3" (ternary, base-3 packed), "f16"

The group size is not stored in the header, so it is passed in (known per route:
A = 36, sweep models = their g, every INT8/INT4 layer in B/C/D = 128).

Self-test (no GPU, no images):  python src/decode_bin.py --roundtrip evidence/models/*.bin
  decode -> re-encode with the SAME writer -> must be byte-identical to the original file.
"""
import argparse, json, math, os, re, sys, tempfile
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_routes import write_model

ROUTE_GROUP = {"A": 36, "B": 128, "C": 128, "D": 128}


def group_for(path):
    """Group size used for int tensors in this file, from its name."""
    name = os.path.basename(path)
    m = re.search(r"-g(\d+)", name)
    if m:
        return int(m.group(1))
    return ROUTE_GROUP[re.search(r"ChotaDeiT-([ABCD])", name).group(1)]


def read_bin(path):
    raw = open(path, "rb").read()
    hl = int.from_bytes(raw[:4], "little")
    return json.loads(raw[4:4 + hl]), memoryview(raw)[4 + hl:]


def _unpack_int(buf, n, bits):
    bitarr = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))[: n * bits].reshape(n, bits)
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.int64)
    return bitarr.astype(np.int64) @ weights            # codes in [0, 2**bits)


def _unpack_base3(buf, n):
    v = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    digits = np.stack([(v // 3 ** k) % 3 for k in range(5)], axis=1).reshape(-1)
    return digits[:n]                                    # codes in {0,1,2}


def decode(path, group=None):
    """Return (state_dict of float32 tensors, header)."""
    group = group or group_for(path)
    header, payload = read_bin(path)
    state, off = OrderedDict(), 0
    for name, kind, shape, cb, sb in header:
        n = int(np.prod(shape)) if shape else 1
        code_buf = payload[off:off + cb]; off += cb
        scale_buf = payload[off:off + sb]; off += sb
        if kind == "f16":
            t = torch.from_numpy(np.frombuffer(code_buf, dtype=np.float16).astype(np.float32).copy())
        elif kind == "t3":
            codes = _unpack_base3(code_buf, n) - 1                       # {-1, 0, +1}
            alpha = np.frombuffer(scale_buf, dtype=np.float16).astype(np.float32)
            t = torch.from_numpy((codes.reshape(shape[0], -1) * alpha[:, None]).astype(np.float32))
        elif kind.startswith("i"):
            bits = int(kind[1:]); qmax = 2 ** (bits - 1) - 1
            g = min(group, n)
            q = _unpack_int(code_buf, n, bits) - (qmax + 1)
            scales = np.frombuffer(scale_buf, dtype=np.float16).astype(np.float32)
            assert len(scales) == math.ceil(n / g), f"{name}: {len(scales)} scales, group {g}"
            q = np.concatenate([q, np.zeros(len(scales) * g - n, dtype=np.int64)])
            t = torch.from_numpy((q.reshape(-1, g) * scales[:, None]).reshape(-1)[:n].astype(np.float32))
        else:
            raise ValueError(f"{name}: unknown kind {kind}")
        state[name] = t.reshape(shape)
    assert off == len(payload), f"{path}: {len(payload) - off} payload bytes unread"
    return state, header


def header_policy(header, group):
    """Re-encode each tensor exactly as the header says it was encoded."""
    kinds = {name: kind for name, kind, *_ in header}

    def policy(name, w):
        k = kinds[name]
        if k == "f16": return ("fp16", None, None)
        if k == "t3":  return ("ternary", None, None)
        return ("int", int(k[1:]), group)
    return policy


def build_model(path, device="cpu"):
    """A timm model with the decoded weights loaded (pruned MLPs resized from the header)."""
    import timm, torch.nn as nn
    state, _ = decode(path)
    arch = "deit_tiny_patch16_224" if "ChotaDeiT-D" in os.path.basename(path) else "deit_small_patch16_224"
    m = timm.create_model(arch, pretrained=False)
    for i, blk in enumerate(m.blocks):
        h = state[f"blocks.{i}.mlp.fc1.weight"].shape[0]
        if h != blk.mlp.fc1.out_features:                 # route C: pruned hidden width
            d = blk.mlp.fc1.in_features
            blk.mlp.fc1, blk.mlp.fc2 = nn.Linear(d, h), nn.Linear(h, d)
    m.load_state_dict(state, strict=True)
    return m.to(device).eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrip", nargs="+", required=True, help=".bin files to self-test")
    a = ap.parse_args()
    bad = 0
    for p in a.roundtrip:
        g = group_for(p)
        state, header = decode(p, g)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "re.bin")
            write_model(state, out, header_policy(header, g))
            same = open(out, "rb").read() == open(p, "rb").read()
        bad += not same
        print(f"  {os.path.basename(p):<28} group {g:<4} {len(header):>4} tensors  "
              f"{os.path.getsize(p):>10,} B  round-trip {'BYTE-IDENTICAL' if same else 'DIFFERS'}")
    print(f"\n{len(a.roundtrip) - bad}/{len(a.roundtrip)} files decode and re-encode byte-identically")
    sys.exit(1 if bad else 0)
