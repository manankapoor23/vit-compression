#!/usr/bin/env python3
"""Rebuild Mix-QViT's printed 11.4 MB for 4-bit DeiT-Small in their own layout (report Sec. 3).

Mix-QViT quantises per output channel, with patch embedding and head at 8 bits. A real file must also
store one scale per channel and keep biases and norms at higher precision. Two variants bracket it:
  symmetric:  one FP16 scale per channel, other tensors FP16
  asymmetric: FP16 scale + FP16 zero point per channel, other tensors FP32
Shapes only (no weights are downloaded).
"""
import timm

PRINTED_MB = 11.4
sd = timm.create_model("deit_small_patch16_224", pretrained=False).state_dict()


def size_mb(per_channel_bytes, other_bytes):
    total = 0.0
    for name, t in sd.items():
        if t.dim() >= 2 and name.endswith("weight"):
            bits = 8 if ("patch_embed" in name or name.startswith("head")) else 4
            total += t.numel() * bits / 8 + t.shape[0] * per_channel_bytes
        else:
            total += t.numel() * other_bytes
    return total / 1e6


for label, scale, other in [("symmetric, FP16 scales and leftovers", 2, 2),
                            ("asymmetric, FP16 scale+zero, FP32 leftovers", 4, 4)]:
    mb = size_mb(scale, other)
    print(f"{label:45s} {mb:.3f} MB  = {100 * (mb / PRINTED_MB - 1):+.1f}% over the printed {PRINTED_MB} MB")
