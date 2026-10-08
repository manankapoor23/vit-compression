#!/usr/bin/env python3
"""
Multi-architecture summary of the Phase 1 size audit (Contribution A breadth).

Two independent derivations of every quantized file size (CLAUDE.md rule 1.2):
  1. MEASURED  -- phase1_audit.py wrote the file and read os.path.getsize()   (the JSON)
  2. PREDICTED -- rebuilt here from tensor shapes alone:
       quantized tensor : ceil(n * bits / 8) code bytes + ceil(n / g) FP16 scales x 2 bytes
       everything else  : n x 2 bytes (FP16)
     The gap between the two is the file header (JSON layer list) -- reported, not hidden.

Why random-init weights are legitimate here: sizes depend only on shapes. Verified
2026-09-25 -- random-init and pretrained audits are byte-identical on all 16 quantized files
for DeiT-S and DeiT-Tiny. The FP32/FP16 torch.save files differ by a constant 391 bytes
(container framing of a newer torch), so float rows are not used for any claim.

Usage:  python src/arch_audit_summary.py evidence/audit_multiarch/*.json
"""
import json, math, sys
import timm

MB = 1_000_000
BITS, GROUPS = (8, 4, 3, 2), (32, 64, 128, 256)

def predicted_bytes(state, bits, g):
    tot = 0
    for name, t in state.items():
        n = t.numel()
        q = t.dim() >= 2 and name.endswith("weight") and not (
            name.startswith("patch_embed") or name.startswith("head"))
        if q:
            gg = min(g, n)
            tot += math.ceil(n * bits / 8) + math.ceil(n / gg) * 2
        else:
            tot += n * 2
    return tot

rows = []
for path in sys.argv[1:]:
    J = json.load(open(path))
    m = timm.create_model(J["model"], pretrained=False)
    state = m.state_dict()
    P = J["total_params"]
    assert P == sum(t.numel() for t in state.values()), f"param count mismatch in {path}"
    meas = {(r["format"], r["group"]): r["bytes"] for r in J["sections"]["sizes"]}
    left = sum(t.numel() for k, t in state.items() if not (
        t.dim() >= 2 and k.endswith("weight") and not (k.startswith("patch_embed") or k.startswith("head"))))
    width = getattr(m, "embed_dim", None) or getattr(m, "num_features", None)
    worst_hdr = 0
    row = dict(model=J["model"], params=P, width=width, leftover_share=left / P, over={})
    for b in BITS:
        for g in GROUPS:
            mb = meas[(f"INT{b}", f"g={g}")]
            pb = predicted_bytes(state, b, g)
            worst_hdr = max(worst_hdr, mb - pb)
            assert 0 <= mb - pb < 200_000, f"{J['model']} INT{b} g{g}: measured {mb} vs predicted {pb}"
            row["over"][(b, g)] = mb / (P * b / 8) - 1
    row["worst_header_bytes"] = worst_hdr
    rows.append(row)

rows.sort(key=lambda r: r["params"])
print("MEASURED file size vs the nominal formula params x bits / 8   (standard policy:")
print("patch_embed + head + norms + biases FP16; symmetric group-wise, FP16 scales)\n")
print(f"{'model':<30}{'params':>12}{'width':>7}{'FP16 share':>11}"
      + "".join(f"{f'INT{b} g256':>11}" for b in BITS) + f"{'INT4 g32':>10}{'max hdr':>9}")
for r in rows:
    print(f"{r['model']:<30}{r['params']:>12,}{str(r['width']):>7}{r['leftover_share']:>10.1%}"
          + "".join(f"{r['over'][(b, 256)]:>+10.1%} " for b in BITS)
          + f"{r['over'][(4, 32)]:>+9.1%}{r['worst_header_bytes']:>8,}B")
print("\nEvery measured file = predicted-from-shapes + a JSON header (last column, worst case);")
print("the assert above fails the script if any file deviates by more than that.")

print("\nMonotonicity checks (the two axes of Contribution A):")
for r in rows:
    seq = [r["over"][(b, 256)] for b in BITS]
    print(f"  {r['model']:<30} overhead rises as bits fall (8>4>3>2): "
          f"{all(x < y for x, y in zip(seq, seq[1:]))}")
vit_family = [r for r in rows if r["model"].startswith(("deit", "vit"))]
seq = [r["over"][(2, 256)] for r in vit_family]
print(f"  plain-ViT family, INT2 g256 overhead shrinks as width grows "
      f"(width {' -> '.join(str(r['width']) for r in vit_family)}: "
      f"{' -> '.join(f'{x:+.1%}' for x in seq)}): "
      f"{all(x > y for x, y in zip(seq, seq[1:]))}")
