#!/usr/bin/env python3
"""
CPU latency for every compression route — the third axis of the P49 Pareto table.

Latency depends on architecture and arithmetic, NOT on weight values, so models are built with
random weights (no download, no ImageNet). What each row measures:

  fp32           eager PyTorch, weights in float32
  int8-dynamic   torch.ao dynamic quantization of every nn.Linear (int8 weights, int8 matmul)

Our own 3-bit and ternary formats have no CPU kernel: at load time they are decoded back to
float, so on a CPU they run at the speed of the architecture they decode into. That is the honest
point of this table — fewer bytes on disk is not fewer milliseconds.

Protocol: batch 1, 3x224x224, fixed thread count, 10 warm-up runs, 50 timed runs, median and
inter-quartile range reported.

Usage:  python src/cpu_latency.py [--threads 1 4] [--runs 50]
"""
import argparse, json, os, platform, statistics, time
import torch, torch.nn as nn, timm

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
ap.add_argument("--runs", type=int, default=50)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--out", default="evidence/latency/cpu_latency.json")
a = ap.parse_args()

def pruned_deit_s(hidden=1162):
    """Route C's shapes: DeiT-S with every MLP hidden layer cut from 1536 to 1162."""
    m = timm.create_model("deit_small_patch16_224", pretrained=False)
    for blk in m.blocks:
        d = blk.mlp.fc1.in_features
        blk.mlp.fc1, blk.mlp.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)
    return m

ARCH = {   # route -> (what runs on the CPU after decoding, builder)
    "DeiT-S baseline (FP32, 88.2 MB)": lambda: timm.create_model("deit_small_patch16_224", pretrained=False),
    "A / B: decode to DeiT-S": lambda: timm.create_model("deit_small_patch16_224", pretrained=False),
    "C: pruned DeiT-S (MLP 1162)": pruned_deit_s,
    "D: DeiT-Tiny": lambda: timm.create_model("deit_tiny_patch16_224", pretrained=False),
}

engines = torch.backends.quantized.supported_engines
torch.backends.quantized.engine = next(e for e in ("x86", "fbgemm", "qnnpack") if e in engines)  # x86 first

def bench(model, threads):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        for _ in range(a.warmup): model(x)
        ts = []
        for _ in range(a.runs):
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1e3)
    q = statistics.quantiles(ts, n=4)
    return {"median_ms": statistics.median(ts), "q1_ms": q[0], "q3_ms": q[2]}

rows, seen = [], {}
for name, build in ARCH.items():
    m = build().eval()
    params = sum(p.numel() for p in m.parameters())
    for kind in ("fp32", "int8-dynamic"):
        mm = m if kind == "fp32" else torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        for th in a.threads:
            key = (params, kind, th)
            r = seen.get(key) or bench(mm, th)       # A/B and the baseline share an architecture
            seen[key] = r
            rows.append({"model": name, "params": params, "kind": kind, "threads": th, **r})
            print(f"{name:<34}{kind:>14}{th:>4} thr  median {r['median_ms']:7.2f} ms  "
                  f"IQR [{r['q1_ms']:.2f}, {r['q3_ms']:.2f}]", flush=True)

meta = {"machine": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "timm": timm.__version__, "qengine": torch.backends.quantized.engine,
        "runs": a.runs, "warmup": a.warmup, "batch": 1}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"meta": meta, "rows": rows}, open(a.out, "w"), indent=2)
print(f"\n{meta}\nWrote {a.out}")
