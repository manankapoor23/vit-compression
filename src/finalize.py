#!/usr/bin/env python3
"""
Recover a route that trained but died before exporting.

Loads models/ChotaDeiT-<R>.ckpt, runs the final eval, writes the .bin at the
route's policy, and records the result. No retraining.

Usage:  python src/finalize.py --route B
"""
import argparse, json, os, sys
from collections import OrderedDict
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1_audit import MB, is_quantizable
from build_routes import write_model, prune_mlp_channels, sensitivity_order, make_policy
from train_routes import attach_fakequant, loaders, evaluate, _rebuild_mlp

ap = argparse.ArgumentParser()
ap.add_argument("--route", required=True)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--final-batches", type=int, default=100)
ap.add_argument("--data", default="data")
ap.add_argument("--models", default="models")
a = ap.parse_args()

import timm
cfg = {r["route"]: r for r in json.load(open(f"{a.models}/routes.json"))["routes"]}[a.route]
knob = cfg["knob"]
name = "deit_tiny_patch16_224" if a.route == "D" else "deit_small_patch16_224"

model = timm.create_model(name, pretrained=True)
ckpt = torch.load(f"{a.models}/ChotaDeiT-{a.route}.ckpt", map_location="cpu")
if a.route == "C":
    _rebuild_mlp(model, ckpt)
model.load_state_dict(ckpt, strict=True)

# The sensitivity ranking MUST come from the PRETRAINED weights -- that is what
# build_routes.py and train_routes.py both used. Ranking the TRAINED weights
# gives a different order, so a different set of layers gets protected: the
# model is then evaluated in a configuration it was never trained in, and the
# file size drifts out of tolerance.
_pre = timm.create_model(name, pretrained=True)
ref = OrderedDict((k, v.detach().cpu().clone()) for k, v in _pre.state_dict().items())
if a.route == "C":
    ref = prune_mlp_channels(ref, knob)
del _pre
policy = make_policy(a.route, ref, knob, sensitivity_order(ref))
attach_fakequant(model, lambda n_, w: policy(n_, w) if is_quantizable(n_, w) else ("none", None, None))
model.cuda().eval()

_, val = loaders(a.data, a.batch, 0)
acc = evaluate(model, val, "cuda", max_batches=a.final_batches)
print(f"ChotaDeiT-{a.route}  top-1 = {acc:.2f}%  "
      f"({a.final_batches*a.batch:,} images)", flush=True)

export = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())
res = write_model(export, f"{a.models}/ChotaDeiT-{a.route}.bin", policy)
print(f"Exported: {res['bytes']/MB:.3f} MB")
json.dump({"route": a.route, "final_top1": acc, "mb": res["bytes"]/MB, "knob": knob},
          open(f"logs/final_{a.route}.json", "w"), indent=2)
