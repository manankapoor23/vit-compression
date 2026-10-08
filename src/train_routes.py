#!/usr/bin/env python3
"""
Recovery training for ChotaDeiT-A / B / C / D.

Quantization-aware training with a straight-through estimator (STE): the forward
pass uses quantized weights, the backward pass updates full-precision master
weights. At export the same policy is re-applied, so the file size is identical
to what build_routes.py produced.

THE CONTROL: every route gets exactly --steps optimizer steps. Not epochs, not
"until converged". Any behavioural difference later therefore cannot be blamed
on one route having been trained harder.

Usage:
    python src/train_routes.py --route A --steps 3000
    python src/train_routes.py --route D --steps 3000 --distill     # teacher -> DeiT-Tiny
"""

import argparse, json, math, os, random, sys, time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

# /dev/shm is 1 GB here and cannot be remounted (no CAP_SYS_ADMIN), so tensors
# must be passed between workers as FILES, not shared-memory segments.
_TMP = "/tmp/vitc"                       # LOCAL overlay disk, NOT /workspace (NFS)
os.makedirs(_TMP, exist_ok=True)
os.environ["TMPDIR"] = _TMP
mp.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1_audit import MB, component_of, is_quantizable
from build_routes import write_model, prune_mlp_channels, sensitivity_order, make_policy


# ---------------------------------------------------------------------------
# fake quantization with a straight-through estimator
# ---------------------------------------------------------------------------

def _fq_int(w, bits, group):
    """Quantize-dequantize, group-wise symmetric. Same math as build_routes."""
    shp, flat = w.shape, w.reshape(-1)
    n = flat.numel(); g = min(group, n); pad = (-n) % g
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    grp = flat.reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    s = (grp.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-8).half().float()
    q = torch.clamp(torch.round(grp / s), -qmax - 1, qmax)
    return (q * s).reshape(-1)[:n].reshape(shp)


def _fq_ternary(w):
    """TWN threshold rule, per output channel. 3 levels, not 4."""
    x = w.reshape(w.shape[0], -1)
    d = 0.7 * x.abs().mean(1, keepdim=True)
    t = torch.zeros_like(x)
    t = torch.where(x > d, torch.ones_like(x), t)
    t = torch.where(x < -d, -torch.ones_like(x), t)
    nz = t.abs().sum(1, keepdim=True).clamp(min=1)
    a = ((x.abs() * t.abs()).sum(1, keepdim=True) / nz).half().float()
    return (t * a).reshape(w.shape)


class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, kind, bits, group):
        if kind == "int":     return _fq_int(w, bits, group)
        if kind == "ternary": return _fq_ternary(w)
        if kind == "fp16":    return w.half().float()
        return w

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None          # straight through


def attach_fakequant(model, policy):
    """Route every Linear/Conv2d weight through FakeQuant, per the route policy."""
    patched = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Conv2d)):
            continue
        wname = f"{name}.weight"
        kind, bits, group = policy(wname, mod.weight)
        if kind == "none":
            continue
        mod._q = (kind, bits, group)
        if isinstance(mod, nn.Linear):
            mod.forward = (lambda m: lambda x: F.linear(
                x, FakeQuant.apply(m.weight, *m._q), m.bias))(mod)
        else:
            mod.forward = (lambda m: lambda x: F.conv2d(
                x, FakeQuant.apply(m.weight, *m._q), m.bias,
                m.stride, m.padding, m.dilation, m.groups))(mod)
        patched += 1
    return patched


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

_SEED = [0]          # set from main(); worker_init_fn needs a module-level handle


def loaders(data_dir, batch, workers, n_train_shards=200):
    """WebDataset tar shards. Sequential reads -- NFS does 1494 files/sec on
    loose files, which would starve the GPU; shards avoid that entirely."""
    import webdataset as wds
    from torchvision import transforms as T

    tr = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.PILToTensor()])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])
    # PILToTensor -> uint8. Normalisation happens on GPU: 4x less shared memory.

    import glob as _glob

    def shards(pat):
        f = sorted(_glob.glob(os.path.join(data_dir, pat)))
        if not f:
            raise SystemExit(f"no shards match {pat} in {data_dir}/ -- check the download")
        return f

    def pipe(files, tf, shuffle):
        # pass an explicit FILE LIST: webdataset does brace expansion, not shell
        # globs, so "train-0[0-1]*.tar" would silently match nothing.
        kw = dict(shardshuffle=shuffle, handler=wds.ignore_and_continue)
        try:
            d = wds.WebDataset(files, empty_check=False, **kw)
        except TypeError:
            d = wds.WebDataset(files, **kw)          # older webdataset
        if shuffle:
            d = d.shuffle(2000)
        return (d.decode("pil")
                 .to_tuple("jpg;jpeg;png;webp", "cls")
                 .map_tuple(tf, lambda y: int(y)))

    tr_files = shards("imagenet1k-train-*.tar")[:n_train_shards]
    va_files = shards("imagenet1k-validation-*.tar")
    print(f"  data: {len(tr_files)} train shards, {len(va_files)} val shards")

    train = torch.utils.data.DataLoader(
        pipe(tr_files, tr, True).batched(batch),
        batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None,
        persistent_workers=False,
        worker_init_fn=lambda wid: (np.random.seed(_SEED[0] + wid),
                                    random.seed(_SEED[0] + wid)))
    val = torch.utils.data.DataLoader(
        pipe(va_files, te, False).batched(batch),
        batch_size=None, num_workers=0)      # 0 workers => zero shm pressure
    return train, val


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def to_gpu(x, dev):
    return (x.to(dev, non_blocking=True).float() - MEAN.to(dev)) / STD.to(dev)


@torch.no_grad()
def evaluate(model, val, dev, max_batches=None):
    model.eval(); c = t = 0
    for i, (x, y) in enumerate(val):
        if max_batches and i >= max_batches: break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(to_gpu(x, dev)).argmax(1)
        c += (p.cpu() == y).sum().item(); t += y.numel()
    model.train()
    return 100.0 * c / max(t, 1)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=list("ABCD"))
    ap.add_argument("--steps", type=int, default=3000, help="THE CONTROL: identical for every route")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)   # 1 GB shm is hard-capped here
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--routes-json", default="models/routes.json")
    ap.add_argument("--distill", action="store_true", help="KL against DeiT-Small teacher")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--knob-override", type=float, default=None,
                    help="force the route knob instead of reading routes.json. For sweeps: "
                         "the file size then floats instead of hitting the target.")
    ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. g128")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed. Models are ONE draw from a random process; "
                         "the headline comparison needs several to have error bars.")
    ap.add_argument("--final-batches", type=int, default=100,
                    help="batches for the final eval; 100 x 256 = 25,600 images")
    ap.add_argument("--smoke", action="store_true",
                    help="20 steps + tiny eval, to validate the pipeline before a real run")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.workers = 20, 20, 0

    import timm
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = {r["route"]: r for r in json.load(open(args.routes_json))["routes"]}[args.route]
    knob = cfg["knob"] if args.knob_override is None else args.knob_override
    if args.knob_override is not None and args.route != "C":
        knob = int(round(knob))
    TAG = (f"-{args.tag}" if args.tag else "")
    base_name = "deit_tiny_patch16_224" if args.route == "D" else "deit_small_patch16_224"

    model = timm.create_model(base_name, pretrained=True)
    if args.route == "C":
        sd = prune_mlp_channels(OrderedDict(model.state_dict()), knob)
        model = timm.create_model(base_name, pretrained=False)
        _rebuild_mlp(model, sd)                 # resize MLP layers to the pruned shapes
        model.load_state_dict(sd, strict=True)

    state = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())
    order = sensitivity_order(state)
    policy = make_policy(args.route, state, knob, order)

    def train_policy(name, w):
        if not is_quantizable(name, w):
            return ("none", None, None)
        return policy(name, w)

    n = attach_fakequant(model, train_policy)
    model.to(dev).train()
    print(f"ChotaDeiT-{args.route} seed={args.seed}: {n} layers fake-quantized | "
          f"knob={knob} | {args.steps} steps")

    teacher = None
    if args.distill:
        teacher = timm.create_model("deit_small_patch16_224", pretrained=True).to(dev).eval()
        for p in teacher.parameters(): p.requires_grad_(False)

    _SEED[0] = args.seed * 1000
    train, val = loaders(args.data, args.batch, args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    os.makedirs("logs", exist_ok=True); os.makedirs(args.models, exist_ok=True)
    hist, step, t0 = [], 0, time.time()
    it = iter(train)
    while step < args.steps:
        try: x, y = next(it)
        except StopIteration:
            it = iter(train); continue
        x, y = to_gpu(x, dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
            if teacher is not None:
                with torch.no_grad(): tl = teacher(x)
                loss = 0.5 * loss + 0.5 * F.kl_div(
                    F.log_softmax(out / 2, 1), F.softmax(tl / 2, 1),
                    reduction="batchmean") * 4.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

        if step % (5 if args.smoke else 50) == 0:
            print(f"  step {step}/{args.steps}  loss {loss.item():.3f}  "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            del it                                  # shut train workers down first
            acc = evaluate(model, val, dev, max_batches=40)
            it = iter(train)
            hist.append({"step": step, "top1_subset": acc})
            print(f"  >> step {step}: top-1 (40 batches) = {acc:.2f}%", flush=True)
            torch.save(model.state_dict(), f"{args.models}/ChotaDeiT-{args.route}{TAG}-s{args.seed}.ckpt")

    torch.cuda.empty_cache()
    final = evaluate(model, val, dev, max_batches=20 if args.smoke else args.final_batches)
    print(f"  (final eval over {args.final_batches} batches "
          f"= {args.final_batches*args.batch:,} images)", flush=True)
    print(f"\nFINAL top-1: {final:.2f}%", flush=True)

    export = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())
    path = f"{args.models}/ChotaDeiT-{args.route}{TAG}-s{args.seed}.bin"
    res = write_model(export, path, policy)
    print(f"Exported {path}: {res['bytes']/MB:.3f} MB (target {cfg['mb']:.3f})")

    json.dump({"route": args.route, "seed": args.seed, "steps": args.steps,
               "knob": knob, "final_top1": final, "history": hist,
               "mb": res["bytes"]/MB},
              open(f"logs/train_{args.route}{TAG}_s{args.seed}.json", "w"), indent=2)


def _rebuild_mlp(model, sd):
    """Resize each block's MLP to match the pruned state dict."""
    for i, blk in enumerate(model.blocks):
        h = sd[f"blocks.{i}.mlp.fc1.weight"].shape[0]
        d = blk.mlp.fc1.in_features
        blk.mlp.fc1 = nn.Linear(d, h)
        blk.mlp.fc2 = nn.Linear(h, d)


if __name__ == "__main__":
    main()
