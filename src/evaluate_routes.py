#!/usr/bin/env python3
"""
Phase 3 — the actual experiment.

Four models, same measured size, same training budget. Do they behave the same?

  1. Clean top-1                  the number everyone reports
  2. Corruption robustness        ImageNet-C corruptions, generated on the fly
  3. Calibration (ECE)            does 90% confidence mean 90% correct
  4. Per-example agreement        do the routes fail on the SAME images

Corruptions are generated ONCE and cached, so every model is scored on
byte-identical inputs. Generating them per model would be 4x the work and would
let RNG differences leak into the comparison.

Usage:
    python src/evaluate_routes.py --smoke                  # 64 images, 2 corruptions
    python src/evaluate_routes.py --n 1000 --severities 3
    python src/evaluate_routes.py --n 1000 --severities 1 3 5
"""

import argparse, itertools, json, os, sys, time
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1_audit import MB, is_quantizable
from build_routes import prune_mlp_channels, sensitivity_order, make_policy

CACHE = "eval_cache"
SLOW = {"glass_blur"}          # ~20x slower than the rest; excluded by --fast


# ---------------------------------------------------------------------------
# stage 1: cache clean + corrupted images ONCE
# ---------------------------------------------------------------------------

def load_clean(data_dir, n):
    """First n validation images, deterministic, as uint8 HWC at 224x224."""
    import webdataset as wds, glob
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
    files = sorted(glob.glob(os.path.join(data_dir, "imagenet1k-validation-*.tar")))
    if not files:
        raise SystemExit(f"no validation shards in {data_dir}/")
    ds = (wds.WebDataset(files, shardshuffle=False, handler=wds.ignore_and_continue)
            .decode("pil").to_tuple("jpg;jpeg;png;webp", "cls"))
    imgs, labs = [], []
    for img, lab in ds:
        imgs.append(np.asarray(tf(img.convert("RGB")), dtype=np.uint8))
        labs.append(int(lab))
        if len(imgs) >= n:
            break
    if len(imgs) < n:
        print(f"  warning: only {len(imgs)} images available, wanted {n}")
    return np.stack(imgs), np.array(labs)


def _patch_skimage():
    """imagecorruptions calls skimage gaussian(multichannel=True), removed in
    scikit-image >=0.19 (now channel_axis). corruptions.py does
    `from skimage.filters import gaussian`, so the MODULE attribute must be
    rebound -- patching skimage itself is too late."""
    import inspect
    import imagecorruptions.corruptions as C
    if "multichannel" in inspect.signature(C.gaussian).parameters:
        return                                   # old skimage, nothing to do
    _orig = C.gaussian

    def gaussian(image, sigma=1, multichannel=None, **kw):
        if multichannel is not None:
            kw["channel_axis"] = -1 if multichannel else None
        return _orig(image, sigma=sigma, **kw)
    C.gaussian = gaussian


def _corrupt_one(task):
    """One (corruption, severity) over the whole batch -> one .npy file.

    Returns an error string instead of raising: one broken corruption must not
    abort the whole pool after minutes of work.
    """
    name, sev, src, dst = task
    try:
        _patch_skimage()
        from imagecorruptions import corrupt
        clean = np.array(np.load(src, mmap_mode="r"))
        out = np.empty_like(clean)
        t0 = time.time()
        for i in range(len(clean)):
            out[i] = corrupt(np.asarray(clean[i]), corruption_name=name, severity=sev)
        np.save(dst, out)
        return name, sev, time.time() - t0, None
    except Exception as e:
        return name, sev, 0.0, f"{type(e).__name__}: {e}"


def build_cache(data_dir, n, severities, fast, workers):
    os.makedirs(CACHE, exist_ok=True)
    clean_p, lab_p = f"{CACHE}/clean_{n}.npy", f"{CACHE}/labels_{n}.npy"
    if not os.path.exists(clean_p):
        print(f"loading {n} clean validation images ...")
        imgs, labs = load_clean(data_dir, n)
        np.save(clean_p, imgs); np.save(lab_p, labs)
    print(f"  clean: {clean_p}")

    from imagecorruptions import get_corruption_names
    names = [c for c in get_corruption_names() if not (fast and c in SLOW)]

    tasks = [(c, s, clean_p, f"{CACHE}/{c}_s{s}_{n}.npy")
             for c, s in itertools.product(names, severities)
             if not os.path.exists(f"{CACHE}/{c}_s{s}_{n}.npy")]
    if tasks:
        print(f"generating {len(tasks)} corrupted sets on {workers} processes "
              f"(one-time; cached afterwards)")
        import multiprocessing as pymp
        with pymp.Pool(min(workers, len(tasks))) as pool:
            for name, sev, dt, err in pool.imap_unordered(_corrupt_one, tasks):
                print(f"  {name} s{sev}: " +
                      (f"FAILED -- {err}" if err else f"{dt:.0f}s"), flush=True)
    else:
        print("  all corrupted sets already cached")

    conds = [("clean", 0, clean_p)]
    for c, s in itertools.product(names, severities):
        f = f"{CACHE}/{c}_s{s}_{n}.npy"
        if os.path.exists(f):
            conds.append((c, s, f))
        else:
            print(f"  MISSING, skipping: {f}")
    return conds, np.load(lab_p)


# ---------------------------------------------------------------------------
# stage 2: score each model on every cached condition
# ---------------------------------------------------------------------------

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def load_route(route, models_dir, seed=0):
    """Rebuild the trained model with its quantization policy attached.

    The sensitivity ranking MUST come from PRETRAINED weights -- that is what
    build_routes and train_routes used. Ranking trained weights gives a
    different order, a different protected set, and a model evaluated in a
    configuration it was never trained in.
    """
    import timm
    from train_routes import attach_fakequant, _rebuild_mlp
    cfg = {r["route"]: r for r in
           json.load(open(f"{models_dir}/routes.json"))["routes"]}[route]
    knob = cfg["knob"]
    name = "deit_tiny_patch16_224" if route == "D" else "deit_small_patch16_224"

    model = timm.create_model(name, pretrained=True)
    # .clone() is REQUIRED. .detach().cpu() on a CPU tensor returns the SAME
    # object, so the load_state_dict below would overwrite `ref` in place and
    # the sensitivity ranking would come from TRAINED weights -- which silently
    # cost route B 3.9 points.
    ref = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
    if route == "C":
        ref = prune_mlp_channels(ref, knob)

    ckpt = torch.load(f"{models_dir}/ChotaDeiT-{route}-s{seed}.ckpt", map_location="cpu")
    if route == "C":
        _rebuild_mlp(model, ckpt)
    model.load_state_dict(ckpt, strict=True)

    policy = make_policy(route, ref, knob, sensitivity_order(ref))
    attach_fakequant(model, lambda n_, w: policy(n_, w)
                     if is_quantizable(n_, w) else ("none", None, None))
    return model.cuda().eval(), policy


@torch.no_grad()
def score(model, arr, batch=128):
    """Return (predicted class, softmax confidence) per image."""
    preds, confs = [], []
    dev = "cuda"
    m, s = MEAN.to(dev), STD.to(dev)
    for i in range(0, len(arr), batch):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + batch]))
        x = x.permute(0, 3, 1, 2).to(dev).float()
        x = (x - m) / s
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(model(x).float(), dim=1)
        c, k = p.max(1)
        preds.append(k.cpu().numpy()); confs.append(c.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


# ---------------------------------------------------------------------------
# stage 3: metrics
# ---------------------------------------------------------------------------

def ece(conf, correct, bins=15):
    """Expected Calibration Error: |accuracy - confidence| per bin, size-weighted."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100.0 * e


def agreement(a, b):
    """How two models' failures overlap. a, b are boolean correct-vectors."""
    both_wrong = (~a & ~b).sum()
    a_only = (a & ~b).sum()
    b_only = (~a & b).sum()
    union_wrong = (~a | ~b).sum()
    return {"both_wrong": int(both_wrong),
            "only_A_correct": int(a_only), "only_B_correct": int(b_only),
            "failure_jaccard": float(both_wrong / union_wrong) if union_wrong else 1.0,
            "disagree_pct": float(100.0 * (a != b).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--severities", type=int, nargs="+", default=[3])
    ap.add_argument("--fast", action="store_true", help="skip glass_blur (~20x slower)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0, help="which training seed to evaluate")
    ap.add_argument("--clean-only", action="store_true",
                    help="skip corruption generation entirely. Use for high-n clean-accuracy "
                         "tests: corrupting 50k images needs ~15 GB per worker and will OOM.")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.severities, a.fast, a.routes = 64, [3], True, ["A", "B"]

    assert not torch.cuda.is_initialized(), \
        "CUDA already initialised; the corruption Pool must fork before any CUDA use"
    if a.out is None:
        a.out = f"logs/eval_s{a.seed}.json"
    if a.clean_only:
        os.makedirs(CACHE, exist_ok=True)
        cp, lp = f"{CACHE}/clean_{a.n}.npy", f"{CACHE}/labels_{a.n}.npy"
        if not os.path.exists(cp):
            print(f"loading {a.n} clean validation images ...")
            imgs, labs = load_clean(a.data, a.n)
            np.save(cp, imgs); np.save(lp, labs)
        conds, labels = [("clean", 0, cp)], np.load(lp)
        print(f"  clean-only: {cp}")
    else:
        conds, labels = build_cache(a.data, a.n, a.severities, a.fast, a.workers)
    labels = labels[:a.n]
    print(f"\n{len(conds)} conditions x {len(labels)} images x {len(a.routes)} models\n")

    res = {}
    for r in a.routes:
        model, _ = load_route(r, a.models, a.seed)
        res[r] = {}
        for cname, sev, path in conds:
            arr = np.load(path, mmap_mode="r")[:len(labels)]
            pred, conf = score(model, np.asarray(arr))
            ok = (pred == labels)
            key = "clean" if cname == "clean" else f"{cname}_s{sev}"
            res[r][key] = {"top1": float(100 * ok.mean()),
                           "ece": float(ece(conf, ok.astype(float))),
                           "mean_conf": float(100 * conf.mean()),
                           "correct": ok.tolist()}
        del model; torch.cuda.empty_cache()
        clean = res[r]["clean"]["top1"]
        corr = [v["top1"] for k, v in res[r].items() if k != "clean"]
        if corr:
            print(f"ChotaDeiT-{r}: clean {clean:.2f}% | corrupted mean "
                  f"{np.mean(corr):.2f}% | drop {clean - np.mean(corr):.2f} pts "
                  f"| ECE {res[r]['clean']['ece']:.2f}", flush=True)
        else:
            print(f"ChotaDeiT-{r}: clean {clean:.2f}% (n={len(labels)}) "
                  f"| ECE {res[r]['clean']['ece']:.2f}", flush=True)

    # pairwise failure agreement on clean images
    print("\nPER-EXAMPLE FAILURE AGREEMENT (clean):")
    pairs = {}
    for x, y in itertools.combinations(a.routes, 2):
        ax = np.array(res[x]["clean"]["correct"])
        ay = np.array(res[y]["clean"]["correct"])
        g = agreement(ax, ay)
        pairs[f"{x}-{y}"] = g
        print(f"  {x} vs {y}: disagree on {g['disagree_pct']:.1f}% of images | "
              f"failure overlap (Jaccard) {g['failure_jaccard']:.3f}")

    os.makedirs("logs", exist_ok=True)
    json.dump({"n": len(labels), "severities": a.severities,
               "conditions": [c[0] if c[0] == "clean" else f"{c[0]}_s{c[1]}" for c in conds],
               "results": res, "pairs": pairs}, open(a.out, "w"))
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
