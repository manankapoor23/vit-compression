#!/usr/bin/env python3
"""
Re-score every model FROM ITS .bin FILE on one GPU (Colab / Kaggle A100).

Fills the gaps left when the original GPU allocation ended:
  G1  P49 "stated accuracy loss": the uncompressed DeiT-S (and DeiT-Tiny) scored in THIS pipeline
  G2  the .bin files themselves evaluated (decoded by src/decode_bin.py), not training checkpoints
  G3  O5 calibration: temperature fitted on even-indexed clean images, ECE reported on odd ones
  G4  O6 accuracy line: the 5 group-sweep models scored under the same corruptions

Models: deit_s_fp32, deit_tiny_fp32 (timm pretrained), 12 route files (A-D x seeds 0-2),
5 sweep files (A-g32..g512, seed 0).
Data:   ImageNet-1k validation as the 64 WebDataset shards of timm/imagenet-1k-wds -- the same
        shards the original run used, so image order matches the archived per-image arrays.
Output: one JSON, rewritten after every model (a disconnect loses at most one model).

Usage:
  python src/rescore.py --val /content/val --models evidence/models --out /content/drive/MyDrive/rescore.json --smoke
  python src/rescore.py --val /content/val --models evidence/models --out /content/drive/MyDrive/rescore.json
"""
import argparse, glob, json, os, sys, time
import multiprocessing as pymp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "motion_blur",
               "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
               "elastic_transform", "pixelate", "jpeg_compression"]          # glass_blur excluded, as before
SEVERITY = 3
# bfloat16 (what the original A100/H100 runs used) needs compute capability >= 8.0.
# A T4 (7.5) gets float16 instead; the choice is recorded in the output JSON.
AMP = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


# ---------------------------------------------------------------- data
def load_val(val_dir, n, cache):
    """First n validation images in shard order, Resize(256)+CenterCrop(224), uint8, to a memmap."""
    img_p, lab_p = f"{cache}/clean_{n}.npy", f"{cache}/labels_{n}.npy"
    if os.path.exists(img_p) and os.path.exists(lab_p):
        return np.load(img_p, mmap_mode="r"), np.load(lab_p)
    import webdataset as wds
    from torchvision import transforms as T
    files = sorted(glob.glob(os.path.join(val_dir, "**", "imagenet1k-validation-*.tar"), recursive=True),
                   key=os.path.basename)   # HF download layout not verified: search subfolders
    if len(files) != 64:
        raise SystemExit(f"expected 64 validation shards in {val_dir}, found {len(files)}")
    tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
    ds = (wds.WebDataset(files, shardshuffle=False, handler=wds.ignore_and_continue)
          .decode("pil").to_tuple("jpg;jpeg;png;webp", "cls"))
    out = np.lib.format.open_memmap(img_p + ".tmp", mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
    labs, t0 = [], time.time()
    for i, (img, lab) in enumerate(ds):
        if i >= n: break
        out[i] = np.asarray(tf(img.convert("RGB")), dtype=np.uint8); labs.append(int(lab))
        if i % 10000 == 0: print(f"  decoded {i:,}/{n:,}  {time.time()-t0:.0f}s", flush=True)
    if len(labs) != n:
        raise SystemExit(f"only {len(labs)} images decoded, wanted {n}")
    out.flush(); del out
    os.replace(img_p + ".tmp", img_p); np.save(lab_p, np.array(labs))
    return np.load(img_p, mmap_mode="r"), np.load(lab_p)


def _compat():
    """Version shims for imagecorruptions. GUESSED, not verified on Colab's versions:
    the smoke run exists to catch these."""
    # NumPy 2 removed these aliases; imagecorruptions (fog, ...) still uses np.float_.
    # Found by the Kaggle smoke run, 2026-10-08. Only set names that are missing.
    for alias, real in (("float_", np.float64), ("complex_", np.complex128), ("float", float),
                        ("int", int), ("bool8", np.bool_)):
        if alias not in np.__dict__:
            setattr(np, alias, real)
    import scipy.ndimage
    if "scipy.ndimage.interpolation" not in sys.modules:
        try:
            import scipy.ndimage.interpolation  # noqa: F401
        except Exception:
            sys.modules["scipy.ndimage.interpolation"] = scipy.ndimage
    from evaluate_routes import _patch_skimage
    _patch_skimage()


def _corrupt(task):
    name, src, dst, n, seed = task
    try:
        _compat()
        from imagecorruptions import corrupt
        np.random.seed(seed)                     # reproducible corruption noise
        clean = np.load(src, mmap_mode="r")[:n]
        out = np.empty_like(clean); t0 = time.time()
        for i in range(n):
            out[i] = corrupt(np.asarray(clean[i]), corruption_name=name, severity=SEVERITY)
        np.save(dst, out)
        return name, time.time() - t0, None
    except Exception as e:
        return name, 0.0, f"{type(e).__name__}: {e}"


def build_corruptions(clean_path, n, cache, names, workers):
    tasks = [(c, clean_path, f"{cache}/{c}_s{SEVERITY}_{n}.npy", n, 1000 + k)
             for k, c in enumerate(names) if not os.path.exists(f"{cache}/{c}_s{SEVERITY}_{n}.npy")]
    if tasks:
        print(f"generating {len(tasks)} corruptions x {n} images on {workers} processes", flush=True)
        with pymp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            for name, dt, err in pool.imap_unordered(_corrupt, tasks):
                print(f"  {name}: {'FAILED -- ' + err if err else f'{dt:.0f}s'}", flush=True)
                if err: raise SystemExit(f"corruption {name} failed; fix before scoring")
    return {c: np.load(f"{cache}/{c}_s{SEVERITY}_{n}.npy", mmap_mode="r") for c in names}


# ---------------------------------------------------------------- models
def model_list(models_dir, smoke):
    m = [("deit_s_fp32", None), ("deit_tiny_fp32", None)]
    m += [(f"{r}-s{s}", f"{models_dir}/ChotaDeiT-{r}-s{s}.bin") for r in "ABCD" for s in (0, 1, 2)]
    m += [(f"A-g{g}-s0", f"{models_dir}/ChotaDeiT-A-g{g}-s0.bin") for g in (32, 64, 128, 256, 512)]
    if smoke:
        m = [m[0], ("A-s0", f"{models_dir}/ChotaDeiT-A-s0.bin"), ("C-s0", f"{models_dir}/ChotaDeiT-C-s0.bin"),
             ("D-s0", f"{models_dir}/ChotaDeiT-D-s0.bin")]
    for _, p in m:
        if p and not os.path.exists(p): raise SystemExit(f"missing model file {p}")
    return m


def load_model(key, path):
    import timm
    if path is None:
        arch = "deit_small_patch16_224" if key == "deit_s_fp32" else "deit_tiny_patch16_224"
        return timm.create_model(arch, pretrained=True).cuda().eval()
    from decode_bin import build_model
    return build_model(path, device="cuda")


@torch.no_grad()
def logits_of(model, arr, batch=250):
    out, m, s = [], MEAN.cuda(), STD.cuda()
    for i in range(0, len(arr), batch):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + batch])).cuda().permute(0, 3, 1, 2).float()
        with torch.autocast("cuda", dtype=AMP):
            out.append(model((x - m) / s).float())
    return torch.cat(out)


# ---------------------------------------------------------------- metrics (same ECE as evaluate_routes)
def ece(conf, correct, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100.0 * e


def summarise(logits, labels, T=1.0):
    p = torch.softmax(logits / T, 1); conf, pred = p.max(1)
    conf, pred = conf.cpu().numpy(), pred.cpu().numpy(); ok = pred == labels
    return {"top1": float(100 * ok.mean()), "ece": float(ece(conf, ok.astype(float))),
            "mean_conf": float(100 * conf.mean()), "correct": ok.tolist()}


def fit_temperature(logits, labels):
    """One scalar T minimising NLL (Guo et al. 2017), LBFGS on log T."""
    y = torch.as_tensor(labels, device=logits.device)
    logT = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(logits / logT.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(logT.detach().exp())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="dir with imagenet1k-validation-0000..0063.tar")
    ap.add_argument("--models", default="evidence/models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/rescore_cache")
    ap.add_argument("--n-clean", type=int, default=50000)
    ap.add_argument("--n-corrupt", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--smoke", action="store_true", help="512 clean / 32 corrupted images, 4 models, all 14 corruptions")
    a = ap.parse_args()
    if a.smoke:
        a.n_clean, a.n_corrupt = 512, 32
        a.out = a.out.replace(".json", "_smoke.json"); a.cache += "_smoke"
    os.makedirs(a.cache, exist_ok=True)
    t_all = time.time()

    clean, labels = load_val(a.val, a.n_clean, a.cache)
    corrupt = build_corruptions(f"{a.cache}/clean_{a.n_clean}.npy", a.n_corrupt, a.cache, CORRUPTIONS, a.workers)
    lab_c = labels[:a.n_corrupt]
    print(f"data ready in {time.time()-t_all:.0f}s", flush=True)

    R = json.load(open(a.out)) if os.path.exists(a.out) else {
        "n_clean": a.n_clean, "n_corrupt": a.n_corrupt, "severity": SEVERITY,
        "conditions": ["clean"] + [f"{c}_s{SEVERITY}" for c in CORRUPTIONS],
        "labels_sha": None, "results": {}, "temperature": {}, "timing_s": {}}
    R["labels_first20"] = labels[:20].tolist()
    R["gpu"], R["amp_dtype"] = torch.cuda.get_device_name(), str(AMP)
    print(f"GPU {R['gpu']}, autocast {AMP}", flush=True)

    for key, path in model_list(a.models, a.smoke):
        if key in R["results"]:
            print(f"{key}: already done, skipping"); continue
        t0 = time.time()
        model = load_model(key, path)
        lg = logits_of(model, clean)
        res = {"clean": summarise(lg, labels)}
        ev, od = np.arange(0, a.n_clean, 2), np.arange(1, a.n_clean, 2)
        T = fit_temperature(lg[ev], labels[ev])
        R["temperature"][key] = {"T": T, "ece_odd_before": summarise(lg[od], labels[od])["ece"],
                                 "ece_odd_after": summarise(lg[od], labels[od], T)["ece"]}
        res["clean_first_corrupt_subset"] = {"top1": float(100 * np.mean(res["clean"]["correct"][:a.n_corrupt]))}
        for c in CORRUPTIONS:
            res[f"{c}_s{SEVERITY}"] = summarise(logits_of(model, corrupt[c]), lab_c)
        del model; torch.cuda.empty_cache()
        R["results"][key] = res; R["timing_s"][key] = time.time() - t0
        tmp = a.out + ".tmp"; json.dump(R, open(tmp, "w")); os.replace(tmp, a.out)
        cm = np.mean([res[f"{c}_s{SEVERITY}"]["top1"] for c in CORRUPTIONS])
        print(f"{key:<14} clean {res['clean']['top1']:6.2f}  corrupted {cm:6.2f}  ECE {res['clean']['ece']:5.2f}"
              f" -> {R['temperature'][key]['ece_odd_after']:5.2f} (T={T:.3f})  {time.time()-t0:.0f}s", flush=True)

    print(f"\nwrote {a.out}  total {time.time()-t_all:.0f}s")
    if a.smoke:
        per_model = np.mean(list(R["timing_s"].values()))
        scale = (50000 + 14 * 5000) / (a.n_clean + 14 * a.n_corrupt)
        print(f"SMOKE OK. Rough full-run model time: 19 models x {per_model*scale/60:.1f} min "
              f"(corruption generation not included -- see per-corruption seconds above, x{5000//a.n_corrupt})")


if __name__ == "__main__":
    main()
