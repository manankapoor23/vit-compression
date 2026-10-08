# GPU run — Kaggle T4 (recommended) or Colab

**What this run fixes:**

| Gap | What gets measured |
|---|---|
| P49 "stated accuracy loss" | Uncompressed DeiT-S and DeiT-Tiny scored in **our** pipeline on the same 50,000 images |
| `.bin` never loaded back | Every model is **decoded from its `.bin` file** (`src/decode_bin.py`) and scored. Decoder already proven: 21/21 files decode and re-encode byte-identically |
| O5 calibration | Temperature fitted on even-numbered images, ECE reported on odd ones |
| O6 accuracy line | The 5 group-sweep models scored under the same 14 corruptions |
| Latency on x86 | `cpu_latency.py` on Colab's Intel/AMD CPU (fbgemm/x86 backend) |

**Total time:** 1–2 hours **[estimate, not verified — the smoke run prints a real estimate]**.
**Disk needed:** ~25 GB (6.4 GB shards + 7.5 GB decoded images + 10.5 GB corruptions). On Kaggle the caches go to `/tmp`
because `/kaggle/working` is capped at 20 GB.

---

## Kaggle version (use this)

**Setup, once (10 min):**
1. Hugging Face: accept the terms at huggingface.co/datasets/timm/imagenet-1k-wds; create a *Read* token.
2. Kaggle → **Datasets → New Dataset** → upload `gpu_bundle/gpu_bundle.zip` → name it `vitc-bundle`, private.
3. New Notebook → right panel: **Accelerator: GPU T4**, **Internet: on**, **Add Input → vitc-bundle**.
4. **Add-ons → Secrets** → add `HF_TOKEN` = your token.

**Cell 1 — files, disk check**
```python
!mkdir -p /kaggle/working/vitc && cd /kaggle/working/vitc && unzip -q -o /kaggle/input/vitc-bundle/gpu_bundle.zip
!df -h /tmp /kaggle/working | cat ; nproc ; free -g | head -2
```
Needs ~25 GB free under `/tmp` (shards + decoded images + corruptions). `/kaggle/working` is capped
at 20 GB, so caches go to `/tmp`. **If `/tmp` shows less than 25 GB free, stop and tell Claude.**

**Cell 2 — libraries + ImageNet validation (~10 min)**
```python
!pip -q install webdataset imagecorruptions
from kaggle_secrets import UserSecretsClient
import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!huggingface-cli download timm/imagenet-1k-wds --repo-type dataset --include "*validation*" --local-dir /tmp/val
!find /tmp/val -name "imagenet1k-validation-*.tar" | wc -l      # must print 64
```

**Cell 3 — smoke test (~5–10 min). Do not skip.**
```python
%cd /kaggle/working/vitc
!python src/rescore.py --val /tmp/val --models evidence/models --out /kaggle/working/rescore.json --smoke
```
Must print `GPU Tesla T4, autocast torch.float16` and end with `SMOKE OK`. **Send Claude the
per-corruption seconds** — × 156 gives the real corruption time on 5,000 images, and that decides
whether the full run fits in one 12-hour session.

**Cell 4 — full run, in the background.** Put cells 1, 2 and this one in the notebook, then
**Save Version → Save & Run All (Commit)**. It runs up to 12 h without the browser open.
```python
%cd /kaggle/working/vitc
!python src/rescore.py --val /tmp/val --models evidence/models --out /kaggle/working/rescore.json
!python src/cpu_latency.py --threads 1 4 --out /kaggle/working/cpu_latency_x86.json
!lscpu | grep "Model name"
```
When it finishes, download `rescore.json` and `cpu_latency_x86.json` from the version's **Output**
tab into `evidence/rescore/`.

**About float16:** the T4 has no fast bfloat16, so it scores in float16 (recorded in the JSON).
Expect per-image agreement with the archived A100/H100 results slightly below 100%, from rounding
alone; the comparison script reports the exact figure.

---

## Colab version (A100 with High-RAM; free-tier Colab has only 2 CPUs)

## Before you start (5 min, once)

1. On Hugging Face, open **huggingface.co/datasets/timm/imagenet-1k-wds**, log in, accept the terms.
   This is the same ImageNet copy the original run used, so image order matches the archived results.
2. Create an access token: huggingface.co/settings/tokens → *Read*.
3. Upload `gpu_bundle/gpu_bundle.zip` (140 MB) to your Google Drive, top level.

## Colab cells (Runtime → Change runtime type → A100, High-RAM)

**Cell 1 — files**
```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/vitc && cd /content/vitc && unzip -q -o /content/drive/MyDrive/gpu_bundle.zip
!ls /content/vitc/src /content/vitc/evidence/models | head -40
```

**Cell 2 — libraries**
```python
!pip -q install webdataset imagecorruptions timm
import torch, numpy, timm; print(torch.__version__, numpy.__version__, timm.__version__, torch.cuda.get_device_name())
```

**Cell 3 — ImageNet validation (~5–10 min)**
```python
from huggingface_hub import login; login()          # paste the token
!huggingface-cli download timm/imagenet-1k-wds --repo-type dataset \
    --include "*validation*" --local-dir /content/val
# if huggingface-cli is missing: !hf download timm/imagenet-1k-wds --repo-type dataset --include "*validation*" --local-dir /content/val
!find /content/val -name "imagenet1k-validation-*.tar" | wc -l    # must print 64
```

**Cell 4 — smoke test (~5 min). Do not skip.**
```python
%cd /content/vitc
!python src/rescore.py --val /content/val --models evidence/models --out /content/drive/MyDrive/rescore.json --smoke
```
Must end with `SMOKE OK`. It runs all 14 corruptions on 32 images, so a library problem shows up
here, not an hour in. Note the per-corruption seconds: multiply by ~156 for the full 5,000 images.
**If a corruption FAILS:** copy the error line to me before going further.

**Cell 5 — full run**
```python
!python src/rescore.py --val /content/val --models evidence/models --out /content/drive/MyDrive/rescore.json
```
It saves to Drive after **every** model. If Colab disconnects, rerun cells 1–3 and this cell: it skips
finished models (the decoded images in `/tmp` are lost on disconnect and get rebuilt, ~5 min).

**Cell 6 — x86 CPU latency (~5 min, can run any time)**
```python
!lscpu | grep "Model name"
!python src/cpu_latency.py --threads 1 4 --out /content/drive/MyDrive/cpu_latency_x86.json
```

## After the run

Download `rescore.json` and `cpu_latency_x86.json` from Drive into `evidence/rescore/` and tell
Claude. The next step (on the Mac, no GPU) compares them with the archived results:

1. **Same images?** The route models' clean accuracies should match the archived 50,000-image
   numbers (A 76.49, B 72.40, C 75.68, D 71.41) within ~0.1 points, and per-image agreement should be
   close to 100%. That proves the `.bin` files *are* the models we reported.
2. **Accuracy loss** = DeiT-S FP32 − ChotaDeiT-A, measured here instead of quoted.
3. **O5**: ECE after temperature, per route.
4. **O6**: the 7-point DeiT-S line and ternary's residual against the pre-declared criterion.

The corruptions are regenerated with fixed seeds, so they are not byte-identical to the original
cache. Every model is re-scored on the new set, so comparisons stay fair; corrupted numbers will
differ slightly from the archived ones and are reported as a separate run, not merged.
