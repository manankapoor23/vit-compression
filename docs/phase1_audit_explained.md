# `phase1_audit.py` — Line by Line, and Why

Nothing in this script should be a black box to you. Every judgment call below is one an
examiner can ask about, so each section gives **what it does**, **why it's done that way**,
and **what to say if challenged**.

---

## 0. What the script is for, in one paragraph

The ViT compression literature reports model size as `parameters × bits ÷ 8`. That number is
wrong — it ignores the scale metadata that quantization requires, and it ignores the layers
that standard tools refuse to quantize. This script loads DeiT-Small, **actually writes the
files to disk**, and measures them with `os.path.getsize()`. It answers three questions:
where the parameters are, what each format really weighs, and how much of your budget the
un-quantized leftovers eat. **It measures size only. No accuracy, no GPU, no ImageNet** —
that is Phase 2.

---

## 1. Map of the file

```
  imports + MB constant
      │
      ├── component_of()          which architectural part a tensor belongs to
      ├── is_quantizable()        do we compress this tensor or leave it alone
      │
      ├── packed_bytes()          how many bytes n codes take at b bits
      ├── pack_codes()            actually do the bit-packing
      ├── base3_bytes()           ternary packing arithmetic (analysis only)
      │
      ├── quantize_groupwise()    float weights -> integer codes + scales
      │
      ├── serialize_float()       write FP32 / FP16 via torch.save
      ├── serialize_quantized()   write a quantized model as raw binary
      │
      └── main()                  4 report sections
```

---

## 2. The setup lines

```python
MB = 1_000_000
```

**What:** one megabyte = one million bytes (decimal), not 1,048,576 (binary, "MiB").

**Why:** model cards and papers quote decimal MB. DeiT-Small at 22.05M params × 4 bytes =
88.2 MB decimal, which is the "88 MB" figure you started from. If you used MiB you'd get
84.1 and your numbers wouldn't line up with anything published.

**If challenged:** *"Decimal MB, consistent with how model sizes are reported. Stated in the
code."* The only sin is switching between them silently.

---

## 3. `component_of(name)` — attribution

```python
if name.startswith("patch_embed"):  return "patch_embed"
if name.startswith("blocks"):
    if ".attn." in name:  return "blocks.attn"
    if ".mlp." in name:   return "blocks.mlp"
```

**What:** maps a PyTorch parameter name to an architectural bucket. In timm, DeiT parameters
are named like `blocks.7.mlp.fc1.weight`, so string matching is enough.

**Why:** the entire point of section 1 is knowing that **64% of the model is the MLP layers**.
You can't say that without attribution.

**Watch out:** `final_norm` is matched by `name.startswith("norm")` — which works only because
it comes *after* the `blocks` check. A parameter named `blocks.3.norm1.weight` starts with
`blocks`, so it's caught earlier and bucketed as `blocks.norm`. Order matters here.

---

## 4. `is_quantizable(name, tensor)` — the compression policy

```python
return tensor.dim() >= 2 and name.endswith("weight")
```

**What:** only compress 2-D-or-larger tensors whose name ends in `weight`.

**Why two conditions:**

| Excluded by | Examples | Reason |
|---|---|---|
| `dim() >= 2` | LayerNorm weights, all biases, `cls_token` | 1-D tensors are tiny (19k params total) and highly sensitive — quantizing them costs accuracy and saves nothing |
| `endswith("weight")` | `pos_embed` (3-D but not a "weight") | Position embeddings are learned lookups, conventionally kept at full precision |

This is the standard convention in AWQ, GPTQ, llama.cpp. **You're not inventing a policy —
you're reproducing the field's.** That's the right call for an audit whose job is to measure
what the field actually ships.

---

## 5. `packed_bytes()` and `pack_codes()` — the bit-packing

```python
def packed_bytes(n_values, bits):
    return math.ceil(n_values * bits / 8)
```

**What:** the exact byte count. `ceil` because you can't write a fraction of a byte.

```python
c = codes.astype(np.uint8).reshape(-1, 1)
bitmat = np.unpackbits(c, axis=1)[:, 8 - bits:]
flat = bitmat.reshape(-1)
pad = (-len(flat)) % 8
flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
return np.packbits(flat).tobytes()
```

**What it's doing, step by step:**

1. `unpackbits` turns each byte into its 8 individual bits — an N×8 matrix of 0s and 1s.
2. `[:, 8 - bits:]` keeps only the **low `bits` bits** of each code. For 4-bit, that's the
   last 4 columns; the top 4 are always zero because codes are < 16.
3. `reshape(-1)` lays every kept bit end to end in one long stream.
4. Pad to a multiple of 8 (`(-len) % 8` is the idiom for "how many to reach the next multiple").
5. `packbits` turns the bit stream back into bytes.

**Why do it this way:** it's exact and it's general. A 3-bit model genuinely occupies
`ceil(n × 3 / 8)` bytes, which no simple `astype` would give you. Three 4-bit codes don't fit
in "one and a half bytes" unless you actually pack the bits.

**Verified:** tested at 2, 3, 4 and 8 bits — produces exactly `ceil(n×bits/8)` bytes and
round-trips back to the original codes.

---

## 6. `quantize_groupwise()` — the heart of it

```python
flat = w.detach().float().reshape(-1)
g = min(group_size, n)
pad = (-n) % g
groups = flat.reshape(-1, g)
```

**What:** flatten the weight matrix, chop into groups of `g` consecutive values, pad the last
group if needed.

**Why groups at all:** one scale for a whole matrix is too coarse — a single large outlier
stretches the scale and crushes the precision of everything else. Groups localise that damage.
**Smaller groups = better accuracy, more scales, bigger file.** That trade-off is the thing
section 2 measures, and nobody reports it in megabytes.

```python
qmax = 2 ** (bits - 1) - 1
scale = groups.abs().amax(dim=1, keepdim=True) / qmax
scale = torch.where(scale == 0, torch.ones_like(scale), scale)
scale = scale.half().float()
```

- `qmax` is the largest positive integer at this bit width. At 4 bits: `2³ − 1 = 7`.
- The scale maps the group's **largest absolute value** onto `qmax`.
- The `where` guards a group of all zeros — dividing by a zero scale gives NaN.
- **`.half().float()` is the important one.** It rounds the scale to FP16 precision and back,
  because that's how it will be *stored*. Without this you'd be measuring a file that used
  FP16 scales while computing error with FP32 ones — quietly optimistic.

```python
q = torch.clamp(torch.round(groups / scale), -qmax - 1, qmax)
err = (q * scale - groups).abs().max().item()
codes = (q + (qmax + 1)).to(torch.uint8).reshape(-1)[:n].numpy()
```

- Divide by scale, round, clamp into range. This is **RTN** (round-to-nearest).
- `err` is the worst single-weight reconstruction error — a cheap correctness check. If it's
  wildly large, something is broken.
- Clamp range is `[-qmax-1, qmax]`, e.g. `[-8, 7]` at 4 bits — the full signed range, all 16
  levels used.
- Adding `qmax + 1` shifts to unsigned `[0, 15]` so it fits in a `uint8` for packing.
- `[:n]` drops the padding added earlier.

**Known simplification, say it before you're asked:** groups are formed over the *flattened*
tensor, so a group can straddle two rows of the matrix. Real implementations group along the
input dimension within a row. **This slightly changes accuracy but not size at all** — and
this script only claims size.

**Second one, and it matters:** this is **RTN**, not GPTQ or AWQ. Those use calibration data
to choose better rounding. They produce **better accuracy at exactly the same file size**. So
your size numbers are valid for any of them; your accuracy numbers (Phase 2) would not be.

---

## 7. `serialize_float()` vs `serialize_quantized()`

```python
def serialize_float(state, dtype, path):
    torch.save({k: v.to(dtype) for k, v in state.items()}, path)
    return os.path.getsize(path)
```

FP32 and FP16 use `torch.save`, because that **is** how models are distributed. Measured
overhead: 35 KB on an 88 MB file — 0.04%, negligible.

The quantized path uses a raw binary container instead:

```python
hdr = json.dumps(header, separators=(",", ":")).encode()
f.write(len(hdr).to_bytes(4, "little"))
f.write(hdr)
f.write(payload)
```

Layout: `[4-byte header length][JSON header][concatenated payload]`. The header records, per
tensor, its name, whether it's quantized or FP16, its shape, and the byte lengths of its codes
and scales — everything needed to read it back.

**Why not `torch.save` here?** The first version did, and **every quantized file came out 4–10
MB larger than its own contents** — pickle framing around the packed byte objects. Since the
entire purpose is measuring real bytes, a serialization layer adding 45% noise defeats the
exercise. Raw binary means `getsize()` *is* the payload.

**This is a genuinely good thing to mention in a viva.** The project exists because the field
reports computed sizes instead of measured ones — and the first thing measuring caught was a
45% discrepancy in the measurement tool itself.

The function returns `file_bytes`, `exact_bytes` and `header_bytes` so any future gap between
"what I accounted for" and "what's on disk" is visible instead of silent.

---

## 8. `main()` — the four sections

### Section 1 — where the parameters are

Sums `numel()` per component, sorts descending, prints shares. Then:

```python
avg_bits = budget_bytes * 8 / total_params
```

**This is the number the whole project turns on:** 10 MB ÷ 22,050,664 params = **3.63 bits per
parameter**. Every design decision afterwards is "can I represent this weight in 3.63 bits."

### Section 2 — real file sizes

Sweeps every (bits, group size) pair, writes each file, measures it, and prints **nominal**
(the literature's `params × bits ÷ 8`) beside **ACTUAL** with the overhead percentage.

**This is your headline table.** The expected findings: INT4 doesn't fit the budget even
nominally, and the gap between nominal and actual grows as group size shrinks.

### Section 3 — the leftovers

Runs each bit width twice — once with `patch_embed` and `head` left at FP16 (what llama.cpp
does), once fully quantized — and reports what fraction of the file the leftovers occupy.

**Why it matters:** those tensors are 680k parameters, about 3% of the model. At FP32 that's
nothing. **At a 10 MB target they can be a third of your file.** This is the trap FTerViT
identified in prior ternary work.

### Section 4 — the ternary projection

Pure arithmetic, no files written. Prints the four rungs: naive 2-bit (2.110 as FTerViT
ships), base-3 packing (1.600), the uniform floor log₂3 (1.585), and measured entropy at 37.3%
zeros (1.580) — with the two gaps labelled separately so they don't get conflated.

**Note carefully:** the script's **INT2 row is not ternary.** INT2 is 4 levels (−2,−1,0,1) at
2.0 bits. Ternary is 3 levels (−α, 0, +α) at 1.585 bits. Different things. Route B in your plan
is ternary and needs its own quantizer; section 4 is the analytic projection for it.

---

## 9. What this script deliberately does NOT do

| Not done | Why | Where it happens |
|---|---|---|
| Measure accuracy | Needs ImageNet and a GPU | Phase 2 |
| Dequantize and verify reconstruction | `max_abs_weight_err` is the cheap proxy | Phase 2 |
| Use GPTQ / AWQ | Better accuracy, **identical file size** | Phase 2 |
| Real ternary quantization | INT2 ≠ ternary; needs its own path | Route B |
| Structured pruning | Changes the architecture, not just the encoding | Route C |
| Entropy-code anything | Measured headroom is 0.020 bits/weight — dead end | Never |

---

## 10. The questions you should expect, and the answers

**"Why RTN instead of a proper quantizer?"**
Because the claim is about file size, and RTN, GPTQ and AWQ produce byte-identical files. A
better quantizer changes accuracy, not bytes.

**"Why not use `bitsandbytes` or a standard library?"**
Because libraries hide exactly what this audit is measuring — they report a model object, not
a byte count, and their internal layouts differ. Explicit packing makes the scale overhead
visible and verifiable.

**"Is decimal MB or binary MiB correct?"**
Either, consistently. This uses decimal, which matches published model sizes, and says so.

**"How do you know the packing is right?"**
It was tested at 2, 3, 4 and 8 bits for exact byte count and round-trip fidelity, and
`max_abs_weight_err` bounds the quantization error on every run.

**"Your group-wise layout isn't what AWQ does."**
Correct — groups here span the flattened tensor. It affects accuracy, not size. The claim is
about size.

**"Isn't measuring file size trivial?"**
It should be. The field doesn't do it, and the first measurement caught a 45% error in this
script's own serialization. That's the argument.

---

## 11. Changing it

```bash
python src/phase1_audit.py --model deit_tiny_patch16_224   # your Route D baseline
python src/phase1_audit.py --bits 8 6 5 4 3 2              # finer bit sweep
python src/phase1_audit.py --groups 16 32 64 128 256 512   # finer group sweep
python src/phase1_audit.py --budget-mb 5                   # a different target
python src/phase1_audit.py --json audit.json               # machine-readable output
```

Run it on **DeiT-Tiny** as well. It's your Route D baseline (5 MB at INT8, 72.2% accuracy),
and the audit tells you exactly what that baseline weighs — which is the number your compressed
DeiT-Small has to beat.
