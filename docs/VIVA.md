# Viva pack — 30 questions, short answers, where to point

Read one section a day out loud without looking. Every number here is in `results.md` and is rebuilt by
`./reproduce.sh` (about 90 s). "Show me" answers: open the notebook or file named in brackets.

## A. The project in one breath

**1. What is your project, in one sentence?**
I compressed DeiT-Small four different ways to the same real file size (~9.8 MB) with the same training
effort, and showed that the four models fail on different images and that the best model on clean photos is
not the best on damaged photos.

**2. What does P49 ask, and what did you deliver?**
A ViT under 10 MB with stated accuracy loss, and a Pareto table of accuracy × size × CPU latency.
Delivered: route A, 9.718 MB, 76.50%, a 3.22-point loss against 79.72% measured in my own pipeline; the
Pareto table on two CPUs (`report/main.pdf`, Table 1).

**3. Where do you deviate from P49, and why?**
ImageNet-1k instead of ImageNet-100 (harder, contains it). MLP-channel pruning instead of token/head pruning
(token pruning saves 0 bytes; MLP holds 64% of the weights, so channels save the most bytes per cut).
No single model uses all three methods: each route combines at most two, so each comparison changes one thing.

**4. What is new? Didn't someone already compress a ViT below 10 MB?**
Yes, FTerViT reaches 5.81 MB. I don't claim a smaller model. What is new is a fair comparison of different
methods at the same *measured* size and equal effort, with luck measured (seed floor).

## B. File size

**5. Why not just use weights × bits ÷ 8?**
A real file also stores the scales and keeps some layers at 16 bits. At 2 bits the real DeiT-Small file is
29.1% bigger than the formula, DeiT-Tiny 53.5% (`notebooks/01_file_size.ipynb`).

**6. Is the formula wrong in published papers then?**
Not in general. With one scale per channel, as most vision papers use, it adds only 2–5%, a fair shortcut.
It fails under small groups and small models, which a hard 10 MB limit forces. (I withdrew a stronger claim.)

**7. How do you know your sizes are right?**
Two derivations: measured with `os.path.getsize()` and predicted from layer shapes
(`src/arch_audit_summary.py:25`, `predicted_bytes`). For all 80 files they differ only by the header (≤14.6 KB).

**8. What is a scale and a group?**
Weights are split into groups (route A: 36 weights). Each group stores one 16-bit scale; each weight stores a
small whole number. Weight ≈ code × scale. Example: 0.12, −0.40, 0.05 → codes 1, −3, 0 × 0.1333
(`src/phase1_audit.py:95`, `quantize_groupwise`).

## C. The routes and the code

**9. Walk me through the four routes.**
A: 3-bit, 36 weights per scale, 9.718 MB. B: ternary, 15 sensitive layers at 8 bits, 9.785 MB.
C: 4-bit + 374 of 1,536 MLP channels pruned, 9.796 MB. D: DeiT-Tiny distilled from DeiT-Small, 8-bit with
38 layers at 16 bits, 9.765 MB. Spread 0.80%.

**10. How did you make them the same size?**
Each route has one setting. A binary search writes the real file, measures it, and narrows the range until
the file lands just under the target. Accuracy is never used (`src/build_routes.py:172`, `build_route`).

**11. What is ternary and how do you store it?**
Each weight becomes −1, 0 or +1 with one scale per row; threshold 0.7 × mean |w| (TWN rule). Five values fit
in one byte because 3⁵ = 243 ≤ 256, so 1.6 bits each (`src/build_routes.py:38`, `:57`).

**12. How does your pruning work?**
Structured: whole MLP channels with the smallest weights are removed, so the matrix really gets smaller
(`src/build_routes.py:72`, `prune_mlp_channels`). Unstructured pruning zeroes single weights and does not shrink
a file without extra formats.

**13. Which layers are "sensitive" in route B?**
Ranked by weight variance, a cheap proxy (`src/build_routes.py:141`). It is a stated limitation.

**14. How do you train a model with rounded weights?**
Quantisation-aware training with a straight-through estimator. `FakeQuant` (`src/train_routes.py:69`) uses the
rounded weight in the forward pass and passes the gradient straight through in the backward pass, so the
full-precision weights keep learning.

**15. What was the training recipe?**
3000 steps, batch 256, AdamW, lr 1e-4 cosine, weight decay 0.05, label smoothing 0.1, ~20% of ImageNet
(200 of 1,024 shards). Route D also uses distillation, temperature 2, equal weight with the label loss.
Identical for every route; 3 seeds each (`src/train_routes.py:249`).

**16. Did you pick the best checkpoint on the validation set?**
No. The exported model is always the step-3000 model; validation is only logged.

## D. Testing and statistics

**17. What is ImageNet-C and why only 14 corruptions?**
Artificially damaged photos: noise, blur, weather, digital. I used 14 of 15 at severity 3 on 5,000 images.
Glass blur was excluded: 865 s per 1,000 images, about 20× the others.

**18. What is a seed, and what is the seed floor?**
A seed fixes the random choices in training. Two runs of the same route already disagree on some images:
8.49% (A), 9.55% (B), 7.72% (C), 6.31% (D). That is the seed floor. Two routes count as different only if
they disagree clearly more (`notebooks/03_results.ipynb`).

**19. Why McNemar's test?**
Every model saw the same images, so the data are paired. McNemar looks only at images where exactly one model
is right and asks whether one side wins too often to be chance (`src/offline_analysis.py:76`).

**20. Why 5,000 and not 70,000 samples for damaged images?**
The 14 corruptions are applied to the same 5,000 photos, so they are not independent. Counting 70,000 would
inflate certainty by about √14 ≈ 3.7×.

**21. What is Bonferroni?**
When you run many tests, one passes by luck more easily, so each must pass 0.05 ÷ (number of tests).
For the six B-vs-D tests that is p < 0.0083; all pass.

## E. Results

**22. What is the headline result?**
Routes of the same size fail on different images: every pair disagrees 1.2–2.0× more than two seeds of one
route. And the clean ranking reverses: ternary beats the distilled model on clean images in all 3 seeds
(+0.9 to +1.1), the distilled model wins on damaged images in all 3 (+3.7 to +3.9).

**23. Isn't the reversal just because D is a different network?**
Yes, mostly, and I measured it: uncompressed DeiT-Tiny already beats ternary by +5.51 on damaged images,
route D by +3.81. Distillation and 8-bit *lowered* Tiny's damaged accuracy by 1.70. So it is a deployment
result, not a distillation effect (`src/robustness_source.py`).

**24. Is ternary specially weak on noise?**
Consistent but not proven. It sits 4.4–5.0 points below the trend line of seven same-shape models in every
seed and passes the 95% rule I fixed in advance, but fails a Bonferroni-strict 98.3% range in 2 of 3 seeds
and a plain-axis fit (`src/o7_trend.py`).

**25. What does calibration show?**
D is worst calibrated before correction (ECE ~12%) and best after one-number temperature scaling (~3%):
its error was a uniform offset (`src/rescore.py:162`, `fit_temperature`).

**26. Is a smaller file faster?**
No. A and B are 9× smaller but run at the original's speed (no CPU kernel for 3-bit/ternary; they are
expanded at load). Pruning gains ~10%. Only D is much faster: 2.23× on M2, 2.38× on Intel.

## F. Honesty and process

**27. What went wrong?**
Ten pipeline bugs, recorded in `results.md`, e.g. a file pattern that silently matched zero training files,
and a tensor aliasing bug that cost route B 3.9 points. Two were visible only because size was measured.
One prediction was wrong (seed floor 3–6%; real 6.3–9.6%).

**28. How do I know the .bin files are the models you report?**
Every file decodes and re-encodes byte-identically (21/21), and re-scoring every model from its file on a
different GPU matches 99.71% of 600,000 per-image answers (`./reproduce.sh`, steps 5–6).

**29. What would you do with more time?**
A fifth model using all three methods (distil → prune → quantise); intermediate stages for B and C;
longer training, since ternary was still improving fastest at step 3000.

**30. Did you use AI?**
Yes, declared in the report: Claude was used for code, writing and checking. I reviewed and ran all code,
and every number is rebuilt by `reproduce.sh`. (Fill the declaration's last line yourself, honestly.)
