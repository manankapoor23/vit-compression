#!/usr/bin/env python3
"""Do the model files reproduce the original run? Compares the Kaggle re-score (every model loaded
from its .bin file) with the archived per-image answers on all 50,000 clean photos."""
import json
import numpy as np

R = json.load(open("evidence/rescore/rescore.json"))
changed = total = 0
for s in (0, 1, 2):
    old = json.load(open(f"evidence/logs/evalclean_s{s}.json"))["results"]
    for r in "ABCD":
        a = np.array(old[r]["clean"]["correct"]); b = np.array(R["results"][f"{r}-s{s}"]["clean"]["correct"])
        assert len(a) == len(b) == 50000
        changed += int((a != b).sum()); total += len(a)
print(f"re-score on {R['gpu']} ({R['amp_dtype']}): {changed} of {total:,} per-photo answers changed "
      f"-> {100 * (1 - changed / total):.2f}% match")
