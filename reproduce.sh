#!/usr/bin/env bash
# Regenerates every number in results.md that can be regenerated without a GPU,
# from the archived evidence, and checks it against the saved outputs.
#   1. Offline statistics (seed floor, per-corruption, within-architecture, calibration) ~40 s
#   2. Size audit of DeiT-S, rebuilt and compared byte-for-byte to the archived audit   ~30 s
#   3. Five-architecture summary: every file re-derived from tensor shapes              ~1 min
#   4. Main result tables (routes, stages) from the measured files                      ~2 s
#   5. Model files reproduce the original run (Kaggle re-score, per photo)             ~2 s
#   6. O7: ternary against the clean-to-noise trend line                               ~60 s
# Usage:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "== 1/6  offline statistics =="
"$PY" src/offline_analysis.py --out "$TMP/analysis" > "$TMP/stdout.txt"
if diff -q "$TMP/stdout.txt" <(sed "s#evidence/analysis#$TMP/analysis#" evidence/analysis/offline_analysis_stdout.txt) >/dev/null; then
  echo "   MATCH: identical to evidence/analysis/offline_analysis_stdout.txt"
else
  echo "   DIFFERS from the archived output:"; diff "$TMP/stdout.txt" evidence/analysis/offline_analysis_stdout.txt | head -20; exit 1
fi

echo "== 2/6  DeiT-S size audit vs archived pretrained audit =="
"$PY" src/phase1_audit.py --random-init --outdir "$TMP/audit_out" --json "$TMP/deit_s.json" > /dev/null
"$PY" - "$TMP/deit_s.json" <<'PYEOF'
import json, sys
new = {(r["format"], r["group"]): r["bytes"] for r in json.load(open(sys.argv[1]))["sections"]["sizes"]}
old = {(r["format"], r["group"]): r["bytes"] for r in json.load(open("evidence/audit_deit/audit.json"))["sections"]["sizes"]}
q = [k for k in old if k[0].startswith("INT")]
bad = [k for k in q if old[k] != new[k]]
print(f"   {len(q) - len(bad)}/{len(q)} quantized file sizes byte-identical"); sys.exit(1 if bad else 0)
PYEOF

echo "== 3/6  five-architecture summary (re-derived from shapes) =="
"$PY" src/arch_audit_summary.py evidence/audit_multiarch/*.json | tail -8
echo "== 4/6  main result tables (sizes, accuracy, CPU time) =="
"$PY" src/main_table.py > "$TMP/main_table.txt"
if diff -q "$TMP/main_table.txt" evidence/rescore/main_table.txt >/dev/null; then
  cat "$TMP/main_table.txt"; echo "   MATCH: identical to evidence/rescore/main_table.txt"
else
  echo "   DIFFERS:"; diff "$TMP/main_table.txt" evidence/rescore/main_table.txt | head -20; exit 1
fi
echo "== 5/6  model files reproduce the original run =="
"$PY" src/rescore_check.py > "$TMP/rc.txt"
diff -q "$TMP/rc.txt" evidence/rescore/rescore_check.txt >/dev/null && { cat "$TMP/rc.txt"; echo "   MATCH"; } || { echo "   DIFFERS"; exit 1; }

echo "== 6/6  O7 trend-line test =="
cp evidence/rescore/o6_line.json "$TMP/o6_saved.json"
"$PY" src/o7_trend.py > "$TMP/o7.txt"
if diff -q "$TMP/o7.txt" evidence/rescore/o7_trend_stdout.txt >/dev/null && cmp -s "$TMP/o6_saved.json" evidence/rescore/o6_line.json; then
  head -3 "$TMP/o7.txt"; echo "   MATCH: identical to evidence/rescore/o7_trend_stdout.txt"
else
  echo "   DIFFERS"; cp "$TMP/o6_saved.json" evidence/rescore/o6_line.json; exit 1
fi
echo; echo "All checks passed."
