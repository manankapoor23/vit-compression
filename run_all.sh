#!/usr/bin/env bash
# One command: reproduce the main table and every reported number from a fresh clone.
#
#   bash run_all.sh
#
# What it does (CPU only, no dataset, no GPU):
#   1. Creates .venv and installs requirements.txt            (first run only, ~3-5 min)
#   2. Downloads the 21 model files if evidence/models/ has none, and checks each SHA-256
#   3. Runs reproduce.sh: re-derives every statistic and table from the saved per-image
#      results and model files, and checks each against the archived outputs (~2 min)
#   4. Prints the main table (Table 1 of the report)
#
# Training and evaluation needed an H100 and ImageNet; they are not re-run here.
# Their logs (with seeds) are in evidence/logs/. See README.md, "Hardware and runtime".
set -euo pipefail
cd "$(dirname "$0")"

MODELS_URL="${MODELS_URL:-https://github.com/manankapoor23/vit-compression/releases/download/models-v1}"

echo "== setup: Python environment =="
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi
.venv/bin/python -c "import numpy, scipy, torch, timm; print('   numpy', numpy.__version__, '| scipy', scipy.__version__, '| torch', torch.__version__, '| timm', timm.__version__)"

echo "== setup: model files =="
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
cd evidence/models
while read -r _ name; do
  if [ ! -f "$name" ]; then
    echo "   downloading $name"
    curl -fsSL -o "$name" "$MODELS_URL/$name"
  fi
done < SHA256SUMS
if sha256 -c SHA256SUMS --quiet; then
  echo "   21/21 model files present, checksums match"
else
  echo "   checksum mismatch: delete the listed files and re-run"; exit 1
fi
cd ../..

echo
PY=.venv/bin/python ./reproduce.sh

echo
echo "== main table (report Table 1) =="
.venv/bin/python src/main_table.py
