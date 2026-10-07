#!/usr/bin/env bash
# Run one phase end to end on the GPU machine, resumable: finished steps are skipped.
# Usage (inside the project dir):  nohup bash scripts/run_phase.sh image > logs/image.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.hf_home" UV_CACHE_DIR="$PWD/.uv_cache" TMPDIR="$PWD/.tmp" PYTHONUNBUFFERED=1
mkdir -p logs features models "$TMPDIR"
PY=.venv/bin/python

step() { [ -f "$1" ] && echo "[skip] $1 exists" || { echo "[run] $*"; shift; "$@"; }; }

case "${1:-image}" in
  deps)
    grep -v -E "^torch" requirements.txt > "$TMPDIR/req.txt"
    uv pip install --python $PY -r "$TMPDIR/req.txt"
    ;;
  image)
    step data/images/train/manifest.csv $PY -m scripts.download_openfake --split train --per-label 3000
    step data/images/test/manifest.csv  $PY -m scripts.download_openfake --split test  --per-label 500
    step features/img_train.npz $PY -m scripts.extract_image_features --root data/images/train --out features/img_train.npz --aug 2
    step features/img_test.npz  $PY -m scripts.extract_image_features --root data/images/test  --out features/img_test.npz  --aug 1
    $PY -m scripts.train_head --train features/img_train.npz --test features/img_test.npz --out models/image_head.joblib
    ;;
  text)
    step features/txt_train.npz $PY -m scripts.extract_text_features --per-label 2000 --out features/txt_train.npz
    step features/txt_test.npz  $PY -m scripts.extract_text_features --per-label 400 --seed 1 --skip 200000 --out features/txt_test.npz
    $PY -m scripts.train_head --scale --train features/txt_train.npz --test features/txt_test.npz --out models/text_head.joblib
    ;;
esac
echo "[done] $1"
