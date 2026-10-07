#!/usr/bin/env bash
# Run one phase end to end, resumable: finished steps are skipped.
#   remote GPU box:  nohup bash scripts/run_phase.sh image > logs/image.log 2>&1 &
#   Kaggle:          PY=python IMG_TRAIN=3000 IMG_TEST=500 bash scripts/run_phase.sh image
# Sizes are env-configurable: small defaults for slow networks, larger on Kaggle.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.hf_home}" UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv_cache}" TMPDIR="${TMPDIR_BTP:-$PWD/.tmp}" PYTHONUNBUFFERED=1
mkdir -p logs features models "$TMPDIR"
PY="${PY:-.venv/bin/python}"
IMG_TRAIN="${IMG_TRAIN:-1500}"; IMG_TEST="${IMG_TEST:-300}"
TXT_TRAIN="${TXT_TRAIN:-2000}"; TXT_TEST="${TXT_TEST:-400}"

step() { [ -f "$1" ] && echo "[skip] $1 exists" || { echo "[run] $*"; shift; "$@"; }; }

case "${1:-image}" in
  deps)
    grep -v -E "^torch" requirements.txt > "$TMPDIR/req.txt"
    if [ "$PY" = "python" ]; then $PY -m pip install -q -r "$TMPDIR/req.txt"; else uv pip install --python $PY -r "$TMPDIR/req.txt"; fi
    ;;
  image)
    step data/images/train/manifest.csv $PY -m scripts.download_openfake --split train --per-label "$IMG_TRAIN"
    step data/images/test/manifest.csv  $PY -m scripts.download_openfake --split test  --per-label "$IMG_TEST"
    step features/img_train.npz $PY -m scripts.extract_image_features --root data/images/train --out features/img_train.npz --aug 2
    step features/img_test.npz  $PY -m scripts.extract_image_features --root data/images/test  --out features/img_test.npz  --aug 1
    $PY -m scripts.train_head --train features/img_train.npz --test features/img_test.npz --out models/image_head.joblib
    ;;
  text)
    step features/txt_train.npz $PY -m scripts.extract_text_features --per-label "$TXT_TRAIN" --out features/txt_train.npz
    step features/txt_test.npz  $PY -m scripts.extract_text_features --per-label "$TXT_TEST" --seed 1 --skip 200000 --out features/txt_test.npz
    $PY -m scripts.train_head --scale --train features/txt_train.npz --test features/txt_test.npz --out models/text_head.joblib
    ;;
esac
echo "[done] $1"
