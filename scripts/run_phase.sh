#!/usr/bin/env bash
# Run one phase end to end, resumable: finished steps are skipped.
#   remote GPU box:  nohup bash scripts/run_phase.sh image > logs/image.log 2>&1 &
#   Kaggle:          PY=python IMG_TRAIN=3000 IMG_TEST=500 bash scripts/run_phase.sh image
# Sizes are env-configurable: small defaults for slow networks, larger on Kaggle.
set -euo pipefail
cd "$(dirname "$0")/.."
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_home}" UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv_cache}" TMPDIR="${TMPDIR_BTP:-$PWD/.tmp}" PYTHONUNBUFFERED=1
mkdir -p logs features models "$TMPDIR"
PY="${PY:-.venv/bin/python}"
IMG_TRAIN="${IMG_TRAIN:-1500}"; IMG_TEST="${IMG_TEST:-300}"
SID_TRAIN="${SID_TRAIN:-1000}"; SID_TEST="${SID_TEST:-300}"
TXT_TRAIN="${TXT_TRAIN:-2000}"; TXT_TEST="${TXT_TEST:-400}"

step() { [ -f "$1" ] && echo "[skip] $1 exists" || { echo "[run] $*"; shift; "$@"; }; }

case "${1:-image}" in
  deps)
    grep -v -E "^torch" requirements.txt > "$TMPDIR/req.txt"
    if [ "$PY" = "python" ]; then $PY -m pip install -q -r "$TMPDIR/req.txt"; else uv pip install --python $PY -r "$TMPDIR/req.txt"; fi
    ;;
  image)
    # training images are only needed while their features or the edit localizer are still missing (resume-friendly)
    if [ ! -f features/img_train.npz ] || [ ! -f models/tamper_head.pt ]; then
      step data/images/train/manifest.csv $PY -m scripts.download_openfake --split train --per-label "$IMG_TRAIN"
    fi
    if [ ! -f features/sid_train.npz ] || [ ! -f models/tamper_head.pt ]; then
      step data/sid/train/.done bash -c "$PY -m scripts.download_sidset --split train --per-class $SID_TRAIN && touch data/sid/train/.done"         || echo "[warn] SID-Set train download failed; continuing with OpenFake only"
    fi
    step data/images/test/manifest.csv  $PY -m scripts.download_openfake --split test  --per-label "$IMG_TEST"
    step data/sid/test/.done  bash -c "$PY -m scripts.download_sidset --split validation --out-split test --per-class $SID_TEST && touch data/sid/test/.done"       || echo "[warn] SID-Set test download failed; evaluation falls back to OpenFake + synthetic edits"
    step features/img_train.npz $PY -m scripts.extract_image_features --root data/images/train --out features/img_train.npz --aug 2
    step features/img_test.npz  $PY -m scripts.extract_image_features --root data/images/test  --out features/img_test.npz  --aug 1
    [ -f data/sid/train/.done ] && step features/sid_train.npz $PY -m scripts.extract_image_features --root data/sid/train --out features/sid_train.npz --aug 1
    [ -f data/sid/test/.done ]  && step features/sid_test.npz  $PY -m scripts.extract_image_features --root data/sid/test  --out features/sid_test.npz
    TR="features/img_train.npz"; TE="features/img_test.npz"
    [ -f features/sid_train.npz ] && TR="$TR features/sid_train.npz"
    [ -f features/sid_test.npz ] && TE="$TE features/sid_test.npz"
    $PY -m scripts.train_head --train $TR --test $TE --out models/image_head.joblib
    ;;
  tamper)
    step models/tamper_head.pt $PY -m scripts.train_tamper
    ;;
  audio)
    step data/audio/manifest.csv $PY -m scripts.download_audio
    for s in train test test_unseen; do
      step features/aud_$s.npz $PY -m scripts.extract_audio_features --root data/audio/$s --out features/aud_$s.npz --aug 1
    done
    $PY -m scripts.train_head --train features/aud_train.npz --test features/aud_test.npz features/aud_test_unseen.npz --out models/audio_head.joblib
    ;;
  evaluate)
    $PY -m scripts.evaluate --n-explain "${N_EXPLAIN:-24}"
    ;;
  text)
    step features/txt_train.npz $PY -m scripts.extract_text_features --per-label "$TXT_TRAIN" --out features/txt_train.npz
    step features/txt_test.npz  $PY -m scripts.extract_text_features --per-label "$TXT_TEST" --seed 1 --skip 200000 --out features/txt_test.npz
    $PY -m scripts.train_head --scale --train features/txt_train.npz --test features/txt_test.npz --out models/text_head.joblib
    ;;
esac
echo "[done] $1"
