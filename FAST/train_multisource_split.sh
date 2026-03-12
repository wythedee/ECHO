#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROFILE="${PRETRAIN_PROFILE:-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0}"
HEAD="${HEAD:-V3}"
DIM1="${DIM1:-96}"
DIM2="${DIM2:-96}"
WIN="${WIN:-100}"
STEP="${STEP:-90}"
LAYERS="${LAYERS:-4}"
LR="${LR:-5e-05}"
WORKERS="${WORKERS:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

case "$PROFILE" in
  main)
    DS="${DS:-MI_01_KoreaU,MI_03_Shin2017A,MI_04_BCI_IV_2a,MI_05_Weibo2014,MI_06_Schirrmeister2017,MI_07_Cho2017,MI_09_Track4_Upper_limb,MI_10_HeBin2021_LR,MI_10_HeBin2021_UD,MI_11_HeBin2024_LR,MI_11_HeBin2024_UD,MI_12_PhysioNet,EMO_02_SEED_IV,EMO_03_SEED_V,EMO_04_SEED,EMO_05_THU-EP}"
    TIME_LENGTH="${TIME_LENGTH:-10}"
    ;;
  isruc)
    DS="${DS:-SLEEP_05_isruc_S3}"
    TIME_LENGTH="${TIME_LENGTH:-30}"
    ;;
  *)
    echo "[error] Unknown PRETRAIN_PROFILE=$PROFILE (expected: main or isruc)." >&2
    exit 1
    ;;
esac

cmd=(
  "$PYTHON_BIN" -u train_multisource_split.py
  --ds "$DS"
  --time_length "$TIME_LENGTH"
  --gpus "$GPUS"
  --head "$HEAD"
  --dim1 "$DIM1"
  --dim2 "$DIM2"
  --win "$WIN"
  --step "$STEP"
  --lay "$LAYERS"
  --lr "$LR"
  --bs "$BATCH_SIZE"
  --workers "$WORKERS"
)

if [[ "$PERSISTENT_WORKERS" == "1" ]]; then
  cmd+=(--persistent_workers)
fi

printf '[run] %s\n' "${cmd[*]}"
"${cmd[@]}"
