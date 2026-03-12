#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PRETRAIN_PROFILE:-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1}"
ENC_VERSION="${ENC_VERSION:-V3}"

case "$PROFILE" in
  main)
    DS="${DS:-MI_01_KoreaU,MI_03_Shin2017A,MI_04_BCI_IV_2a,MI_05_Weibo2014,MI_06_Schirrmeister2017,MI_07_Cho2017,MI_09_Track4_Upper_limb,MI_10_HeBin2021_LR,MI_10_HeBin2021_UD,MI_11_HeBin2024_LR,MI_11_HeBin2024_UD,MI_12_PhysioNet,EMO_02_SEED_IV,EMO_03_SEED_V,EMO_04_SEED,EMO_05_THU-EP}"
    TIME_LEN="${TIME_LEN:-10}"
    WHISPER_MODEL="${WHISPER_MODEL:-tiny}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    ;;
  isruc)
    DS="${DS:-SLEEP_05_isruc_S3}"
    TIME_LEN="${TIME_LEN:-30}"
    WHISPER_MODEL="${WHISPER_MODEL:-base}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    ;;
  *)
    echo "[error] Unknown PRETRAIN_PROFILE=$PROFILE (expected: main or isruc)." >&2
    exit 1
    ;;
esac

cmd=(
  "$PYTHON_BIN" -u MultiSource_EEG2Text_Split.py
  --ds_name "$DS"
  --gpus "$GPUS"
  --time_len "$TIME_LEN"
  --enc_version "$ENC_VERSION"
  --whisper "$WHISPER_MODEL"
  --bs "$BATCH_SIZE"
)

printf '[run] %s\n' "${cmd[*]}"
"${cmd[@]}"
