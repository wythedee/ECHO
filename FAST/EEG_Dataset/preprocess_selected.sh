#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# 确保可以用包名导入 EEG_Dataset 下的模块
export PYTHONPATH="${PYTHONPATH:-}:$SCRIPT_DIR/.."
PROFILE="${PREPROCESS_PROFILE:-manual}"

# 默认使用优先级映射（不平均），如需切换回均值可 export FAST_CHANNEL_STRATEGY=mean
export FAST_CHANNEL_STRATEGY="${FAST_CHANNEL_STRATEGY:-priority}"
# 输出根目录默认和 share.py 保持一致，可通过 FAST_EEG_OUTPUT 覆盖
export FAST_EEG_OUTPUT="${FAST_EEG_OUTPUT:-${ECHO_DATASET_ROOT:-/path/to/EEG_Standardized_Group}}"

DATASETS=("$@")
if ((${#DATASETS[@]} == 0)); then
  case "$PROFILE" in
    main)
      DATASETS=(
        "MI_01_SSVEP_KoreaU.py"
        "MI_03_Shin2017A.py"
        "MI_04_BCI_IV_2a.py"
        "MI_05_Weibo2014.py"
        "MI_06_Schirrmeister2017.py"
        "MI_07_Cho2017.py"
        "MI_09_Track4_Upper_limb.py"
        "MI_10_HeBin2021.py"
        "MI_11_HeBin2024.py"
        "MI_12_PhysioNet.py"
        "EMO_02_SEED_IV.py"
        "EMO_03_SEED_V.py"
        "EMO_04_SEED.py"
        "EMO_05_THU_EP.py"
      )
      ;;
    isruc)
      DATASETS=("SLEEP_05_isruc.py")
      ;;
    manual)
      DATASETS=()
      ;;
    *)
      echo "[error] Unknown PREPROCESS_PROFILE=$PROFILE (expected: main, isruc, manual)." >&2
      exit 1
      ;;
  esac
fi

if ((${#DATASETS[@]} == 0)); then
  echo "[error] No dataset scripts selected. Use PREPROCESS_PROFILE=main|isruc or pass script names as arguments." >&2
  exit 1
fi

for ds in "${DATASETS[@]}"; do
  if [[ -f "$ds" ]]; then
    echo "[RUN] python $ds"
    python "$ds"
  else
    echo "[SKIP] $ds does not exist; please check the file name." >&2
  fi
done
