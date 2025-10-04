import os
import numpy as np
import h5py
import mne
from typing import List, Tuple

# Keep framework style
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'TUH')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'TUH')
# Roots and names (follow project convention)
NAME = 'TUH_04_TUEV_Events'

# TUEV 实际数据路径形如: /home/workspace/dataset/TUH_04_TUEV_Events/edf/{train,eval}/000xxxxx/*.edf
SIGNALS_ROOT = os.path.join(SRC_FOLDER, NAME, 'edf')
OUT_DIR = DATA_FOLDER  # h5 输出目录

# 复用 CHB-MIT 的 16 导双极通道顺序（与 preprocessing_tuev.py 中构造一致）
CH_NAMES_BIPOLAR = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
]

# 将双极名映射到 10-10 模板名，供 pipeline 统一到模板空间
CHANNEL_NAME_MAP = {
    'FP1-F7': 'Fp1',
    'F7-T7': 'FT7',
    'T7-P7': 'TP7',
    'P7-O1': 'PO7',
    'FP2-F8': 'Fp2',
    'F8-T8': 'FT8',
    'T8-P8': 'TP8',
    'P8-O2': 'PO8',
    'FP1-F3': 'F3',
    'F3-C3': 'C3',
    'C3-P3': 'CP3',
    'P3-O1': 'PO3',
    'FP2-F4': 'F4',
    'F4-C4': 'C4',
    'C4-P4': 'CP4',
    'P4-O2': 'PO4',
}
CH_NAMES_MAPPED = [CHANNEL_NAME_MAP[k] for k in CH_NAMES_BIPOLAR]

# 采样率 250 Hz，5 秒窗（事件前后各 2 秒 + 事件本身约 1 秒，截断至 5s）
RESAMPLE_RATE = 250
TIME_LENGTH = 5


def _convert_signals_to_bipolar(signals: np.ndarray, raw: mne.io.BaseRaw) -> np.ndarray:
    name_to_idx = {k: v for (k, v) in zip(raw.info['ch_names'], list(range(len(raw.info['ch_names']))))}
    # 参照 preprocessing_tuev.py 的双极构造
    stack = np.vstack(
        (
            signals[name_to_idx['EEG FP1-REF']] - signals[name_to_idx['EEG F7-REF']],
            signals[name_to_idx['EEG F7-REF']] - signals[name_to_idx['EEG T3-REF']],
            signals[name_to_idx['EEG T3-REF']] - signals[name_to_idx['EEG T5-REF']],
            signals[name_to_idx['EEG T5-REF']] - signals[name_to_idx['EEG O1-REF']],
            signals[name_to_idx['EEG FP2-REF']] - signals[name_to_idx['EEG F8-REF']],
            signals[name_to_idx['EEG F8-REF']] - signals[name_to_idx['EEG T4-REF']],
            signals[name_to_idx['EEG T4-REF']] - signals[name_to_idx['EEG T6-REF']],
            signals[name_to_idx['EEG T6-REF']] - signals[name_to_idx['EEG O2-REF']],
            signals[name_to_idx['EEG FP1-REF']] - signals[name_to_idx['EEG F3-REF']],
            signals[name_to_idx['EEG F3-REF']] - signals[name_to_idx['EEG C3-REF']],
            signals[name_to_idx['EEG C3-REF']] - signals[name_to_idx['EEG P3-REF']],
            signals[name_to_idx['EEG P3-REF']] - signals[name_to_idx['EEG O1-REF']],
            signals[name_to_idx['EEG FP2-REF']] - signals[name_to_idx['EEG F4-REF']],
            signals[name_to_idx['EEG F4-REF']] - signals[name_to_idx['EEG C4-REF']],
            signals[name_to_idx['EEG C4-REF']] - signals[name_to_idx['EEG P4-REF']],
            signals[name_to_idx['EEG P4-REF']] - signals[name_to_idx['EEG O2-REF']],
        )
    )
    return stack


def _build_events(signals_bipolar: np.ndarray, times: np.ndarray, event_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_events = event_data.shape[0]
    fs = float(RESAMPLE_RATE)
    num_chan, _ = signals_bipolar.shape
    feat = np.zeros([num_events, num_chan, int(fs) * TIME_LENGTH], dtype=np.float32)
    labels = np.zeros([num_events], dtype=np.uint8)

    offset = signals_bipolar.shape[1]
    pad = np.concatenate([signals_bipolar, signals_bipolar, signals_bipolar], axis=1)

    # 每个事件中心段：start..end 两侧各扩展 2 秒
    for i in range(num_events):
        start = np.where(times >= event_data[i, 1])[0][0]
        end = np.where(times >= event_data[i, 2])[0][0]
        left = int(2 * fs)
        right = int(2 * fs)
        s = offset + start - left
        e = offset + end + right
        seg = pad[:, s:e]
        if seg.shape[1] < int(fs) * TIME_LENGTH:
            # 长度不足则跳过
            continue
        seg = seg[:, : int(fs) * TIME_LENGTH]
        feat[i, :, :] = seg
        # 将标签改为从 0 开始编码（假设原始标签从 1 开始）
        if event_data.shape[1] > 3:
            lab = int(event_data[i, 3])
            labels[i] = lab - 1 if lab > 0 else 0
        else:
            labels[i] = 0
    # 可能有全零行（由于长度不足被 continue 后仍预分配），过滤掉
    valid = ~(np.all(feat == 0, axis=(1, 2)))
    return feat[valid], labels[valid]


def _proc_subject(sub_dir: str) -> Tuple[str, np.ndarray, np.ndarray]:
    sub = os.path.basename(sub_dir.rstrip('/'))
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for root, _, files in os.walk(sub_dir):
        for f in files:
            if not f.lower().endswith('.edf'):
                continue
            edf_path = os.path.join(root, f)
            rec_path = edf_path[:-4] + '.rec'
            if not os.path.exists(rec_path):
                continue
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
                raw.resample(RESAMPLE_RATE, verbose=False)
                raw.filter(l_freq=0.3, h_freq=75.0, verbose=False)
                raw.notch_filter(60.0, verbose=False)
                signals = raw.get_data(units='uV')
                _, times = raw[:]
                event = np.genfromtxt(rec_path, delimiter=',')
                if event.ndim == 1 and event.size > 0:
                    event = np.expand_dims(event, 0)
                if event.size == 0:
                    continue
                # 构造 16 双极
                sig_bi = _convert_signals_to_bipolar(signals, raw)
                # 取事件段
                feat, y = _build_events(sig_bi, times, event)
                if len(feat) == 0:
                    continue
                X_list.append(feat.astype(np.float32))
                Y_list.append(y.astype(np.uint8))
            except (ValueError, KeyError, IndexError, OSError):
                # 跳过异常文件
                continue

    if len(X_list) == 0:
        raise ValueError(f"No valid data found for subject {sub}")
        return sub, np.empty((0, len(CH_NAMES_MAPPED), RESAMPLE_RATE * TIME_LENGTH), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    # 映射到模板空间（保持项目一致风格）
    X = pipeline(X, CH_NAMES_MAPPED)
    return sub, X, Y


def _list_subjects() -> Tuple[List[str], List[str]]:
    train_root = os.path.join(SIGNALS_ROOT, 'train')
    eval_root = os.path.join(SIGNALS_ROOT, 'eval')
    train_subs = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    eval_subs = [d for d in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, d))]

    def _stable_numeric_sort(names: List[str]) -> List[str]:
        # 数字型 ID 优先按整数排序；否则退回字典序，保证稳定
        if all(n.isdigit() for n in names):
            return sorted(names, key=lambda s: int(s))
        return sorted(names)

    train_subs = _stable_numeric_sort(train_subs)
    eval_subs = _stable_numeric_sort(eval_subs)
    return train_subs, eval_subs


# 注册 META（供训练时查询形状/速率）；注意：实际写入包含 X 与 Y
_TRAIN_SUBS, _EVAL_SUBS = _list_subjects() if os.path.isdir(SIGNALS_ROOT) else ([], [])
if len(_TRAIN_SUBS) > 0 or len(_EVAL_SUBS) > 0:
    split_idx = int(0.8 * len(_TRAIN_SUBS))
    _TRAIN_HEAD = [f'1_{s}' for s in _TRAIN_SUBS[:split_idx]]
    _VAL_TAIL = [f'2_{s}' for s in _TRAIN_SUBS[split_idx:]]
    _TEST_SUBS = [f'3_{s}' for s in _EVAL_SUBS]
    SUBJECTS = _TRAIN_HEAD + _VAL_TAIL + _TEST_SUBS
else:
    SUBJECTS = []
TEXT_LABELS = ['TUEV/SPSW', 'TUEV/GPED', 'TUEV/PLED', 'TUEV/EYEM', 'TUEV/ARTF', 'TUEV/BCKG']  # 占位，真实标签整型写入 Y

TUH_TUEV_Events = META(NAME, CH_NAMES_MAPPED, SUBJECTS, TEXT_LABELS, resample_rate=RESAMPLE_RATE, time_length=TIME_LENGTH)


def proc_all() -> str:
    train_root = os.path.join(SIGNALS_ROOT, 'train')
    eval_root = os.path.join(SIGNALS_ROOT, 'eval')
    train_subs, eval_subs = _list_subjects()

    # 80/20 切分 train → train/val
    split_idx = int(0.8 * len(train_subs))
    train_head = train_subs[:split_idx]
    val_tail = train_subs[split_idx:]

    results: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for sub in train_head:
        results.append(_proc_subject(os.path.join(train_root, sub)))
    for sub in val_tail:
        results.append(_proc_subject(os.path.join(train_root, sub)))
    for sub in eval_subs:
        results.append(_proc_subject(os.path.join(eval_root, sub)))

    os.makedirs(OUT_DIR, exist_ok=True)
    h5_path = os.path.join(OUT_DIR, f'{NAME}.h5')
    with h5py.File(h5_path, 'w') as f:
        for sub, X, Y in results:
            if X.shape[0] == 0:
                continue
            # 前缀强制排序：train=1_, val=2_, test=3_
            if sub in train_head:
                key = f'1_{sub}'
            elif sub in val_tail:
                key = f'2_{sub}'
            else:
                key = f'3_{sub}'
            f.create_dataset(f'{key}/X', data=X)
            f.create_dataset(f'{key}/Y', data=Y)
            print(f"{key}: {X.shape}, {Y.shape}, {np.unique(Y, return_counts=True)}")
    return h5_path


if __name__ == '__main__':
    path = proc_all()
    print('Saved to', path)
