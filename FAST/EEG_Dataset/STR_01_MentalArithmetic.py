import os
import re
import mne
import h5py
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple

# 保持项目风格
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline

DATA_FOLDER = os.path.join(DATA_FOLDER, 'STR')
# 基本信息
NAME = 'STR_01_MentalArithmetic'

# 数据根目录（用户已展示列表）
ROOT_DIR = os.path.join(
    SRC_FOLDER,
    'STR',
    'STR_01_MentalArithmetic',
    'eeg-during-mental-arithmetic-tasks-1.0.0'
)

# 通道选择：该数据集多为标准10-20/10-10名称；尽量使用通用集合
CH_NAMES = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
]

# 采样率与窗口长度
RESAMPLE_RATE = 250
TIME_LENGTH = 4  # 秒

# 二分类：心算任务与静息，若无法从文件名区分，则统一按0，占位
CLASSES = ['STR/Control', 'STR/MentalArithmetic']

STR_MentalArithmetic = META(NAME, CH_NAMES, [], CLASSES, resample_rate=RESAMPLE_RATE, time_length=TIME_LENGTH)


def _list_edf_files(root: str) -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Data root not found: {root}')
    for f in os.listdir(root):
        if f.lower().endswith('.edf'):
            paths.append(os.path.join(root, f))
    paths.sort()
    return paths


def _infer_subject_and_label(filename: str) -> Tuple[str, int]:
    base = os.path.basename(filename)
    # 示例: Subject02_1.edf / Subject02_2.edf
    m = re.match(r'(?i)subject(\d+)_([12])\.edf$', base)
    if m:
        sub = f'Subject{int(m.group(1)):02d}'
        sess = int(m.group(2))
        # 简单规则：_1 记为静息(0)，_2 记为任务(1)。若数据集含义相反可在此处调换
        label = 0 if sess == 1 else 1
        return sub, label
    # 回退：按文件名前缀提取编号
    m = re.match(r'(?i)subject(\d+).*\.edf$', base)
    sub = f'Subject{int(m.group(1)) :02d}' if m else os.path.splitext(base)[0]
    label = 0
    return sub, label


def _load_and_segment_one(file_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
    sub, label = _infer_subject_and_label(file_path)

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')

    # 通道名规范化：统一大小写并替换T3/T4/T5/T6等到10-10近似名
    rename1 = {}
    for ch in raw.ch_names:
        cname = ch.strip()
        if cname.startswith('EEG '):
            cname = cname[4:]
        if cname.endswith('-LE') or cname.endswith('-REF'):
            cname = cname.rsplit('-', 1)[0]
        cname = cname.replace('Fpz','Fpz').replace('FpZ','Fpz')
        cname = cname.replace('Cz','Cz').replace('Pz','Pz').replace('Fz','Fz')
        rename1[ch] = cname
    raw.rename_channels(rename1)

    # 10-20到10-10近似映射
    mapping_1020_to_1010 = {'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
    rename2 = {c: mapping_1020_to_1010.get(c, c) for c in raw.ch_names}
    raw.rename_channels(rename2)

    # 选择需要的通道；缺失通道将被丢弃
    avail = [c for c in CH_NAMES if c in raw.ch_names]
    if len(avail) == 0:
        return sub, np.zeros((0, len(CH_NAMES), TIME_LENGTH * RESAMPLE_RATE), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    raw.pick_channels(avail, ordered=True)

    # 重采样与滤波
    if raw.info['sfreq'] != RESAMPLE_RATE:
        raw.resample(RESAMPLE_RATE)
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)
    raw.notch_filter(50, verbose=False)

    data = raw.get_data().T  # (T, C)
    points, chs = data.shape
    seg_len = TIME_LENGTH * RESAMPLE_RATE
    rem = points % seg_len
    if rem:
        data = data[:-rem, :]
        points -= rem
    if points == 0:
        return sub, np.zeros((0, chs, seg_len), dtype=np.float32), np.zeros((0,), dtype=np.uint8)

    data = data.reshape(-1, seg_len, chs)  # (N, L, C)
    data = np.transpose(data, (0, 2, 1)).astype(np.float32)  # (N, C, L)

    # pipeline 需要传入与 CH_NAMES 对齐的通道顺序。当前仅保留 avail 顺序，需映射到模板
    X = pipeline(data, avail)  # (N, 75, L) after map_to_template
    Y = np.full((X.shape[0],), label, dtype=np.uint8)

    return sub, X, Y


def _merge_subjects(items: List[Tuple[str, np.ndarray, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    merged: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for sub, X, Y in items:
        if X is None or Y is None or X.size == 0:
            continue
        if sub not in merged:
            merged[sub] = ([], [])
        merged[sub][0].append(X)
        merged[sub][1].append(Y)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sub, (xs, ys) in merged.items():
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        out[sub] = (X, Y)
    return out


def proc_all():
    files = _list_edf_files(ROOT_DIR)
    if len(files) == 0:
        raise FileNotFoundError(f'No EDF found under {ROOT_DIR}')

    subjects_src = sorted({ _infer_subject_and_label(p)[0] for p in files }, key=lambda s: int(re.search(r'(\d+)', s).group(1)) if re.search(r'(\d+)', s) else 1e9)
    # 三位数字键，从001开始，长度与subjects_src一致
    subjects_keys = [f'{i+1:03d}' for i in range(len(subjects_src))]
    sub_map = {src: key for src, key in zip(subjects_src, subjects_keys)}
    STR_MentalArithmetic.subjects = subjects_keys

    with mp.Pool(min(len(files), os.cpu_count() or 4)) as pool:
        results = pool.map(_load_and_segment_one, files)

    merged = _merge_subjects(results)

    os.makedirs(DATA_FOLDER, exist_ok=True)
    h5_path = os.path.join(DATA_FOLDER, f'{NAME}.h5')
    with h5py.File(h5_path, 'w') as f:
        for src in subjects_src:
            if src not in merged:
                continue
            key = sub_map[src]
            X, Y = merged[src]
            f.create_dataset(f'{key}/X', data=X)
            f.create_dataset(f'{key}/Y', data=Y)
            print(key, X.shape, Y.shape, np.unique(Y, return_counts=True))


if __name__ == '__main__':
    proc_all()



