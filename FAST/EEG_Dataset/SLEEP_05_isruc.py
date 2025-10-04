import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import h5py
from tqdm import tqdm
import scipy.io as sio
import re
import mne
mne.set_log_level('WARNING')

from share import META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'SLEEP')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'SLEEP')

NAME = 'SLEEP_05_isruc'
GROUPS = ['S1']

# 使用 6 通道子集（按 .mat 实际通道）
CH_NAMES = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']

# 30s epoch，原始200Hz重采样到250Hz
ORIGINAL_RATE = 200  # Original sampling rate from data files
RESAMPLE_RATE = 250  # Target sampling rate after resampling
EPOCH_SECONDS = 30
SEQ_LEN = 20  # 序列长度不再使用，仅保留为可选配置

# 标签映射
LABEL2ID = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}

SLEEP_ISRUC_S1 = META(NAME + '_S1', CH_NAMES, [], ['SLEEP/W', 'SLEEP/N1', 'SLEEP/N2', 'SLEEP/N3', 'SLEEP/REM'], resample_rate=RESAMPLE_RATE, time_length=EPOCH_SECONDS)
SLEEP_ISRUC_S3 = META(NAME + '_S3', CH_NAMES, [], ['SLEEP/W', 'SLEEP/N1', 'SLEEP/N2', 'SLEEP/N3', 'SLEEP/REM'], resample_rate=RESAMPLE_RATE, time_length=EPOCH_SECONDS)


def _list_mat_files(group_root):
    files = []
    if not os.path.isdir(group_root):
        return files
    for fn in os.listdir(group_root):
        if fn.lower().endswith('.mat'):
            files.append(os.path.join(group_root, fn))
    # 排序以确保 subject 序号有序
    def _sort_key(p):
        base = os.path.splitext(os.path.basename(p))[0]
        digits = ''.join([c for c in base if c.isdigit()])
        return int(digits) if digits.isdigit() else base
    return sorted(files, key=_sort_key)


def _extract_subject_id(name):
    digits = ''.join([c for c in name if c.isdigit()])
    return digits if digits else None


def _find_label_txt_for_mat(mat_path):
    base = os.path.splitext(os.path.basename(mat_path))[0]
    sid = _extract_subject_id(base)
    if sid is None:
        return None
    mat_dir = os.path.dirname(mat_path)
    # If under .../S1_Data/mat → go one level up to .../S1_Data
    parent_name = os.path.basename(mat_dir)
    root = os.path.dirname(mat_dir) if parent_name.lower() == 'mat' else mat_dir
    candidates = [
        os.path.join(root, sid, sid, f'{sid}_1.txt'),
        os.path.join(root, sid, sid, f'{sid}_1.TXT'),
        os.path.join(root, sid, sid, f'{sid}_1.csv'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _load_isruc_labels_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    nums = re.findall(r'-?\d+', content)
    if len(nums) == 0:
        return None
    return np.array([int(x) for x in nums], dtype=np.int64)


def detect_sampling_frequency(data_array, epoch_seconds=30):
    """Detect sampling frequency from data array shape"""
    if len(data_array.shape) != 2:
        return None
    
    n_epochs, n_samples = data_array.shape
    implied_fs = n_samples / epoch_seconds
    
    # Common sampling frequencies in sleep studies
    common_fs = [100, 128, 200, 250, 256, 500, 512, 1000]
    
    # Find closest common frequency
    closest_fs = min(common_fs, key=lambda x: abs(x - implied_fs))
    
    if abs(closest_fs - implied_fs) < 1.0:  # Within 1 Hz tolerance
        return float(closest_fs)
    else:
        return float(implied_fs)  # Return calculated frequency

def resample_sleep_data(data, original_fs, target_fs, ch_names_montage):
    """
    Resample sleep EEG data using MNE
    
    Parameters:
    -----------
    data : np.ndarray
        Shape (n_epochs, n_channels, n_samples)
    original_fs : float
        Original sampling frequency
    target_fs : float  
        Target sampling frequency
    ch_names_montage : list
        Channel names for MNE info
        
    Returns:
    --------
    np.ndarray : Resampled data
    """
    if abs(original_fs - target_fs) < 0.1:
        print(f"No resampling needed: {original_fs} Hz ≈ {target_fs} Hz")
        return data
    
    print(f"Resampling from {original_fs} Hz to {target_fs} Hz")
    
    n_epochs, n_channels, n_samples = data.shape
    new_n_samples = int(n_samples * target_fs / original_fs)
    
    # Create MNE info object
    info = mne.create_info(ch_names=ch_names_montage, sfreq=original_fs, ch_types='eeg')
    
    # Resample each epoch
    resampled_data = np.zeros((n_epochs, n_channels, new_n_samples), dtype=data.dtype)
    
    for epoch_idx in range(n_epochs):
        # Create MNE RawArray for this epoch
        epoch_data = data[epoch_idx]  # (n_channels, n_samples)
        raw = mne.io.RawArray(epoch_data, info, verbose=False)
        
        # Resample using MNE
        raw_resampled = raw.resample(sfreq=target_fs, verbose=False)
        
        # Get resampled data
        resampled_data[epoch_idx] = raw_resampled.get_data()
    
    print(f"Resampled shape: {data.shape} -> {resampled_data.shape}")
    print(f"Time samples: {n_samples} -> {new_n_samples}")
    
    return resampled_data

def _normalize_labels(raw_labels):
    labels = np.array(raw_labels).squeeze()
    if labels.dtype.kind in ['U', 'S', 'O']:
        # 字符串标签，按映射表转换
        mapped = []
        for v in labels.flatten():
            s = str(v)
            if s in LABEL2ID:
                mapped.append(LABEL2ID[s])
        labels = np.array(mapped, dtype=np.uint8)
    else:
        labels = labels.astype(np.int64).flatten()
        # 常见编码：{0,1,2,3,5} 或 {1,2,3,4,5}
        uniq = set(labels.tolist())
        if 5 in uniq and 4 not in uniq and 0 in uniq:
            # {0,1,2,3,5} → map 5->4
            labels = np.array([4 if v == 5 else v for v in labels], dtype=np.uint8)
        elif min(uniq) == 1 and max(uniq) == 5:
            # {1..5} → {0..4}
            labels = (labels - 1).astype(np.uint8)
        else:
            labels = labels.astype(np.uint8)
    return labels


def _load_isruc_subject(mat_path):
    try:
        md = sio.loadmat(mat_path, simplify_cells=True)
    except Exception:
        md = sio.loadmat(mat_path)
    
    # 收集各通道矩阵 (epochs, samples)
    chan_arrays = []
    epochs_counts = []
    ch_names_file = ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1']
    
    for ch in ch_names_file:
        if ch not in md:
            return None, None
        arr = md[ch]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            return None, None
        chan_arrays.append(arr)
        epochs_counts.append(arr.shape[0])
    
    if len(chan_arrays) == 0:
        return None, None
    
    # 检测采样频率
    original_fs = detect_sampling_frequency(chan_arrays[0], EPOCH_SECONDS)
    if original_fs is None:
        print(f"Warning: Could not detect sampling frequency for {mat_path}")
        return None, None
    
    print(f"Processing {os.path.basename(mat_path)}: detected {original_fs} Hz")
    
    # 对齐 epoch 数（取最小）
    n_epoch = int(min(epochs_counts))
    chan_arrays = [a[:n_epoch, :].astype(np.float32) for a in chan_arrays]
    
    # 转换为 (N, C, T) 格式
    X = np.stack(chan_arrays, axis=1)  # (N, C, T) 因为每个 a 是 (N, T)
    
    # 应用重采样
    if abs(original_fs - RESAMPLE_RATE) > 0.1:
        X = resample_sleep_data(X, original_fs, RESAMPLE_RATE, CH_NAMES)
    else:
        print(f"No resampling needed: {original_fs} Hz ≈ {RESAMPLE_RATE} Hz")
    
    # 读取标签 txt
    txt_path = _find_label_txt_for_mat(mat_path)
    if txt_path is None:
        return None, None
    y_raw = _load_isruc_labels_txt(txt_path)
    if y_raw is None or y_raw.size == 0:
        return None, None
    y = _normalize_labels(y_raw)[:-30]
    
    # pipeline: clip + robust norm + map to template (75ch)
    X = pipeline(X, CH_NAMES)
    
    return X, y


def proc_group(meta, group_name):
    group_root = os.path.join(SRC_FOLDER, NAME, group_name)
    # 回退候选：S1_Data/mat 或 S3_Data/mat 等
    fallback_roots = [
        os.path.join(SRC_FOLDER, NAME, f'{group_name}_Data', 'mat'),
        os.path.join(SRC_FOLDER, NAME, f'{group_name}_Data'),
        os.path.join(SRC_FOLDER, NAME, group_name, 'mat'),
    ]
    if not os.path.isdir(group_root):
        for r in fallback_roots:
            if os.path.isdir(r):
                group_root = r
                break
    print(group_root)
    # 仅使用 .mat 文件
    mat_files = _list_mat_files(group_root)
    subs = []
    results = []
    for mat_path in tqdm(mat_files, desc=f'ISRUC {group_name} (mat)'):
        X, Y = _load_isruc_subject(mat_path)
        if X is None or Y is None or X.shape[0] == 0:
            print(f"  Warning: skipping {mat_path} due to load failure. X.shape:{X.shape if X is not None else None}, Y.shape:{Y.shape if Y is not None else None}")
            continue
        base = os.path.splitext(os.path.basename(mat_path))[0]
        subs.append(base)
        results.append((base, X, Y))
    meta.subjects = subs
    return results


def proc_all():
    for group in GROUPS:
        res = proc_group(SLEEP_ISRUC_S1 if group == 'S1' else SLEEP_ISRUC_S3, group)
        print(len(res))
        os.makedirs(DATA_FOLDER, exist_ok=True)
        out_h5 = os.path.join(DATA_FOLDER, f'{NAME}_{group}.h5')
        with h5py.File(out_h5, 'w') as f:
            for sub, X, Y in res:
                # X: (trials, channels, sequence) after pipeline (mapped to template)
                # Y: (trials,)
                f.create_dataset(f'{sub}/X', data=X)
                f.create_dataset(f'{sub}/Y', data=Y)
                print(group, sub, X.shape, Y.shape, np.unique(Y, return_counts=True))


if __name__ == '__main__':
    proc_all()



