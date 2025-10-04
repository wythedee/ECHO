import os
import glob
import pickle
from scipy import signal
import numpy as np
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline

# NAME 与目录
NAME = 'EMO_06_FACED'
# 默认原始数据根目录（可按需调整为你的 FACED 路径结构）
ROOT_DIR = os.path.join(SRC_FOLDER, 'EMO', 'EMO_06_FACED', 'Processed_data')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'EMO')

# 32 通道（常见 10-10/10-20 32ch 布局，匹配 FACED 32 通道数据）
CH_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',
    'TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10'
]

# FACED 的 trial->label 映射，参照用户给出的示例（每文件 28 个 trial）
FACED_TRIAL_LABELS = np.array([
    0,0,0,
    1,1,1,
    2,2,2,
    3,3,3,
    4,4,4,4,
    5,5,5,
    6,6,6,
    7,7,7,
    8,8,8,
], dtype=np.uint8)

# 9 类标签名占位（未知具体语义时以编号代替）
CLASSES = ['EMO/Amusement', 'EMO/Inspiration', 'EMO/Joy', 'EMO/Tenderness', 'EMO/Anger', 'EMO/Fear', 'EMO/Disgust', 'EMO/Sadness', 'EMO/Neutral']

# FACED 每个样本（切 3 段后）时长：10 个窗口 * 200 点 = 2000 点；
# 若视采样率为 200 Hz，则每样本 10 秒。
RESAMPLE_RATE = 250
TIME_LENGTH = 10

EMO_FACED = META(NAME, CH_NAMES, [], CLASSES, resample_rate=RESAMPLE_RATE, time_length=TIME_LENGTH)


def _list_subject_files():
    """列出 ROOT_DIR 下可用的 pickle 文件，返回绝对路径与 subject 名称（去后缀）。"""
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f'FACED root not found: {ROOT_DIR}')
    files = [fn for fn in os.listdir(ROOT_DIR) if not fn.startswith('.')]
    files = sorted(files)
    fpaths = [os.path.join(ROOT_DIR, fn) for fn in files]
    subjects = [os.path.splitext(fn)[0] for fn in files]
    return fpaths, subjects


def _load_faced_array(pkl_path):
    """读取 FACED 的 pickle，返回 numpy 数组，期望形状 (trial, channel, time)。"""
    with open(pkl_path, 'rb') as f:
        arr = pickle.load(f)
    if arr.ndim != 3:
        raise ValueError(f'Unexpected array ndim={arr.ndim} for {pkl_path}, expect 3D (trial, channel, time)')
    return arr


def _segment_trials(arr_3d):
    """将 (trial, ch, time) 重采样到 6000 点，然后每个 trial 切成 3 段样本，
    每段样本形状 (ch, 2000)；返回 (num_samples, ch, 2000) 与对应标签。
    逻辑参考用户给定 FACED 预处理脚本。
    """
    # axis=2 为时间维，重采样到 6000
    eeg = signal.resample(arr_3d, 6000, axis=2)
    n_trial, n_ch, _ = eeg.shape

    # 依据示例：按每 200 点一个窗口，共 30 个窗口
    try:
        eeg_win = eeg.reshape(n_trial, n_ch, 30, 200)
    except Exception as e:
        raise ValueError(f'Cannot reshape to (trial, ch, 30, 200), got {eeg.shape} from {arr_3d.shape}')

    # 标签：按示例固定为 28 个 trial；若不匹配则报错
    if n_trial != len(FACED_TRIAL_LABELS):
        raise ValueError(f'FACED trials ({n_trial}) != expected labels ({len(FACED_TRIAL_LABELS)})')

    samples = []
    labels = []
    for i in range(n_trial):
        # 3 段，每段取 10 个窗口 → (ch, 10, 200) → 合并为 (ch, 2000)
        for j in range(3):
            seg = eeg_win[i, :, 10*j:10*(j+1), :]  # (ch, 10, 200)
            seg = seg.reshape(n_ch, 2000)
            samples.append(seg)
            labels.append(FACED_TRIAL_LABELS[i])
    X = np.stack(samples, axis=0).astype(np.float32)
    Y = np.array(labels, dtype=np.uint8)
    return X, Y


def proc_one(subject, pkl_path):
    arr = _load_faced_array(pkl_path)
    X, Y = _segment_trials(arr)
    X = pipeline(X, CH_NAMES)
    print(subject, X.shape, Y.shape, np.unique(Y, return_counts=True))
    return subject, X, Y


def proc_all():
    fpaths, subjects = _list_subject_files()
    # 更新 META subjects
    global EMO_FACED
    EMO_FACED.subjects = subjects

    with mp.Pool(len(subjects)) as pool:
        res = pool.starmap(proc_one, [(sub, fp) for sub, fp in zip(subjects, fpaths)])

    os.makedirs(DATA_FOLDER, exist_ok=True)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))


if __name__ == '__main__':
    proc_all()

