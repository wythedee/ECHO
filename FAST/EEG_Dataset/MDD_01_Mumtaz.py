import os
import mne
import numpy as np
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline

NAME = 'MDD_01_Mumtaz'

# 数据根目录（可按需修改）。脚本将自动遍历其中的 EDF 文件。
ROOT_DIR = os.path.join(SRC_FOLDER, 'MDD', 'MDD_01_Mumtaz')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'MDD')

# 选取的通道名（与preprocessing_mumtaz.py保持一致）
SELECTED_CHANNELS = [
    'EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE',
    'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE',
    'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE'
]

# 映射到标准化 CH_NAMES（去除前后缀，与 pipeline 期望一致，使用经典 10-20 名称）
CH_NAMES = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
]

RESAMPLE_RATE = 250
TIME_LENGTH = 5

# 类别：0 正常(H), 1 MDD
CLASSES = ['MDD/H', 'MDD/MDD']

MDD_Mumtaz = META(NAME, CH_NAMES, [], CLASSES, resample_rate=RESAMPLE_RATE, time_length=TIME_LENGTH)


# 采用preprocessing_mumtaz.py的文件遍历逻辑，但加强文件类型检查
def _iter_files(root_dir):
    """遍历文件夹，按照原脚本逻辑筛选文件"""
    files_H, files_MDD = [], []
    for file in os.listdir(root_dir):
        # 只处理EDF文件，排除TASK文件，只处理EC和EO状态
        if file.lower().endswith('.edf') and 'TASK' not in file:
            if 'MDD' in file:
                files_MDD.append(file)
            else:
                files_H.append(file)
    return sorted(files_H), sorted(files_MDD)


def _load_and_segment_edf(edf_path):
    """按照preprocessing_mumtaz.py的逻辑处理EDF文件"""
    # 读取EDF文件
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    
    # 选择通道（与原脚本一致）
    raw.pick_channels(SELECTED_CHANNELS, ordered=True)
    
    # 预处理步骤（完全按照原脚本）
    raw.resample(RESAMPLE_RATE)  # 重采样到200Hz
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)  # 带通滤波 0.3-75Hz
    raw.notch_filter((50), verbose=False)  # 50Hz陷波滤波
    
    # 转换为数组（不使用pandas，直接获取数据）
    eeg_array = raw.get_data().T  # (time_points, channels)
    points, chs = eeg_array.shape
    
    # 按5秒窗口分割（与原脚本逻辑一致）
    a = points % (TIME_LENGTH * RESAMPLE_RATE)
    if a != 0:
        eeg_array = eeg_array[:-a, :]
        points -= a
    
    if points == 0:
        return np.zeros((0, chs, TIME_LENGTH * RESAMPLE_RATE), dtype=np.float32)
    
    # 重塑数组：(n_seg, 5, 200, ch) → (n_seg, ch, 5, 200) → (n_seg, ch, 1000)
    eeg_array = eeg_array.reshape(-1, TIME_LENGTH, RESAMPLE_RATE, chs)
    eeg_array = eeg_array.transpose(0, 3, 1, 2)  # (n_seg, ch, 5, 200)
    eeg_array = eeg_array.reshape(eeg_array.shape[0], chs, TIME_LENGTH * RESAMPLE_RATE).astype(np.float32)
    
    return eeg_array


def proc_one(filename):
    """处理单个文件，采用与原脚本一致的标签逻辑"""
    sub = os.path.splitext(filename)[0]
    # 标签逻辑：MDD=1, 健康=0（与原脚本一致）
    label = 1 if 'MDD' in filename else 0
    
    X = _load_and_segment_edf(os.path.join(ROOT_DIR, filename))
    Y = np.full((X.shape[0],), label, dtype=np.uint8)
    
    if X.shape[0] == 0:
        print(f'Skip {filename}: empty after segmentation')
        return sub, None, None
    
    # 应用pipeline处理
    X = pipeline(X, CH_NAMES)
    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    return sub, X, Y


def proc_all():
    """主处理函数，采用原脚本的文件筛选逻辑"""
    # 使用原脚本的文件遍历逻辑（排除TASK文件）
    files_H, files_MDD = _iter_files(ROOT_DIR)
    print(f"找到健康对照组文件: {len(files_H)}")
    print(f"找到MDD患者文件: {len(files_MDD)}")
    
    subjects = [os.path.splitext(f)[0] for f in files_H + files_MDD]
    MDD_Mumtaz.subjects = subjects

    files_all = files_H + files_MDD
    print(f"总共处理文件数: {len(files_all)}")
    
    # 多进程处理
    with mp.Pool(min(len(files_all), os.cpu_count() or 4)) as pool:
        res = pool.map(proc_one, files_all)

    # 保存到HDF5文件
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            if X is None or Y is None:
                continue
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(f"保存 {sub}: X{X.shape}, Y{Y.shape}, 标签分布{np.unique(Y, return_counts=True)}")


if __name__ == '__main__':
    proc_all()



