import os
import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
from scipy import signal
import multiprocessing as mp
import h5py
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'MI')
SRC_NAME = 'MI_SHU'
NAME = 'MI_13_SHU'

# 根据数据检查结果，共有25个被试
SUBJECTS = [f'sub-{i:03d}' for i in range(1, 26)]  # 25 subjects

# 根据数据检查结果，有32个通道
CH_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 
    'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 
    'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 
    'O2', 'PO10'
]

MI_SHU = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], time_length=4, resample_rate=250)

def proc_one(sub):
    # 构建被试数据路径
    sub_dir = f'{SRC_FOLDER}/{SRC_NAME}/files'
    
    # 获取该被试的所有文件
    all_files = [f for f in os.listdir(sub_dir) if f.startswith(sub) and f.endswith('.mat')]
    all_files = sorted(all_files)
    
    print(f"Processing subject {sub}, found {len(all_files)} files")
    
    # 收集所有数据
    all_data = []
    all_labels = []
    
    # 处理每个会话文件
    for file in all_files:
        file_path = os.path.join(sub_dir, file)
        data = scipy.io.loadmat(file_path)
        
        # 提取EEG数据和标签
        eeg_data = data['data']  # Shape: (100, 32, 1000) - 250Hz * 4s = 1000 points
        labels = data['labels'][0]  # Shape: (100,)
        
        print(f"  File {file}: data shape {eeg_data.shape}, labels shape {labels.shape}")
        
        # 直接使用原始的4秒数据，不进行分割
        # Shape: (100, 32, 1000)
        for i in range(eeg_data.shape[0]):
            all_data.append(eeg_data[i])  # Shape: (32, 1000)
            all_labels.append(labels[i] - 1)  # 调整标签为0-based索引
    
    # 转换为numpy数组
    X = np.array(all_data)  # Shape: (N, 32, 1000)
    Y = np.array(all_labels)  # Shape: (N,)
    
    print(f"Subject {sub}: Combined data shape {X.shape}, labels shape {Y.shape}")
    print(f"  Unique labels: {np.unique(Y, return_counts=True)}")
    
    # 应用MNE预处理
    sfreq = 250  # 原始采样率
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(X, info)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    # 注意：这里不再重采样，因为我们已经使用正确的采样率
    X = epochs.get_data(copy=False).astype(np.float32)
    Y = Y.astype(np.uint8)
    
    print(f"Subject {sub}: After MNE preprocessing - X shape {X.shape}, Y shape {Y.shape}")
    
    # 应用标准化管道
    X = pipeline(X, CH_NAMES)
    
    return sub, X, Y

def proc_all():
    with mp.Pool(min(len(SUBJECTS), 10)) as pool:  # 限制进程数以避免内存问题
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(f"Saved subject {sub}: X shape {X.shape}, Y shape {Y.shape}")
            print(f"  Unique labels: {np.unique(Y, return_counts=True)}")

if __name__ == '__main__':
    proc_all()
