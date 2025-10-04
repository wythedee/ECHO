import os
import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
import h5py
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'MI')
NAME = 'MI_02_ShanghaiU'
SRC_NAME = 'MI_ShanghaiU'
SUBJECTS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 
            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 
            'S20', 'S21', 'S22', 'S23', 'S24', 'S25']
CH_NAMES = [
    'Fp1', 'Fp2','Fz', 'F3', 'F4', 'F7', 'F8', 
    'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4',
    'A1', 'A2',
    'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 
    'PO3', 'PO4', 'Oz', 'O1', 'O2'
]

# Channel grouping into 5 functional zones (frontal, central, temporal, parietal, occipital)
ZONES = {
    'Frontal': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'A1', 'A2'],
    'Central': ['C3', 'Cz', 'C4'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Parietal': ['CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4'],
    'Occipital': ['PO3', 'PO4', 'O1', 'Oz', 'O2'],
}

MI_ShanghaiU = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], time_length=4, resample_rate=250)

def proc_one(sub):
    sess1 = scipy.io.loadmat(f'{SRC_FOLDER}/{SRC_NAME}/{sub}/d1.mat')
    sess2 = scipy.io.loadmat(f'{SRC_FOLDER}/{SRC_NAME}/{sub}/d2.mat')
    sess3 = scipy.io.loadmat(f'{SRC_FOLDER}/{SRC_NAME}/{sub}/d3.mat')
    sess4 = scipy.io.loadmat(f'{SRC_FOLDER}/{SRC_NAME}/{sub}/d4.mat')
    sess5 = scipy.io.loadmat(f'{SRC_FOLDER}/{SRC_NAME}/{sub}/d5.mat')
    epoched = np.concatenate([sess1['data'], sess2['data'], sess3['data'], sess4['data'], sess5['data']], axis=0)
    label = np.concatenate([sess1['labels'][0], sess2['labels'][0], sess3['labels'][0], sess4['labels'][0], sess5['labels'][0]], axis=0)
    sfreq = 250
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(epoched, info)
    # 直接使用 SciPy 重采样到 800 点，不做滤波或 MNE 自带重采样
    X = epochs.get_data(copy=False).astype(np.float32)
    X = scipy.signal.resample(X, 800, axis=2).astype(np.float32)
    Y = label.astype(np.uint8) - 1
    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    # X = pipeline(X, CH_NAMES)
    return sub, X, Y

def proc_all():
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    proc_all()