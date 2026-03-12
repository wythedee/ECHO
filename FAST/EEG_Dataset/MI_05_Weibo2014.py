import mne
import numpy as np
import os
import pickle
import scipy
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')
# https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/Weibo2014.py#L64-L188
# [
#     'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
#     'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
#     'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
#     'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
#     'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2', 'VEO', 'HEO',
# ]
SRC_NAME = 'MI_Weibo2014'
NAME = 'MI_05_Weibo2014'
SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
    'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
    'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
    'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2',
]

MI_Weibo2014 = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=4)

def proc_one(subject):
    data = np.load(f'{SRC_FOLDER}/{SRC_NAME}/{subject}.npz', allow_pickle=True)
    
    print(data['fs'], data['metadata'])
    print(data['x_data'].shape, data['y_data'].shape, np.unique(data['y_data'], return_counts=True))
    
    x, y = data['x_data'], data['y_data'].astype(np.uint8)
    mask = (y == 1) | (y == 2)
    x = x[mask]
    y = y[mask] - 1
    # y = y - 1
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    
    events = dict(
        left_hand=1,
        right_hand=2,
        hands=3,
        feet=4,
        left_hand_right_foot=5,
        right_hand_left_foot=6,
        rest=7,
    )
    info = mne.create_info(ch_names=CH_NAMES, sfreq=data['fs'], ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    epochs.resample(250, npad='auto')
    x = epochs.get_data(copy=False).astype(np.float32)
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    x = pipeline(x, CH_NAMES)
    return subject, x, y

def proc_all():
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__=='__main__':
    proc_all()