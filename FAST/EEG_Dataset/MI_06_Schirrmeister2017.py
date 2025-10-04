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
DATA_FOLDER = os.path.join(DATA_FOLDER, 'MI')
SRC_NAME = 'MI_Schirrmeister2017'
NAME = 'MI_06_Schirrmeister2017'
SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14']
CH_NAMES = [
    'Fp1','Fp2','Fpz','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
    'M1','T7','C3','Cz','C4','T8','M2','CP5','CP1','CP2','CP6','P7','P3',
    'Pz','P4','P8','POz','O1','Oz','O2','AF7','AF3','AF4','AF8','F5','F1',
    'F2','F6','FC3','FCz','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P5',
    'P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8',
    'FT9','FT10','TPP9h','TPP10h','PO9','PO10','P9','P10','AFF1','AFz','AFF2',
    'FFC5h','FFC3h','FFC4h','FFC6h','FCC5h','FCC3h','FCC4h','FCC6h','CCP5h','CCP3h',
    'CCP4h','CCP6h','CPP5h','CPP3h','CPP4h','CPP6h','PPO1','PPO2','I1','Iz','I2','AFp3h',
    'AFp4h','AFF5h','AFF6h','FFT7h','FFC1h','FFC2h','FFT8h','FTT9h','FTT7h','FCC1h',
    'FCC2h','FTT8h','FTT10h','TTP7h','CCP1h','CCP2h','TTP8h','TPP7h','CPP1h','CPP2h',
    'TPP8h','PPO9h','PPO5h','PPO6h','PPO10h','POO9h','POO3h','POO4h','POO10h','OI1h','OI2h'
]

MI_Schirrmeister2017 = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=10)

def proc_one(subject):
    data = np.load(f'{SRC_FOLDER}/{SRC_NAME}/{subject}.npz', allow_pickle=True)
    
    print(data['fs'], data['metadata'])
    print(data['x_data'].shape, data['y_data'].shape, np.unique(data['y_data'], return_counts=True))
    
    x, y = data['x_data'], data['y_data'].astype(np.uint8)
    mask = (y == 2) | (y == 3)
    x = x[mask]
    y = y[mask] - 2
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    
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