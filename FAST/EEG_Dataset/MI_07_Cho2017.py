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
SRC_NAME = 'MI_Cho2017'
NAME = 'MI_07_Cho2017'
SUBJECTS = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 
            's19', 's2', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 
            's28', 's29', 's3', 's30', 's31', 's33', 's34', 's35', 's36', 's37', 
            's38', 's39', 's4', 's40', 's41', 's42', 's43', 's44', 's45', 's47', 
            's48', 's5', 's50', 's51', 's52', 's6', 's7', 's8', 's9']
CH_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]
MI_Cho2017 = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=10)

def proc_one(subject):
    data = np.load(f'{SRC_FOLDER}/{SRC_NAME}/{subject}.npz', allow_pickle=True)
    
    print(data['fs'], data['metadata'])
    print(data['x_data'].shape, data['y_data'].shape, np.unique(data['y_data'], return_counts=True))
    
    x, y = data['x_data']/100, data['y_data'].astype(np.uint8)
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
    
