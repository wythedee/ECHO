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
SRC_NAME = 'MI_Track1_Few_shot'
# https://brain.korea.ac.kr/bci2021/competition.php
NAME = 'MI_08_Track1_Few_shot'
SUBJECTS = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
    '12', '13', '14', '15', '16', '17', '18', '19', '20']
CH_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 
    'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 
    'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 
    'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 
    'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 
    'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 
    'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
]

MI_Track1_Few_shot = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'])

def proc_one(sub):
    fname_train = f'{SRC_FOLDER}/{SRC_NAME}/Training set/Data_Sample{sub}.mat'
    fname_valid = f'{SRC_FOLDER}/{SRC_NAME}/Validation set/Data_Sample{sub}.mat'
    data_1 = scipy.io.loadmat(fname_train, squeeze_me=True, struct_as_record=False)['Training']
    data_2 = scipy.io.loadmat(fname_valid, squeeze_me=True, struct_as_record=False)['Validation']
    sfreq = data_1.fs
    x_1, y_1 = data_1.x, data_1.y_dec - 1
    x_2, y_2 = data_2.x, data_2.y_dec - 1
    x = np.concatenate([x_1, x_2], axis=1).transpose((1, 2, 0))
    y = np.concatenate([y_1, y_2], axis=0)
    ### in the description, right is 1 and left is 2, here we need to swap them
    y = 1-y
    print(sub, sfreq, x.shape, y.shape, np.unique(y, return_counts=True))
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    epochs.resample(250, npad='auto')
    X = epochs.get_data(copy=False).astype(np.float32)
    Y = y.astype(np.uint8)
    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
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
