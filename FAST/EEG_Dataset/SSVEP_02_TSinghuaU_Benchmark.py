import os
import mne
import scipy.io
mne.set_log_level('WARNING')
import numpy as np
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
import h5py
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'SSVEP')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'SSVEP')
NAME = 'SSVEP_02_TSinghuaU_Benchmark'
SUBJECTS = [
    'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 
    'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 
    'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35']
ORIGIN_CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
    'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2',
]
# drop 'CB1', 'CB2'
CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
    'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2',
]

# FREQUECIES = scipy.io.loadmat(f'{SRC_FOLDER}/SSVEP_02_TSinghuaU_Benchmark/Freq_Phase.mat')['freqs'].flatten()
# print([f'{f:.1f}' for f in FREQUECIES])
FREQUECIES = ['f/08.0', 'f/09.0', 'f/10.0', 'f/11.0', 'f/12.0', 'f/13.0', 'f/14.0', 'f/15.0', 'f/08.2', 'f/09.2', 
              'f/10.2', 'f/11.2', 'f/12.2', 'f/13.2', 'f/14.2', 'f/15.2', 'f/08.4', 'f/09.4', 'f/10.4', 'f/11.4', 
              'f/12.4', 'f/13.4', 'f/14.4', 'f/15.4', 'f/08.6', 'f/09.6', 'f/10.6', 'f/11.6', 'f/12.6', 'f/13.6', 
              'f/14.6', 'f/15.6', 'f/08.8', 'f/09.8', 'f/10.8', 'f/11.8', 'f/12.8', 'f/13.8', 'f/14.8', 'f/15.8']

SSVEP_TSinghuaU_Benchmark = META(NAME, CH_NAMES, SUBJECTS, FREQUECIES)

def proc_one(sub):
    data = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/{sub}.mat')
    # Phase_info = np.tile(PHASE, x.shape[1])
    x = data['data']
    # (64, 1500, 40, 6) -> (40, 6, 64, 1500)
    x = x.transpose(2, 3, 0, 1) 
    y = np.tile(np.arange(len(FREQUECIES)), 6)
    x = x.reshape(-1, x.shape[2], x.shape[3])
    print(sub, x.shape, y.shape, y[:80])

    info = mne.create_info(ch_names=ORIGIN_CH_NAMES, sfreq=250, ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.drop_channels(['CB1', 'CB2'])
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
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
    # proc_one('S1')


