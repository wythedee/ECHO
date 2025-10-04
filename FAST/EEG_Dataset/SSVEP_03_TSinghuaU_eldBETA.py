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
NAME = 'SSVEP_03_TSinghuaU_eldBETA'
SUBJECTS = [
    'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 
    'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25',
    'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 
    'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46', 'S47', 'S48', 'S49', 
    'S50', 'S51', 'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 
    'S62', 'S63', 'S64', 'S65', 'S66', 'S67', 'S68', 'S69', 'S70', 'S71', 'S72', 'S73', 
    'S74', 'S75', 'S76', 'S77', 'S78', 'S79', 'S80', 'S81', 'S82', 'S83', 'S84', 'S85', 
    'S86', 'S87', 'S88', 'S89', 'S90', 'S91', 'S92', 'S93', 'S94', 'S95', 'S96', 'S97', 
    'S98', 'S99', 'S100']

ORIGIN_CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
    'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

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

# data = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/S1.mat')['data'][0]
# label = data['Suppl_info'][0][0]['Frequency'][0][0]
# print([f'{f:.1f}' for f in label])
FREQUECIES = ['f/08.0', 'f/09.5', 'f/11.0', 'f/08.5', 'f/10.0', 'f/11.5', 'f/09.0', 'f/10.5', 'f/12.0']
SSVEP_TsinghuaU_eldBETA = META(NAME, CH_NAMES, SUBJECTS, FREQUECIES)
    
def proc_one(sub):
    data = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/{sub}.mat')['data'][0]
    x = data['EEG'][0][0]['Epoch'][0]
    y = data['Suppl_info'][0][0]['Frequency'][0][0]
    sfreq = data['Suppl_info'][0][0]['Srate'][0][0]
    phase = data['Suppl_info'][0][0]['Phase'][0][0]
    # print(sub, x.shape, y.shape, np.unique(y, return_counts=True))

    x = x.transpose(2, 3, 0, 1) 
    y = np.tile(np.arange(len(FREQUECIES)), 7)
    x = x.reshape(-1, x.shape[2], x.shape[3])
    print(sub, x.shape, y.shape, y[:80], sfreq)

    info = mne.create_info(ch_names=ORIGIN_CH_NAMES, sfreq=250, ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.drop_channels(['CB1', 'CB2'])
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    # epochs.resample(250, npad='auto')
    X = epochs.get_data(copy=False).astype(np.float32)
    Y = y.astype(np.uint8)
    # Phase = phase.astype(np.float32)
    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
    return sub, X, Y#, Phase

def proc_all():
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            # f.create_dataset(f'{sub}/Phase', data=Phase)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    print(f'{DATA_FOLDER}/{NAME}.h5')

if __name__ == '__main__':
    # proc_one('S1')
    proc_all()