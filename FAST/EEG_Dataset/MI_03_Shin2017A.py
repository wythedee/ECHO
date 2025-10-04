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
SRC_NAME = 'MI_Shin2017A'
# https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bbci_eeg_fnirs.py#L192-L312
NAME = 'MI_03_Shin2017A'
SUBJECTS = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
    '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 
    '23', '24', '25', '26', '27', '28', '29']
CH_NAMES = [
    'F7', 'AFF5h', 'F3', 'AFp1', 'AFp2', 'AFF6h', 'F4', 'F8', 'AFF1h', 
    'AFF2h', 'Cz', 'Pz', 'FCC5h', 'FCC3h', 'CCP5h', 'CCP3h', 'T7', 'P7',
    'P3', 'PPO1h', 'POO1', 'POO2', 'PPO2h', 'P4', 'FCC4h', 'FCC6h', 'CCP4h',
    'CCP6h', 'P8', 'T8'
]

MI_Shin2017A = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=4)

def proc_one(subject):
    fname1 = f'{SRC_FOLDER}/{SRC_NAME}/subject {subject}/with occular artifact/cnt.mat'
    fname2 = f'{SRC_FOLDER}/{SRC_NAME}/subject {subject}/with occular artifact/mrk.mat'
    data = scipy.io.loadmat(fname1, squeeze_me=True, struct_as_record=False)["cnt"]
    mrk = scipy.io.loadmat(fname2, squeeze_me=True, struct_as_record=False)["mrk"]
    X, Y = [], []
    for ii in [0, 2, 4]:
        sfreq = 200
        ch_names = list(data[ii].clab)
        ch_types = ["eeg"] * 30 + ["eog"] * 2
        eeg = data[ii].x.T #* 1e-6
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = mne.io.RawArray(data=eeg, info=info, verbose=False)
        raw.drop_channels(['VEOG', 'HEOG'])
        raw.filter(l_freq=1, h_freq=40, verbose=False)

        trig_offset = 0
        mkr_time = ((mrk[ii].time - 1) // 5) / sfreq
        mkr = mrk[ii].event.desc // 16 + trig_offset
        print(mkr_time.shape, mkr.shape)
        
        events = np.column_stack(((mkr_time * sfreq).astype(int), np.zeros_like(mkr), mkr)).astype(int)
        tmin = 0
        # tmax = 10
        tmax = 4
        epochs = mne.Epochs(raw, events=events, event_id=None, tmin=tmin, tmax=tmax, preload=True, verbose=False, baseline=None)
        epochs.resample(250, npad="auto")
        x = epochs.get_data(copy=False)[:, : ,1:].astype(np.float32)
        y = mkr - 1
        X.append(x)
        Y.append(y)
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
    return subject, X, Y

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