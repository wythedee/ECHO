import os
import mne
import h5py
import numpy as np
import sys
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
sys.path.append(os.path.abspath('.'))
from EEG_Dataset.EMO_03_SEED_V import SUBJECTS
from EEG_Montage.AdaptiveGrouping import AdaptiveGrouping
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline, split_trial

SRC_FOLDER = os.path.join(SRC_FOLDER, 'EMO')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'EMO')
NAME = "EMO_05_THU-EP"

CH_NAMES = [
    'Fp1', 'Fp2','Fz', 'F3', 'F4', 'F7', 'F8', 
    'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4',
    'A1', 'A2',
    'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 
    'PO3', 'PO4', 'Oz', 'O1', 'O2'
]

SUBJECTS = [f'sub_{i}' for i in range(1, 81) if f'sub_{i}' != 'sub_38']

EMO_THU_EP = META(NAME, CH_NAMES, SUBJECTS, ['EMO/Neg', 'EMO/Pos'], resample_rate=250, time_length=4)

label_type = 'A'

def proc_one(sub):
    sfreq = 250
    binary = True
    src_data_path = os.path.join(SRC_FOLDER, NAME, f'EEG_data/{sub}.mat')
    try:
        with h5py.File(src_data_path, 'r') as f:
            data = f['data'][()]
            data = data[:,:,:,-1] # broad band filtered
    except OSError as e:
        print(f'Error loading {sub}, {e}')
        return None, None, None
    X = np.transpose(data, (2, 1, 0))
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(X, info)
    # epochs.filter(l_freq=1, h_freq=40, verbose=False)
    epochs.resample(250, npad='auto')
    X = epochs.get_data(copy=False).astype(np.float32)

    # Ratings
    src_label_path = os.path.join(SRC_FOLDER, NAME, f'Ratings/ratings.mat')
    with h5py.File(src_label_path, 'r') as f:
        ratings = f['ratings'][()]
        ratings = np.transpose(ratings, (2, 1, 0))
        if label_type == 'A':
            sub_label = ratings[int(sub.split('_')[1]) - 1][:, 8]
        elif label_type == 'V':
            sub_label = ratings[int(sub.split('_')[1]) - 1][:, 9]
        else:
            raise ValueError(f'Unknown label type: {label_type}')
        if binary:
            Y = np.where(sub_label <= 3, 0, 1)
        else:
            Y = sub_label
    # ---- Split into 4-second segments (1000 samples at 250 Hz) ----
    # Convert X to list format (time, channel)
    X_list = [trial.T for trial in X]  # (trial, time, channel)
    Y_list = Y.tolist()

    X_split, Y_split = split_trial(X_list, Y_list, segment_length=4, overlap=0, sampling_rate=250)

    # Flatten segments and convert back to (segment, channel, time)
    X_final = []
    Y_final = []
    for segs, labels in zip(X_split, Y_split):
        for seg, lab in zip(segs, labels):
            X_final.append(seg.T)  # (channel, time)
            Y_final.append(lab)

    if len(X_final) == 0:
        print(f'No segments extracted for {sub}')
        return sub, None, None

    X = np.array(X_final)
    Y = np.array(Y_final).astype(np.uint8)

    X = pipeline(X, CH_NAMES)
    return sub, X, Y

def proc_all():
    # Load data for all subjects using multiprocessing
    print(SUBJECTS)
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)

    # Save processed data to HDF5 file
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            if sub is None:
                print(f'Skipping {sub} because of missing data')
                continue
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    proc_all()
    # proc_one('sub_1')

