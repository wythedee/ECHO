from curses import keyname
import os
import mne
import h5py
import numpy as np
import pandas as pd
import sys
import scipy.io as sio
import multiprocessing as mp
import multiprocessing.dummy as dmp
import os.path as osp
sys.path.append(os.path.abspath('.'))
from EEG_Dataset.EMO_03_SEED_V import SUBJECTS
from EEG_Montage.AdaptiveGrouping import AdaptiveGrouping
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline, split_trial


NAME = "EMO_04_SEED"

CH_NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
            'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
            'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
            'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5',
            'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

ZONES = {
    'Frontal': ['FP1', 'FPZ', 'FP2',  # Fronto-polar
                'AF3', 'AF4',       # Anterior-frontal
                'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'], # Frontal
    'Central': ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', # Fronto-central
                'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'],      # Central
    'Temporal': ['FT7', 'FT8',      # Fronto-temporal
                 'T7', 'T8',         # Temporal
                 'TP7', 'TP8'],      # Temporo-parietal (lateral)
    'Parietal': ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', # Centro-parietal
                 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'], # Parietal
    'Occipital': ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', # Parieto-occipital
                  'O1', 'OZ', 'O2',                               # Occipital
                  'CB1', 'CB2']                                  # Cerebellar (often grouped near Occipital)
}


SUBJECTS = range(0, 15)

EMO_SEED = META(NAME, CH_NAMES, SUBJECTS, ['EMO/Neg', 'EMO/Pos'], resample_rate=250, time_length=4)

label_type = 'A'

def load_one_session(file_to_load, feature_key):
    # Load EEG data from the specified file
    print('Loading file:{}'.format(file_to_load))
    data = sio.loadmat(file_to_load, verify_compressed_data_integrity=False)
    data_session = []
    keys_to_select = [k for k in data.keys() if feature_key in k]
    for k in keys_to_select:
        one_trial = data[k]
        data_session.append(one_trial)
    min_length = min([item.shape[1] for item in data_session])
    data_session = [sess[:, :min_length] for sess in data_session]
    data_session = np.array(data_session)
    return data_session

def proc_one(sub):
    sub += 1
    keep_dim = False
    num_classes = 2
    data_folder = osp.join(SRC_FOLDER, NAME, 'SEED/Preprocessed_EEG')
    label = sio.loadmat(osp.join(data_folder, 'label.mat'))['label']
    label += 1
    label = np.squeeze(label)
    files_this_subject = []
    for root, dirs, files in os.walk(data_folder, topdown=False):
        for name in files:
            if sub < 10:
                sub_code = name[:2]
            else:
                sub_code = name[:3]
            if '{}_'.format(sub) == sub_code:
                files_this_subject.append(name)
    files_this_subject = sorted(files_this_subject)

    data_subject = []
    label_subject = []
    for file in files_this_subject:
        sess = load_one_session(
            file_to_load=osp.join(data_folder, file), feature_key='eeg'
        )
        if num_classes == 2:
            idx_keep = np.delete(np.arange(label.shape[-1]), np.where(label == 1)[0])
            sess = [sess[idx] for idx in idx_keep]
            label_selected = [label[idx] for idx in idx_keep]
            label_selected = np.where(np.array(label_selected) == 2, 1, 0)
        else:
            label_selected = label
        if keep_dim:
            data_subject.append(sess)
            label_subject.append(label_selected)
        else:
            data_subject.extend(sess)
            label_subject.extend(list(label_selected))
    data_subject = np.array(data_subject)
    print(f'{sub} data_subject.shape:', data_subject.shape)
    if len(data_subject) == 0:
        print(f'No data for subject {sub}')
        return sub, None, None
    info = mne.create_info(ch_names=CH_NAMES, sfreq=1000, ch_types='eeg')
    epochs = mne.EpochsArray(data_subject, info)
    epochs.resample(250, npad='auto')
    X = epochs.get_data(copy=False).astype(np.float32)
    Y = np.array(label_subject)
    
    # Convert to list format for split_trial: (trial, channel, time) -> list of (time, channel)
    X_list = [X[i].T for i in range(X.shape[0])]  # transpose each trial from (channel, time) to (time, channel)
    Y_list = Y.tolist()
    
    # Split trials into 4-second segments
    X_split, Y_split = split_trial(X_list, Y_list, segment_length=4, overlap=0, sampling_rate=250, sub_segment=0, sub_overlap=0.0)
    
    if len(X_split) == 0:
        print(f'No segments extracted for subject {sub}')
        return sub, None, None
    
    # Convert back to numpy array format: flatten all segments into (total_segments, channel, time)
    X_final = []
    Y_final = []
    for trial_segments, trial_labels in zip(X_split, Y_split):
        for segment, label in zip(trial_segments, trial_labels):
            X_final.append(segment.T)  # transpose back from (time, channel) to (channel, time)
            Y_final.append(label)
    
    X = np.array(X_final)
    Y = np.array(Y_final)
    
    X = pipeline(X, CH_NAMES)
    return sub, X, Y


def proc_all():
    # Load data for all subjects using multiprocessing
    print(SUBJECTS)
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)

    # Save processed data to HDF5 file
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        min_length = min([item.shape[2] for _, item, _ in res if item is not None])
        for sub, X, Y in res:
            if X is None or Y is None:
                print(f'Skipping {sub} because of missing data')
                continue
            X = X[:, :, :min_length]
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    proc_all()
    # proc_one(1)

