from functools import partial
import torch
import einops
import glob
import mne
import numpy as np
import os
import pickle
import scipy
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import warnings
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline, split_trial


NAME = 'EMO_03_SEED_V'
# SUBJECTS = ['1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_', 
#             '11_', '12_', '13_', '14_', '15_', '16_']
# subject 7 have issues in data
SUBJECTS = ['1_', '2_', '3_', '4_', '5_', '6_', '8_', '9_', '10_', 
            '11_', '12_', '13_', '14_', '15_', '16_']

ORIGINAL_CH_NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
                     'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                     'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
                     'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7',
                     'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                     'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                     'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                     'CB1', 'O1', 'OZ', 'O2', 'CB2']

CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
            'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'O1', 'Oz', 'O2',]

NEW_CH_NAMES = [
    'A1', 'A2', 'TP9', 'TP10', 'F9', 'F10', 'Fp1', 'Fp2', 'Fpz',
    'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T3', 'C3', 'Cz', 'C4', 'T4', 'CP5', 'CP1', 'CP2', 'CP6',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'POz', 'O1', 'Oz', 'O2',
    'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
    'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3',
    'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3',
    'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
    'FT9', 'FT10', 'PO9', 'PO10', 'P9', 'P10', 'AFz'
] # 75 channels

ZONES = {
    'Frontal': ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8'],
    'Central': ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
    'Temporal': ['T7', 'T8', 'TP7', 'TP8'],
    'Parietal': ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    'Occipital': ['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
}

session_labels ={
    1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

time_stamp = {
    1: {
        'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
        'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
    },
    2: {
        'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
        'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]
    },  
    3: {
        'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
        'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
    },
}

EMO_SEED_V = META(NAME, CH_NAMES, SUBJECTS, ['EMO/Disgust', 'EMO/Fear', 'EMO/Sad', 'EMO/Neutral', 'EMO/Happy'], resample_rate=250, time_length=1)

def proc_one(sub):
    X, Y = [], []
    for session in [1, 2, 3]:
        fn = list(glob.glob(f'{SRC_FOLDER}/{NAME}/EEG_raw/{sub}{session}*.cnt'))[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            eeg_raw = mne.io.read_raw_cnt(fn)
        
        useless_ch = ['M1', 'M2', 'VEO', 'HEO', 'CB1', 'CB2']
        eeg_raw.drop_channels(useless_ch)
        sfreq = int(eeg_raw.info['sfreq'])
        time_stamp_start = time_stamp[session]['start']
        time_stamp_end = time_stamp[session]['end']
        data_matrix = eeg_raw.get_data()

        for i in range(1, 16):
            trial = data_matrix[:, time_stamp_start[i-1]*sfreq : time_stamp_end[i-1]*sfreq]
            raw = mne.io.RawArray(trial, mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types='eeg'))
            raw = raw.resample(250)
            raw.filter(l_freq=1, h_freq=40, verbose=False)
            x = raw.get_data().astype(np.float32)
            # Convert to list format for split_trial: (channel, time) -> (time, channel)
            x_transposed = x.T  # (time, channel)
            label_for_trial = [session_labels[session][i-1]]
            
            # Split into 4-second segments
            x_split, y_split = split_trial([x_transposed], label_for_trial, segment_length=1, overlap=0, sampling_rate=250)
            
            # Convert back to (segments, channel, time) format
            if len(x_split) > 0 and len(x_split[0]) > 0:
                x = x_split[0].transpose(0, 2, 1)  # (segments, time, channel) -> (segments, channel, time)
                y = y_split[0].astype(np.uint8)
            else:
                x = np.array([]).reshape(0, x.shape[0], 1000)  # empty array with correct shape
                y = np.array([]).astype(np.uint8)
            X.append(x)
            Y.append(y)
        
        print('Subject:', sub, 'Session', session, 'Done')

    X, Y = np.concatenate(X), np.concatenate(Y)
    print('Subject: ',sub,X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
    return sub, X, Y

def proc_all():
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    proc_all()
