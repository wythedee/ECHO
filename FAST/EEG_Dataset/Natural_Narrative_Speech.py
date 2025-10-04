import os
import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
from functools import partial
import h5py
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = "/media/james/public/dataset"

NAME = 'Natural_Narrative_Speech'
SUBJECTS = [
    'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 
    'Subject7', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12', 
    'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject18', 
    'Subject19'
]
CH_NAMES = [
    "Cz", "A2", "CPz", "A4", "P1", "A6", "P3", "A8", "A9", "PO7", 
    "A11", "A12", "A13", "A14", "O1", "A16", "PO3", "CMS", "Pz", 
    "A20", "POz", "A22", "Oz", "A24", "Iz", "A26", "A27", 
    "O2", "A29", "PO4", "DRL", "P2", "B1", "CP2", "B3", "P4", 
    "B5", "B6", "PO8", "B8", "B9", "P10", "P8", "B12", "P6", 
    "TP8", "B15", "CP6", "B17", "CP4", "B19", "C2", "B21", "C4", 
    "B23", "C6", "B25", "T8", "FT8", "B28", "FC6", "B30", "FC4", 
    "B32", "C1", "C2", "C3", "F4", "F6", "C6", "F8", "AF8", "C9", 
    "C10", "FC2", "F2", "C13", "C14", "AF4", "Fp2", "Fpz", "C18", 
    "AFz", "C20", "Fz", "C22", "FCz", "FC1", "F1", "C26", "C27", 
    "AF3", "Fp1", "AF7", "C31", "C32", "D1", "D2", "D3", "F3", 
    "F5", "D6", "F7", "FT7", "D9", "FC5", "D11", "FC3", "D13", 
    "C1", "D15", "CP1", "D17", "D18", "C3", "D20", "C5", "D22", 
    "T7", "TP7", "D25", "CP5", "D27", "CP3", "P5", "D30", "P7", "P9"
]
RUN = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5', 'Run6', 'Run7', 'Run8', 'Run9', 'Run10',
       'Run11', 'Run12', 'Run13', 'Run14', 'Run15', 'Run16', 'Run17', 'Run18', 'Run19', 'Run20',]

WORD_MAP = set()
WORD_VEC = []
ONSET_TIME = []
OFFSET_TIME = []
FS = 128

for run in RUN:
    data = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/Stimuli/Text/{run}.mat')
    wordVec = data['wordVec']
    onset_time = data['onset_time']
    offset_time = data['offset_time']

    for word in wordVec:
        WORD_MAP.add(word[0][0])

    onset_indices = (onset_time * FS).astype(int)
    offset_indices = (offset_time * FS).astype(int)

    ONSET_TIME.append(onset_indices)
    OFFSET_TIME.append(offset_indices)
    WORD_VEC.append(wordVec)

WORD_MAP = list(WORD_MAP)
WORD_MAP = np.array(WORD_MAP)
WORD_MAP = np.unique(WORD_MAP)
WORD_MAP = WORD_MAP.tolist()

def proc_one(sub, run):
    # Original MAT: eegData, fs, mastoids
    data = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/EEG/{sub}/{sub}_{run}.mat')
    eegData = np.array(data['eegData'])
    wordVec = np.array(WORD_VEC[RUN.index(run)])
    mastoids_data = data['mastoids']
    onset_time = np.array(ONSET_TIME[RUN.index(run)])
    offset_time = np.array(OFFSET_TIME[RUN.index(run)])
    eegData = eegData[onset_time[0][0]:offset_time[-1][0], :]
    eegData = np.transpose(eegData)
    eegData = np.expand_dims(eegData, axis=0)

    mne_epoch = mne.EpochsArray(eegData, mne.create_info(ch_names=CH_NAMES, sfreq=FS, ch_types='eeg'))
    mne_epoch.filter(l_freq=1, h_freq=40, verbose=False)
    data = mne_epoch.get_data().astype(np.float32)
    data = data.squeeze(axis=0)

    # Slice the eegData regarding the onset and offset time
    # Also expand the Y data | Y data should also map to the WORD_MAP
    X, Y = [], []
    for idx, word in enumerate(wordVec):
        start = int(onset_time[idx][0]) - onset_time[0][0]
        end = int(offset_time[idx][0]) - onset_time[0][0]
        slice_length = end - start
        X.append(data[:, start:end])
        Y.extend([WORD_MAP.index(word[0])] * slice_length)

    X = np.concatenate(X, axis=1)
    Y = np.array(Y)

def proc_all():
    with mp.Pool(24) as pool:
        res = pool.map(partial(proc_one, run=run), SUBJECTS)
    return


if __name__ == '__main__':
    proc_one('Subject1', 'Run1')
    



