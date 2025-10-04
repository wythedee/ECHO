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

SRC_FOLDER = os.path.join(SRC_FOLDER, 'CS')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'CS')
NAME = 'CS_03_Large_Spanish_EEG'
# 'sub-009' is only 16 trials, so we exclude it
SUBJECTS = ['002','003','005','006','008','010','011','012','013','014','015','016',
            '017','018','019','021','022','023','024','025','026','027','028','029',
            '030','031','032','033','034','035','036','037','038','039','040','041',
            '042','043','044','046','047','048','049','050','051','052','053','055',
            '057','058','059','060','061','062','063']

CH_NAMES = ['Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6',
            'F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5',
            'C3','C1','Cz','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPz',
            'CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','Pz','P2','P4','P6',
            'P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','O1','Oz','O2']

CLASSES = [f'CS/Spanish{i:02d}' for i in range(30)]
CS_Large_Spanish_EEG = META(NAME, CH_NAMES, SUBJECTS, CLASSES)

def proc_one(sub, task):
    assert task in ['perception', 'production'], f'{task} not in [perception, production]'
    tsv_file = f'{SRC_FOLDER}/{NAME}/sub-{sub}/ses-01/eeg/sub-{sub}_ses-01_task-sentences_events.tsv'
    edf_file = f'{SRC_FOLDER}/{NAME}/sub-{sub}/ses-01/eeg/sub-{sub}_ses-01_task-sentences_eeg.edf'
    tsv_data = pd.read_csv(tsv_file, sep='\t')
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.drop_channels(['CB1', 'CB2', 'HEO', 'VEO', 'EKG', 'EMG', 'Trigger'])
    raw = raw.filter(1, 40, verbose=False)
    # resample from 1000Hz to 250Hz
    raw = raw.resample(250, npad='auto')
    data = raw.get_data().astype(np.float32)

    X, Y = [], []
    for idx, event in tsv_data.iterrows():
        if event.trial_type.startswith(task):
            start = int(event['sample']/4) - 250
            end = int(event['sample']/4) + 5*250 + 250
            X.append(data[:, start:end])
            _, label = event.trial_type.split('_')
            Y.append(int(label)-1)
    X, Y = np.array(X), np.array(Y)
    X = pipeline(X, CH_NAMES)
    print(sub, task, X.shape, Y.shape)
    return sub, X, Y

import gc
def proc_all():
    for task in ['perception', 'production']:
        with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
            res = pool.map(partial(proc_one, task=task), SUBJECTS)
        with h5py.File(f'{DATA_FOLDER}/{NAME}_{task}.h5', 'w') as f:
            for sub, X, Y in res:
                f.create_dataset(f'{sub}/X', data=X)
                f.create_dataset(f'{sub}/Y', data=Y)
                print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
        gc.collect()
            
if __name__ == '__main__':
    proc_all()