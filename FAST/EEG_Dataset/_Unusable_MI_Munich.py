import mne
import numpy as np
import os
import pickle
import scipy
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from EEG_Dataset.share import META, SRC_FOLDER, DATA_FOLDER, pipeline

NAME = 'MI_06_MunichMI'
SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']

# https://zenodo.org/records/1217449
# This is the dataset of 10 motor imagery subjects upon which the paper cited here is written. 
# It is saved in the EEGLAB .set format with the digitized electrode positions included. 
# The only issue is that the names for channels 65-128 are missing, 
# and the head model that corresponds to these electrode locations is also poorly specified (see field Headmodel of the EEG struct). 
# When using this data please cite:
# Grosse-Wentrup, Moritz, et al. "Beamforming in noninvasive brain–computer interfaces." 
# IEEE Transactions on Biomedical Engineering 56.4 (2009): 1209-1219.

CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
    'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
    'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
    'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2',
]
print(len(CH_NAMES))
exit()

MI_MunichMI = META(NAME, CH_NAMES, SUBJECTS, ['left', 'right'])

def proc_one(subject):
    data = np.load(f'{SRC_FOLDER}/{NAME}/{subject}.npz', allow_pickle=True)
    
    print(data['fs'], data['metadata'])
    print(data['x_data'].shape, data['y_data'].shape, np.unique(data['y_data'], return_counts=True))
    
    x, y = data['x_data'], data['y_data']-1
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=data['fs'], ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.filter(l_freq=1, h_freq=40, verbose=False)
    epochs.resample(250, npad='auto')
    x = epochs.get_data().astype(np.float32)
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    x = pipeline(x, CH_NAMES)
    return subject, x, y

def proc_all():
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
        
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, epoch, label in res:
            f.create_dataset(f'{sub}/X', data=epoch)
            f.create_dataset(f'{sub}/Y', data=label)
            print(sub, epoch.shape, label.shape, np.unique(label, return_counts=True))

if __name__=='__main__':
    proc_all()