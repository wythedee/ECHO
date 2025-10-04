from functools import partial
import torch
import einops
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

SRC_FOLDER = os.path.join(SRC_FOLDER, 'EMO')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'EMO')
NAME = 'EMO_01_DEAP'
SUBJECTS = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 
            's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30', 
            's31', 's32']
CH_NAMES = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
        'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
        'CP2', 'P4', 'P8', 'PO4', 'O2']
ZONES = {
    'Frontal': ['Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'F7', 'F8', 'FC5', 'FC1', 'FC2', 'FC6'],
    'Central': ['C3', 'Cz', 'C4'],
    'Temporal': ['T7', 'T8'],
    'Parietal': ['CP5', 'CP1', 'CP2', 'CP6', 'P3', 'Pz', 'P4', 'P7', 'P8'],
    'Occipital': ['PO3', 'PO4', 'O1', 'Oz', 'O2']
}

EMO_DEAP = META(NAME, CH_NAMES, SUBJECTS, ['EMO/Low-Valence', 'EMO/High-Valence'])

def proc_one(sub):
    src_sfreq = 128
    fn = f'{SRC_FOLDER}/{NAME}/{sub}.dat'
    with open(fn, 'rb') as fp:
        data = pickle.load(fp, encoding='latin1')

    x = data['data'][:, 0:32, 3 * 128:].astype(np.float32)
    # (valence, arousal, dominance, liking)
    valence = (data['labels'][:,0] > 5).astype(np.uint8)
    # arousal = (data['labels'][:,1] > 5).astype(np.uint8)
    # dominance = (data['labels'][:,2] > 5).astype(np.uint8)
    # liking = (data['labels'][:,3] > 5).astype(np.uint8)
    y = valence
    info = mne.create_info(ch_names=CH_NAMES, sfreq=src_sfreq, ch_types='eeg')
    epoched = mne.EpochsArray(x, info).resample(250)
    x = epoched.get_data().astype(np.float32)
    sfreq = 250
    x = torch.from_numpy(x).unfold(-1, sfreq*10, sfreq*10)
    # y = torch.from_numpy(y).repeat(x.shape[2]).view(-1, x.shape[2]).view(-1)
    newy = []
    for i in range(y.shape[0]):
        newy.extend([y[i]] * x.shape[2])
    y = torch.tensor(newy)
    x = einops.rearrange(x, 'B C N T -> (B N) C T')
    x, y = x.numpy(), y.numpy()
    print(x.shape, y.shape)
    x = pipeline(x, CH_NAMES)
    return sub, x, y

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