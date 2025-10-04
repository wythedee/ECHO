from functools import partial
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

SRC_FOLDER = os.path.join(SRC_FOLDER, 'CS')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'CS')
NAME_CS = 'CS_01_Muyun2024'
NAME_OS = 'OS_01_Muyun2024'

SUBJECTS = ['S0009', 'S0011', 'S0012', 'S0013', 'S0014', 'S0015', 'S0016', 'S0017',
            'S0018', 'S0020', 'S0021', 'S0022', 'S0023', 'S0025', 'S0026', 'S0027',
            'S0028', 'S0029', 'S0030', 'S0031', 'S0032', 'S0033', 'S0034', 'S0035',
            'S0036', 'S0037', 'S0038', 'S0039', 'S0040', 'S0041', 'S0042', 'S0043',
            'S0044', 'S0045', 'S0046', 'S0047', 'S0048', 'S0049', 'S0050', 'S0051',
            'S0052', 'S0053', 'S0054', 'S0055', 'S0056', 'S0057', 'S0058', 'S0059',
            'S0060', 'S0061', 'S0062', 'S0063', 'S0064', 'S0065', 'S0066', 'S0067', 'S0068']

CH_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 
    'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 
    'CP6', 'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 
    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7',
    'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz'
]

CS_Muyun2024 = META(NAME_CS, CH_NAMES, SUBJECTS, ['CS/Go-there', 'CS/Distract-target', 'CS/Follow-me', 'CS/Explore-here', 'CS/Terminate'])
OS_Muyun2024 = META(NAME_OS, CH_NAMES, SUBJECTS, ['OS/Go-there', 'OS/Distract-target', 'OS/Follow-me', 'OS/Explore-here', 'OS/Terminate'])

def proc_one(sub, task):
    with h5py.File(f'{SRC_FOLDER}/CS_01_OS_Muyun2024/{task}.h5', 'r') as f:
        X = f[f'{sub}/data'][()]
        Y = f[f'{sub}/label'][()]
        head = f[f'{sub}/head'][()]
        tail = f[f'{sub}/tail'][()]
        X = np.concatenate([head, X, tail], axis=-1)
    X = pipeline(X, CH_NAMES)
    print(sub, X.shape, Y.shape)
    return sub, X, Y

def proc_all(task, save_name):
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(partial(proc_one, task=task), SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{save_name}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    import gc
    proc_all('S_1hz_40hz_ICA2_RE', NAME_CS)
    gc.collect()
    proc_all('L_1hz_40hz_ICA2_RE', NAME_OS)
