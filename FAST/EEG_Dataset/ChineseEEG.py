import os
import sys
import h5py
import mne
import numpy as np
import pandas as pd
import multiprocessing as mp
import multiprocessing.dummy as dmp
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

NAME = 'ChineseEEG'
subjects = ['sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-13', 'sub-14', 'sub-15']
ch_map = pd.read_csv(f'EEG_Montage/EGI_to_1005.csv')
CH_NAMES = ch_map['1005_ch'].values

def proc_one(subject):
    x = np.load(f'{SRC_FOLDER}/ChineseEEG/James/epochs/{subject}.npy')
    print(f'Load {subject} shape: {x.shape}')
    x = pipeline(x, CH_NAMES)
    return subject, x

if __name__ == '__main__':
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
            res = pool.map(proc_one, subjects)
            for subject, x in res:
                f.create_dataset(f'{subject}/X', data=x)
                print(f'{subject} shape: {x.shape}')