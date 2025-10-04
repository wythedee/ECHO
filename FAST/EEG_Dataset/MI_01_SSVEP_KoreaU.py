import os
import mne
import numpy as np
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
from functools import partial
import h5py

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')

SRC_NAME = 'MI_KoreaU'
NAME_MI = 'MI_01_KoreaU'
NAME_SSVEP = 'SSVEP_01_KoreaU'
CH_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 
    'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 
    'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 
    'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 
    'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 
    'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 
    'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
]

# # 根据脑区域对电极进行分类
# ZONES = {
#     'Pre-frontal': ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8'],
#     'Frontal': ['F7', 'F3', 'Fz', 'F4', 'F8', 'F9', 'F10'],
#     'Precentral': ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
#     'Central': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz'],
#     'Postcentral': ['CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz'],
#     'Parietal': ['P1', 'P2', 'P3', 'P4', 'P7', 'P8', 'Pz'],
#     'Temporal': ['T7', 'T8', 'TP7', 'TP8', 'TP9', 'TP10', 'TTP7h', 'TPP9h', 'TPP8h', 'TPP10h', 'FT9', 'FT10', 'FTT9h', 'FTT10h'],
#     'Parietal-Occipital Junction': ['PO3', 'PO4', 'PO9', 'PO10', 'POz'],
#     'Occipital': ['O1', 'O2', 'Oz'],
# }

SUBJECTS = [
    's13', 's22', 's28', 's5', 's31', 's46', 's41', 's36', 's8', 
    's2', 's52', 's25', 's17', 's19', 's10', 's21', 's6', 's45', 
    's32', 's38', 's35', 's42', 's48', 's1', 's26', 's51', 's14', 
    's24', 's53', 's16', 's37', 's40', 's3', 's9', 's4', 's47', 
    's30', 's11', 's29', 's54', 's23', 's50', 's27', 's15', 's49', 
    's43', 's34', 's7', 's39', 's33', 's44', 's12', 's18', 's20'
] 

MI_KoreaU = META(NAME_MI, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=4)
SSVEP_KoreaU = META(NAME_SSVEP, CH_NAMES, SUBJECTS, ['f/05.4', 'f/06.6', 'f/08.6', 'f/12.0'], resample_rate=250, time_length=4)

def proc_one(sub, task):
    try:
        sess1 = scipy.io.loadmat(f'{SRC_FOLDER}/{SRC_NAME}/BCI_dataset/DB_mat/session1/{sub}/EEG_{task}.mat')
        sess2 = scipy.io.loadmat(f'{SRC_FOLDER}/{SRC_NAME}/BCI_dataset/DB_mat/session2/{sub}/EEG_{task}.mat')
    except:
        print(f'{SRC_FOLDER}/{SRC_NAME}/BCI_dataset/DB_mat/session1/{sub}/EEG_{task}.mat not exists')
        print(f'{SRC_FOLDER}/{SRC_NAME}/BCI_dataset/DB_mat/session2/{sub}/EEG_{task}.mat not exists')
        return sub, None, None

    d1, l1 = np.transpose(sess1[f'EEG_{task}_train']['smt'][0,0], (1, 2, 0)), sess1[f'EEG_{task}_train']['y_dec'][0,0][0]
    d2, l2 = np.transpose(sess1[f'EEG_{task}_test']['smt'][0,0], (1, 2, 0)),  sess1[f'EEG_{task}_test']['y_dec'][0,0][0]
    d3, l3 = np.transpose(sess2[f'EEG_{task}_train']['smt'][0,0], (1, 2, 0)), sess2[f'EEG_{task}_train']['y_dec'][0,0][0]
    d4, l4 = np.transpose(sess2[f'EEG_{task}_test']['smt'][0,0], (1, 2, 0)),  sess2[f'EEG_{task}_test']['y_dec'][0,0][0]
    print(sub, d1.shape, l1.shape, d2.shape, l2.shape, d3.shape, l3.shape, d4.shape, l4.shape)
    epoch = np.concatenate([d1, d2, d3, d4], axis=0)
    label = np.concatenate([l1, l2, l3, l4], axis=0)
    mne_epo = mne.EpochsArray(epoch, mne.create_info(ch_names=CH_NAMES, sfreq=1000, ch_types='eeg'), verbose=False)
    print("Filtering data...")
    mne_epo.filter(l_freq=1, h_freq=40, verbose=False)
    print("Resampling data...")
    mne_epo.resample(250, npad='auto', verbose=False)
    print("getting x")
    x = mne_epo.get_data(copy=False).astype(np.float32)
    y = label - 1

    ################ Special Notice ################
    # KoreaU MI dataset 0 -> right , 1 -> left
    if task == 'MI':
        y = 1 - y

    print(sub, task, x.shape, label.shape)
    x = pipeline(x, CH_NAMES)
    del epoch, label, mne_epo, d1, d2, d3, d4, l1, l2, l3, l4, sess1, sess2
    return sub, x, y

def proc_all(task):
    assert task in ['MI', 'SSVEP', 'ERP']
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(partial(proc_one, task=task), SUBJECTS)
    # res = []
    # for sub in SUBJECTS:
    #     print(f"==> [START] Processing subject: {sub}", flush=True)
    #     res.append(proc_one(sub, task))
    with h5py.File(f'{DATA_FOLDER}/{task}/{task}_01_KoreaU.h5', 'w') as f:
        for sub, X, Y in res:
            if X is None:
                continue
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    proc_all('MI')
    proc_all('SSVEP')
