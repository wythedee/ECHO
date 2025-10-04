import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from functools import partial
import torch
import einops
import glob
import mne
import numpy as np
import pickle
import scipy
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'EMO')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'EMO')
NAME = 'EMO_02_SEED_IV'
SUBJECTS = ['1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_', '11_',
            '12_', '13_', '14_', '15_']

ORIGINAL_CH_NAMES = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3','POz',
    'PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

L_FREQ = 0.3
H_FREQ = 50
RESAMPLE_RATE = 250
TIME_LENGTH = 4
# remove CB1 and CB2
CH_NAMES = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3','POz',
    'PO4','PO6','PO8','O1','Oz','O2']

ZONES = {
    'Frontal': ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8'],
    'Central': ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
    'Temporal': ['T7', 'T8', 'TP7', 'TP8'],
    'Parietal': ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    'Occipital': ['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
}

session_labels = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

EMO_SEED_IV = META(NAME, CH_NAMES, SUBJECTS, ['EMO/Neutral', 'EMO/Sad', 'EMO/Fear', 'EMO/Happy'], resample_rate=RESAMPLE_RATE, time_length=TIME_LENGTH)

def proc_one(sub):
    try:
        print(f"==> [START] Processing subject: {sub}", flush=True)
        src_sfreq = 200
        X, Y = [], []
        for session in [1, 2, 3]:
            path = f'{SRC_FOLDER}/{NAME}/eeg_raw_data/{session}/{sub}*.mat'
            # print(path)
            fn = list(glob.glob(path))[0]
            print(fn)
            sess_label = session_labels[session]
            data = scipy.io.loadmat(fn, squeeze_me=True)
            prefix = list(data.keys())[-1].split('_')[0]
            for i in range(1, 25):
                x = data[f'{prefix}_eeg{i}']
                # print(x.shape)
                raw = mne.io.RawArray(x, mne.create_info(ch_names=ORIGINAL_CH_NAMES, sfreq=src_sfreq, 
                                                        ch_types='eeg'), verbose=False)
                raw.drop_channels(['CB1', 'CB2'])
                raw = raw.resample(RESAMPLE_RATE)
                raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
                x = raw.get_data().astype(np.float32)
                x = torch.tensor(x).unfold(1, RESAMPLE_RATE*TIME_LENGTH, RESAMPLE_RATE*TIME_LENGTH)
                x = einops.rearrange(x, 'C N T -> N C T').numpy()
                y = np.array(sess_label[i-1]).repeat(x.shape[0]).astype(np.uint8)
                # print(session, i, x.shape, y.shape, len(CH_NAMES))
                X.append(x);Y.append(y)
        X, Y = np.concatenate(X), np.concatenate(Y)
        X = pipeline(X, CH_NAMES)
        print(f"<== [SUCCESS] Finished subject: {sub}. Final shape: X={X.shape}, Y={Y.shape}", flush=True)
        return sub, X, Y
    except Exception as e:
        print(f'Error processing {sub}: {e}')
        return sub, None, None

def proc_all():
    results_dict = {}
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        process_iterator = pool.imap_unordered(proc_one, SUBJECTS)
 
        print("Starting data preprocessing...")
        for sub, X, Y in tqdm(process_iterator, total=len(SUBJECTS), desc="Processing Subjects"):
            results_dict[sub] = (X, Y)

    # for sub in tqdm(SUBJECTS, desc="Processing Subjects"):
    #     sub, X, Y = proc_one(sub)
    #     results_dict[sub] = (X, Y)

    res = []
    for sub in SUBJECTS:
        if sub in results_dict:
            X, Y = results_dict[sub]
            res.append((sub, X, Y))

    print("preprocess complete: ", len(res))
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            if X is None:
                print(f'Skipping {sub} due to error in processing.')
                continue
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    proc_all()
    # proc_one(SUBJECTS[0])