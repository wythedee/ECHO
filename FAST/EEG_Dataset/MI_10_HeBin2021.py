import os
import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
import h5py
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')
NAME = 'MI_HeBin2021'
NAME_LR = 'MI_10_HeBin2021_LR'
NAME_UD = 'MI_10_HeBin2021_UD'

# 'S34' don't have full 450 data
SUBJECTS = [
    'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14',
    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27',
    'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40',
    'S41', 'S42', 'S43', 'S44', 'S45', 'S46', 'S47', 'S48', 'S49', 'S50', 'S51', 'S52', 'S53',
    'S54', 'S55', 'S56', 'S57', 'S58', 'S59']

ORIGINAL_CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
            'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
            'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1',
            'O1', 'Oz', 'O2', 'CB2']

CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
            'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
            'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']

# Task number is used to identify the individual BCI tasks (1 = 'LR', 2 = 'UD', 3 = '2D').
# Target number is used to identify which target was presented to the participants (1 = right, 2 = left, 3 = up, 4 = down).
# This dataset is too large
MI_HeBin2021_LR = META(NAME_LR, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=5)
MI_HeBin2021_UD = META(NAME_UD, CH_NAMES, SUBJECTS, ['MI/Up', 'MI/Down'], resample_rate=250, time_length=5)

# trials can be from 5 sec to 11 sec, first 2 sec is baseline
# (array([ 5,  6,  7,  8,  9, 10, 11]), array([  6,  44,  62,  41,  47,  32, 218]))
start, end = 1, 6

def proc_one_mat(fn):
    BCI = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=True)['BCI']
    trial_buf, label_buf = [], []
    for i in range(450):
        trial = BCI.data[i]
        targetnumber = BCI.TrialData[i].targetnumber
        if trial.shape[1] < end*1000:
            trial = np.pad(trial, ((0, 0), (0, end*1000-trial.shape[1])), 'edge')
        x = trial[:, start*1000:end*1000]
        trial_buf.append(x)
        label_buf.append(targetnumber)
    trial_buf, label_buf = np.array(trial_buf), np.array(label_buf).astype(np.int32)
    epoch = mne.EpochsArray(trial_buf, mne.create_info(ORIGINAL_CH_NAMES, 1000, 'eeg'))
    epoch.drop_channels(['CB1', 'CB2'])
    epoch = epoch.resample(250)
    epoch = epoch.filter(1, 40)
    x = epoch.get_data().astype(np.float32)
    y = label_buf.astype(np.uint8) - 1
    print(x.shape, y.shape, np.unique(y, return_counts=True))
    return x, y

def proc_one(sub):
    fn_npz = f'{CACHE_FOLDER}/{sub}.npz'
    if os.path.exists(fn_npz):
        print(f'{fn_npz} exists')
        return

    X, Y = [], []
    for i in range(1, 12):
        fn = f'{SRC_FOLDER}/{NAME}/{sub}_Session_{i}.mat'
        if not os.path.exists(fn):
            print(f'{fn} not exists')
            continue
        x, y = proc_one_mat(fn)
        X.append(x); Y.append(y)
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)

    # (1 = right, 2 = left, 3 = up, 4 = down)
    mask0_1 = (Y == 0) | (Y == 1) # LR
    mask2_3 = (Y == 2) | (Y == 3) # UD

    # here we swap the label for left and right
    X_LR, Y_LR = X[mask0_1], 1 - Y[mask0_1]
    X_UD, Y_UD = X[mask2_3], Y[mask2_3] - 2
    del X, Y

    np.savez(fn_npz, X_LR=X_LR, Y_LR=Y_LR, X_UD=X_UD, Y_UD=Y_UD)
    print(sub, '-LR-', X_LR.shape, Y_LR.shape, np.unique(Y_LR, return_counts=True))
    print(sub, '-UD-', X_UD.shape, Y_UD.shape, np.unique(Y_UD, return_counts=True))

def proc_all():
    with mp.Pool(32) as pool:
        pool.map(proc_one, SUBJECTS)

    f_LR = h5py.File(f'{DATA_FOLDER}/{NAME_LR}.h5', 'w')
    f_UD = h5py.File(f'{DATA_FOLDER}/{NAME_UD}.h5', 'w')
    for sub in SUBJECTS:
        fn = f'{CACHE_FOLDER}/{sub}.npz'
        with np.load(fn) as data:
            f_LR.create_dataset(f'{sub}/X', data=data['X_LR'])
            f_LR.create_dataset(f'{sub}/Y', data=data['Y_LR'])
            print(sub, '-LR-', data['X_LR'].shape, data['Y_LR'].shape, np.unique(data['Y_LR'], return_counts=True))

            f_UD.create_dataset(f'{sub}/X', data=data['X_UD'])
            f_UD.create_dataset(f'{sub}/Y', data=data['Y_UD'])
            print(sub, '-UD-', data['X_UD'].shape, data['Y_UD'].shape, np.unique(data['Y_UD'], return_counts=True))
        # os.remove(fn)
    f_LR.close()
    f_UD.close()

if __name__ == '__main__':
    CACHE_FOLDER = '/home/workspace/EEG_Standardized_Group_new/cache'
    os.system(f'rm -rf {CACHE_FOLDER}')
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    proc_all()
