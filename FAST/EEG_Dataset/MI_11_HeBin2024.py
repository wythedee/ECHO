import glob
import os
import mne
import numpy as np
import multiprocessing as mp
import multiprocessing.dummy as dmp
import h5py
import mat73
import warnings
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')
# original dataset folder name
SRC_NAME = 'MI_HeBin2024'

# keep: full records name (not used)
FILE_NAME = 'MI_11_HeBin2024'
NAME_LR   = 'MI_11_HeBin2024_LR'       # left-right tasks only
NAME_UD   = 'MI_11_HeBin2024_UD'       # up-down tasks only

SUBJECTS = [
    'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 
    'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 
    'S21', 'S22', 'S23', 'S24', 'S25']

ORIGINAL_CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 
    'Cz', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
    'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 
    'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 
    'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2', 
]

MI_HeBin2024_LR = META(NAME_LR, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=10)
MI_HeBin2024_UD = META(NAME_UD, CH_NAMES, SUBJECTS, ['MI/Up', 'MI/Down'], resample_rate=250, time_length=10)

def proc_one_mat(fn):
    """Load one .mat file and preprocess each event segment.

    Steps:
    1. Slice raw (1000 Hz) by event (start, end) to keep index correct.
    2. For each slice: drop extra channels → band-pass → resample to 250 Hz.
    3. Return X with shape (n_trial, n_channel, n_sample) and its labels.
    """

    data = mat73.loadmat(fn)['eeg']

    # Raw 64-ch data (µV → V)
    X_all = data['data'].astype(np.float32)[:64] / 1e6
    sfreq_src = 1000  # original sampling rate

    segments = []
    for evt in data['event']:
        start = int(evt['latency'])
        end = start + int(evt['duration'])

        # Cut first to avoid resample index shift
        seg = X_all[:, start:end]

        info = mne.create_info(ch_names=ORIGINAL_CH_NAMES, sfreq=sfreq_src, ch_types='eeg')
        raw = mne.io.RawArray(seg, info, verbose=False)

        # Preprocess
        raw = raw.drop_channels(['CB1', 'CB2', 'M1', 'M2'])  # drop extra refs
        raw = raw.filter(0.3, 50, filter_length='auto', verbose=False)
        raw = raw.resample(250, npad='auto', verbose=False)

        seg_proc = raw.get_data().astype(np.float32)  # (ch, time)

        # --- enforce fixed length 10 s (2500 samples) ---
        target_len = 250 * 10  # 10 s @ 250 Hz
        cur_len = seg_proc.shape[1]
        if cur_len < target_len:
            last_val = seg_proc[:, -1:]                           # (chan, 1)
            pad = np.repeat(last_val, target_len - cur_len, axis=1)  # (chan, pad_len)
            seg_proc = np.concatenate([seg_proc, pad], axis=1)
        elif cur_len > target_len:
            seg_proc = seg_proc[:, :target_len]

        segments.append(seg_proc)

    X = np.stack(segments, axis=0)  # (trial, channel, time)
    labels = data['targets'].astype(np.int64)
    return X, labels

def proc_one(sub):
    """Process all .mat files for one subject, split by LR/UD task types."""

    X_LR_buf, Y_LR_buf = [], []   # LR: 1=Right,2=Left → final 0=Left,1=Right
    X_UD_buf, Y_UD_buf = [], []   # UD: 1=Up,2=Down → final 0=Up,1=Down

    for fn in glob.glob(f'{SRC_FOLDER}/{SRC_NAME}/{sub}/*.mat'):
        fn_lower = fn.lower()
        # determine task type from filename (lr/ud/2d)
        task = None
        if 'lr' in fn_lower:
            task = 'LR'
        elif 'ud' in fn_lower:
            task = 'UD'
        elif '2d' in fn_lower:
            task = '2D'

        if task is None:
            raise ValueError(f'Unknown task type: {fn}')
            continue

        X_seg, y_raw = proc_one_mat(fn)  # y_raw keeps 1,2,(3,4)

        if task == 'LR':
            # 1:Right,2:Left → convert to 0/1 and swap to match [Left,Right]
            y_mapped = 1 - (y_raw - 1)              # Right→1, Left→0
            X_LR_buf.append(X_seg)
            Y_LR_buf.append(y_mapped)

        elif task == 'UD':
            # 1:Up,2:Down → 0/1
            y_mapped = y_raw - 1                    # Up→0, Down→1
            X_UD_buf.append(X_seg)
            Y_UD_buf.append(y_mapped)

        elif task == '2D':
            # 2D data contains both LR and UD, need to split
            # --- LR part ---
            mask_lr = (y_raw == 1) | (y_raw == 2)
            if np.any(mask_lr):
                y_lr = y_raw[mask_lr]
                y_lr = 1 - (y_lr - 1)              # same: Right→1, Left→0
                X_LR_buf.append(X_seg[mask_lr])
                Y_LR_buf.append(y_lr)

            # --- UD part ---
            mask_ud = (y_raw == 3) | (y_raw == 4)
            if np.any(mask_ud):
                y_ud = y_raw[mask_ud] - 3          # 3→0(Up),4→1(Down)
                X_UD_buf.append(X_seg[mask_ud])
                Y_UD_buf.append(y_ud)

        print(f'Processed {fn} | task={task}')

    # concatenate buffers
    X_LR = np.concatenate(X_LR_buf, axis=0) if X_LR_buf else np.empty((0, len(CH_NAMES), 250*10), dtype=np.float32)
    Y_LR = np.concatenate(Y_LR_buf, axis=0).astype(np.uint8) if Y_LR_buf else np.empty((0,), dtype=np.uint8)

    X_UD = np.concatenate(X_UD_buf, axis=0) if X_UD_buf else np.empty((0, len(CH_NAMES), 250*10), dtype=np.float32)
    Y_UD = np.concatenate(Y_UD_buf, axis=0).astype(np.uint8) if Y_UD_buf else np.empty((0,), dtype=np.uint8)

    # normalization & channel mapping
    if X_LR.size:
        X_LR = pipeline(X_LR, CH_NAMES)
    if X_UD.size:
        X_UD = pipeline(X_UD, CH_NAMES)

    # print info for debugging
    print(sub, '-LR-', X_LR.shape, Y_LR.shape, np.unique(Y_LR, return_counts=True) if Y_LR.size else 'empty')
    print(sub, '-UD-', X_UD.shape, Y_UD.shape, np.unique(Y_UD, return_counts=True) if Y_UD.size else 'empty')

    return sub, X_LR, Y_LR, X_UD, Y_UD

if __name__ == '__main__':
    # silence warnings / info
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    mne.set_log_level('WARNING')

    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(proc_one, SUBJECTS)

    # write LR and UD h5 files separately
    f_LR = h5py.File(f'{DATA_FOLDER}/{NAME_LR}.h5', 'w')
    f_UD = h5py.File(f'{DATA_FOLDER}/{NAME_UD}.h5', 'w')

    for sub, X_LR, Y_LR, X_UD, Y_UD in res:
        if Y_LR.size:
            f_LR.create_dataset(f'{sub}/X', data=X_LR)
            f_LR.create_dataset(f'{sub}/Y', data=Y_LR)
            print(sub, '-LR-', X_LR.shape, Y_LR.shape, np.unique(Y_LR, return_counts=True))

        if Y_UD.size:
            f_UD.create_dataset(f'{sub}/X', data=X_UD)
            f_UD.create_dataset(f'{sub}/Y', data=Y_UD)
            print(sub, '-UD-', X_UD.shape, Y_UD.shape, np.unique(Y_UD, return_counts=True))

    f_LR.close()
    f_UD.close()

    # inspect written files
    print('\n=== LR Dataset ===')
    with h5py.File(f'{DATA_FOLDER}/{NAME_LR}.h5', 'r') as fi:
        for sub in sorted(fi.keys()):
            X = fi[f'{sub}/X'][:]
            Y = fi[f'{sub}/Y'][:]
            unique, counts = np.unique(Y, return_counts=True)
            print(f'{sub}: X{X.shape} Y{Y.shape} labels{list(zip(unique, counts))}')

    print('\n=== UD Dataset ===')
    with h5py.File(f'{DATA_FOLDER}/{NAME_UD}.h5', 'r') as fi:
        for sub in sorted(fi.keys()):
            X = fi[f'{sub}/X'][:]
            Y = fi[f'{sub}/Y'][:]
            unique, counts = np.unique(Y, return_counts=True)
            print(f'{sub}: X{X.shape} Y{Y.shape} labels{list(zip(unique, counts))}')

    print('\n=== Summary ===')
    with h5py.File(f'{DATA_FOLDER}/{NAME_LR}.h5', 'r') as fi:
        total_lr = sum(fi[f'{sub}/X'].shape[0] for sub in fi.keys())
        print(f'LR total trials: {total_lr}')
    with h5py.File(f'{DATA_FOLDER}/{NAME_UD}.h5', 'r') as fi:
        total_ud = sum(fi[f'{sub}/X'].shape[0] for sub in fi.keys())
        print(f'UD total trials: {total_ud}')