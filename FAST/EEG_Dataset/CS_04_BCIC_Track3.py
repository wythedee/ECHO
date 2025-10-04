import os
import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
import scipy.signal
import multiprocessing as mp
import h5py
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'CS')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'CS')
NAME = 'CS_04_BCIC_Track3'

# Subjects are aligned with file suffixes 01..15, split into train/valid/test
# Numeric subject names 0..44 in order: train(0-14), valid(15-29), test(30-44)
SUBJECTS = [f'{i}' for i in range(45)]

# 64-channel montage consistent with Track3 imagined speech setting
CH_NAMES = [
    'Fp1','Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
    'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
    'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
    'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
    'PO3', 'POz', 'PO4', 'PO8'
]

CLASSES = ['CS/Hello', 'CS/Help-me', 'CS/Stop', 'CS/Thank-you', 'CS/Yes']
# Target sampling frequency after resampling: 250 Hz
# Target time length: 4.0 seconds (1000 samples / 250 Hz)
CS_BCIC_Track3 = META(NAME, CH_NAMES, SUBJECTS, CLASSES, resample_rate=250, time_length=4.0)

def _load_mat(path, key):
    # Train/Valid are MATLAB <7.3; Test might be 7.3 (HDF5)
    try:
        m = scipy.io.loadmat(path)
        return m[key]
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            return f[key]

def resample_data(x, original_fs, target_fs):
    """Resample EEG data from original_fs to target_fs"""
    if abs(original_fs - target_fs) < 0.1:
        # No resampling needed
        return x
    
    # Calculate resampling ratio
    resample_ratio = target_fs / original_fs
    new_length = int(x.shape[-1] * resample_ratio)
    
    print(f"Resampling from {original_fs} Hz to {target_fs} Hz (ratio: {resample_ratio:.4f})")
    print(f"Time dimension: {x.shape[-1]} -> {new_length} samples")
    
    # Resample each trial and channel
    x_resampled = np.zeros((x.shape[0], x.shape[1], new_length), dtype=x.dtype)
    
    for trial in range(x.shape[0]):
        for ch in range(x.shape[1]):
            x_resampled[trial, ch, :] = scipy.signal.resample(x[trial, ch, :], new_length)
    
    return x_resampled

def _pad_time(x, target=1000):
    """Pad or truncate time dimension to target length (default: 1000 samples for 4s at 250Hz)"""
    if x.shape[-1] < target:
        pad = target - x.shape[-1]
        x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode='edge')
    elif x.shape[-1] > target:
        x = x[..., :target]
    else:
        print("No padding or truncation needed")
    return x

def get_sampling_frequency(file_path, file_type='train'):
    """Extract sampling frequency from data files"""
    try:
        if file_type == 'train':
            # Training/validation files use scipy.io.loadmat
            data = scipy.io.loadmat(file_path)
            if 'epo_train' in data:
                fs = np.asarray(data['epo_train']['fs'])[0][0][0][0]
            elif 'epo_validation' in data:
                fs = np.asarray(data['epo_validation']['fs'])[0][0][0][0]
            else:
                return None
        else:
            # Test files use h5py
            with h5py.File(file_path, 'r') as f:
                if 'epo_test/fs' in f:
                    fs = f['epo_test/fs'][()][0][0]
                else:
                    return None
        
        return float(fs)
    except Exception as e:
        print(f"Warning: Could not extract sampling frequency from {file_path}: {e}")
        return None

def load_test_true_labels():
    """Load true labels from the answer sheet Excel file"""
    answer_file = os.path.join(SRC_FOLDER, NAME, "Test set", "Track3_Answer Sheet_Test.xlsx")
    
    if not os.path.exists(answer_file):
        print(f"Warning: Answer sheet not found at {answer_file}")
        return {}
    
    try:
        # Read the Excel file
        df = pd.read_excel(answer_file, sheet_name='Track3', header=None)
        test_labels = {}
        
        # Parse Excel structure: Data_SampleXX in row 1, True Labels in adjacent column
        for col_idx in range(df.shape[1]):
            cell_value = df.iloc[1, col_idx]
            if pd.notna(cell_value) and 'Data_Sample' in str(cell_value):
                sample_name = str(cell_value)
                label_col_idx = col_idx + 1
                
                if label_col_idx < df.shape[1]:
                    # Extract labels from row 3 onwards (row 2 is header)
                    labels = []
                    for row_idx in range(3, df.shape[0]):
                        label_value = df.iloc[row_idx, label_col_idx]
                        if pd.notna(label_value) and isinstance(label_value, (int, float)):
                            # Convert from 1-based to 0-based indexing
                            labels.append(int(label_value) - 1)
                        else:
                            break
                    
                    if labels:
                        test_labels[sample_name] = np.array(labels, dtype=np.uint8)
        
        print(f"Loaded true labels for {len(test_labels)} test files from answer sheet")
        return test_labels
        
    except Exception as e:
        print(f"Error loading answer sheet: {e}")
        return {}

def proc_one_pair(sub_suffix):
    """Return two subjects: train_{id}, valid_{id}"""
    train_path = f"{SRC_FOLDER}/{NAME}/Training set/Data_Sample{sub_suffix}.mat"
    valid_path = f"{SRC_FOLDER}/{NAME}/Validation set/Data_Sample{sub_suffix}.mat"

    data_train = scipy.io.loadmat(train_path)
    data_valid = scipy.io.loadmat(valid_path)

    x_t = np.asarray(data_train['epo_train']['x'])[0][0]
    y_t = np.asarray(data_train['epo_train']['y'])[0][0].argmax(0).astype(np.uint8)
    x_v = np.asarray(data_valid['epo_validation']['x'])[0][0]
    y_v = np.asarray(data_valid['epo_validation']['y'])[0][0].argmax(0).astype(np.uint8)

    # Get original sampling frequency
    original_fs = float(np.asarray(data_train['epo_train']['fs'])[0][0][0][0])
    target_fs = 250.0

    x_t = np.transpose(x_t, (2, 1, 0)).astype(np.float32)
    x_v = np.transpose(x_v, (2, 1, 0)).astype(np.float32)
    
    # Apply resampling and padding
    print(f"Processing Data_Sample{sub_suffix} (Train/Valid)")
    x_t = resample_data(x_t, original_fs, target_fs)
    x_v = resample_data(x_v, original_fs, target_fs)
    x_t = _pad_time(x_t)
    x_v = _pad_time(x_v)

    x_t = pipeline(x_t, CH_NAMES)
    x_v = pipeline(x_v, CH_NAMES)

    return (x_t, y_t), (x_v, y_v)

def _try_read_test_labels(f):
    # Attempt to read one-hot labels for test set; if not present, return None
    try:
        y = f['epo_test/y']
        arr = y[()]
        # Direct one-hot (5, N)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] in (5,):
            return arr.argmax(0).astype(np.uint8)
        # Reference-like: not reliable across generators; fall back
        return None
    except Exception:
        return None

def proc_one_test(sub_suffix, true_labels_dict=None):
    """Process one test file, using true labels from answer sheet if available"""
    test_path = f"{SRC_FOLDER}/{NAME}/Test set/Data_Sample{sub_suffix}.mat"
    sample_name = f"Data_Sample{sub_suffix}"
    
    with h5py.File(test_path, 'r') as f:
        x = f['epo_test/x'][()]
        # Get original sampling frequency
        original_fs = float(f['epo_test/fs'][()][0][0])
        # Try to read labels from .mat file (for backward compatibility)
        y_idx_mat = _try_read_test_labels(f)
    
    target_fs = 250.0
    x = np.transpose(x, (0, 1, 2)).astype(np.float32)
    
    # Apply resampling and padding
    print(f"Processing {sample_name} (Test)")
    x = resample_data(x, original_fs, target_fs)
    x = _pad_time(x)
    x = pipeline(x, CH_NAMES)
    
    # Priority: 1) True labels from answer sheet, 2) .mat file labels, 3) Unknown labels
    if true_labels_dict and sample_name in true_labels_dict:
        y_idx = true_labels_dict[sample_name]
        print(f"Using TRUE labels for {sample_name}: {len(y_idx)} samples")
        # Verify label count matches data
        if len(y_idx) != x.shape[0]:
            print(f"Warning: Label count ({len(y_idx)}) doesn't match sample count ({x.shape[0]})")
            y_idx = np.full((x.shape[0],), 255, dtype=np.uint8)
    elif y_idx_mat is not None and len(y_idx_mat) == x.shape[0]:
        y_idx = y_idx_mat
        print(f"Using .mat file labels for {sample_name}: {len(y_idx)} samples")
    else:
        y_idx = np.full((x.shape[0],), 255, dtype=np.uint8)  # 255 means unknown label
        print(f"No valid labels found for {sample_name}, using 255 (unknown)")
    
    return x, y_idx

def proc_all():
    # Verify sampling frequency
    print("Verifying sampling frequency...")
    train_file = f"{SRC_FOLDER}/{NAME}/Training set/Data_Sample01.mat"
    test_file = f"{SRC_FOLDER}/{NAME}/Test set/Data_Sample01.mat"
    
    train_fs = get_sampling_frequency(train_file, 'train')
    test_fs = get_sampling_frequency(test_file, 'test')
    
    print(f"Training file sampling frequency: {train_fs} Hz")
    print(f"Test file sampling frequency: {test_fs} Hz")
    
    if train_fs and test_fs and abs(train_fs - test_fs) > 0.1:
        print(f"Warning: Sampling frequency mismatch between train ({train_fs}) and test ({test_fs})")
    
    expected_fs = 256.0
    if train_fs and abs(train_fs - expected_fs) > 0.1:
        print(f"Warning: Expected {expected_fs} Hz, but found {train_fs} Hz")
    
    # Load true labels from answer sheet
    print("Loading test set true labels from answer sheet...")
    true_labels = load_test_true_labels()
    
    # Process training and validation sets
    with mp.Pool(15) as pool:
        res_tv = pool.map(proc_one_pair, [f'{i:02d}' for i in range(1, 16)])
    
    # Process test sets with true labels
    print("Processing test sets...")
    res_t = []
    for i in range(1, 16):
        sub_suffix = f'{i:02d}'
        x, y = proc_one_test(sub_suffix, true_labels)
        res_t.append((x, y))
    
    # Save all data to HDF5
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        f.create_dataset('subjects', data=np.array(SUBJECTS, dtype='S'))
        # train: subject 0..14; valid: 15..29
        for i, pair in enumerate(res_tv):
            x_t, y_t = pair[0]
            x_v, y_v = pair[1]
            f.create_dataset(f'{i}/X', data=x_t)
            f.create_dataset(f'{i}/Y', data=y_t)
            f.create_dataset(f'{15+i}/X', data=x_v)
            f.create_dataset(f'{15+i}/Y', data=y_v)
        
        # test: 30..44
        for i, (X, Y) in enumerate(res_t):
            f.create_dataset(f'{30+i}/X', data=X)
            f.create_dataset(f'{30+i}/Y', data=Y)
    
    # Print summary
    total_true_labels = sum(1 for _, y in res_t if not np.all(y == 255))
    print(f"\nSummary:")
    print(f"- Training sets: 15 files processed")
    print(f"- Validation sets: 15 files processed") 
    print(f"- Test sets: 15 files processed")
    print(f"- Test files with TRUE labels: {total_true_labels}/15")
    if true_labels:
        total_samples = sum(len(labels) for labels in true_labels.values())
        print(f"- Total test samples with true labels: {total_samples}")
        unique_labels = np.unique(np.concatenate(list(true_labels.values())))
        print(f"- Test label range: {unique_labels}")
    print(f"- Output saved to: {DATA_FOLDER}/{NAME}.h5")

if __name__ == '__main__':
    proc_all()



