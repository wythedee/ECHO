import os
import mne
import h5py
import numpy as np
import sys
sys.path.append(os.path.abspath('.'))
from EEG_Montage.AdaptiveGrouping import AdaptiveGrouping

def find_available_path(folder_list):
    for folder in folder_list:
        if os.path.exists(folder):
            return folder
    raise FileNotFoundError(f'None of the given path exists {str(folder_list)}')

THREADS = 100
SRC_FOLDER = '/path/to/your/dataset_root'
DATA_FOLDER = SRC_FOLDER

def pipeline(X, ch_names):
    X = np.clip(X, np.percentile(X, 0.1), np.percentile(X, 99.9))
    median = np.median(X)
    q25 = np.percentile(X, 25)
    q75 = np.percentile(X, 75)
    iqr = q75 - q25
    # eps = 1e-6
    # X = (X - median) / (iqr + eps)
    X = (X - median) / iqr
    X = AdaptiveGrouping('ch75').map_to_template(X, ch_names)
    return X

template_ch_names = [
    'A1', 'A2', 'TP9', 'TP10', 'F9', 'F10', 'Fp1', 'Fp2', 'Fpz',
    'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T3', 'C3', 'Cz', 'C4', 'T4', 'CP5', 'CP1', 'CP2', 'CP6',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'POz', 'O1', 'Oz', 'O2',
    'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
    'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3',
    'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3',
    'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
    'FT9', 'FT10', 'PO9', 'PO10', 'P9', 'P10', 'AFz'
] # 75 channels

template_zones = {
    'Frontal': ['Fp1', 'Fp2', 'Fpz', 'AF7', 'AF3', 'AF4', 'AF8', 'AFz', 'F9', 'F10', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F5', 'F1', 'F2', 'F6', 'FC5', 'FC1', 'FC2', 'FC6', 'FC3', 'FCz', 'FC4', 'FT7', 'FT8', 'FT9', 'FT10', 'A1', 'A2'],
    'Central': ['C3', 'Cz', 'C4', 'C5', 'C1', 'C2', 'C6'],
    'Temporal': ['T3', 'T4', 'T5', 'T6', 'TP9', 'TP10', 'TP7', 'TP8'],
    'Parietal': ['CP5', 'CP1', 'CP2', 'CP6', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'P5', 'P1', 'P2', 'P6', 'P9', 'P10'],
    'Occipital': ['POz', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'PO9', 'PO10', 'O1', 'Oz', 'O2'],
}

class META:
    def __init__(self, h5_name, ch_names, subjects, classes, resample_rate=250, time_length=10):
        self.h5_name = h5_name
        self.h5_path = f'{DATA_FOLDER}/{h5_name}.h5'
        self.ch_names = ch_names
        self.subjects = subjects
        self.classes = classes
        self.resample_rate = resample_rate
        self.time_length = time_length
        self.Ycache = {}

    def loadfn_for_train(self, sub, sample, all_classes=None):
        # Notes from James (2024-June-12)
        # Load all the labels of this subject into RAM, map to global class id if all_classes is provided
        with h5py.File(self.h5_path, 'r') as f:
            X = f[f'{sub}/X'][sample]
            if sub not in self.Ycache:
                Y = f[f'{sub}/Y'][()]
                if all_classes is not None:
                    Y = np.array([all_classes.index(self.classes[y]) for y in Y])
                self.Ycache[sub] = Y
            Y = self.Ycache[sub][sample]
        return X, Y

    def load_data(self, sub):
        assert sub in self.subjects, f'{sub} not in {self.h5_path}'
        with h5py.File(self.h5_path, 'r') as f:
            return f[f'{sub}/X'][()], f[f'{sub}/Y'][()]

    def load_label(self, sub):
        assert sub in self.subjects, f'{sub} not in {self.h5_path}'
        with h5py.File(self.h5_path, 'r') as f:
            return f[f'{sub}/Y'][()]

    def get_sub_x_shape(self, sub):
        assert sub in self.subjects, f'{sub} not in {self.h5_path}'
        with h5py.File(self.h5_path, 'r') as f:
            # return len(f[f'{sub}/X'])
            return f[f'{sub}/X'].shape

    def __str__(self):
        return f'{self.fn_h5} CH:{len(self.ch_names)} SUB:{len(self.subjects)} {self.classes}'

    def get_resample_rate(self):
        if not self.resample_rate:
            raise ValueError('You need to specified resample rate for this dataset in EEG_Dataset folder')
        else:
            return self.resample_rate

    def get_time_length(self):
        if not self.time_length:
            raise ValueError('You need to specified resample rate for this dataset in EEG_Dataset folder')
        else:
            return self.time_length

def sliding_window(data, window_length, overlap):
    """
    This function split EEG data into shorter segments using sliding windows
    Parameters
    ----------
    data: data, channel
    window_length: how long each window is
    overlap: overlap rate

    Returns
    -------
    data: (num_segment, window_length, channel)
    """
    idx_start = 0
    idx_end = window_length
    step = int(window_length * (1 - overlap))
    data_split = []
    while idx_end < data.shape[0]:
        data_split.append(data[idx_start:idx_end])
        idx_start += step
        idx_end = idx_start + window_length
    return np.stack(data_split)

def split_trial(data: list, label: list, segment_length: int = 4,
                overlap: float = 0, sampling_rate: int = 250, sub_segment=0,
                sub_overlap=0.0) -> tuple:
    """
    This function split one trial's data into shorter segments
    Parameters
    ----------
    data: list of (time, chan) or list of (time, chan, f)
    label: list of label
    segment_length: how long each segment is (e.g. 1s, 2s,...)
    overlap: overlap rate
    sampling_rate: sampling rate
    sub_segment: how long each sub-segment is (e.g. 1s, 2s,...)
    sub_overlap: overlap rate of sub-segment

    Returns
    -------
    data:list of (num_segment, segment_length, chan) or list of (num_segment, segment_length, chan, f)
    label: list of (num_segment,)
    """
    data_segment = sampling_rate * segment_length
    sub_segment = sampling_rate * sub_segment
    data_split = []
    label_split = []

    for i, trial in enumerate(data):
        trial_split = sliding_window(trial, data_segment, overlap)
        label_split.append(np.repeat(label[i], len(trial_split)))
        if sub_segment != 0:
            trial_split_split = []
            for seg in trial_split:
                trial_split_split.append(sliding_window(seg, sub_segment, sub_overlap))
            trial_split = np.stack(trial_split_split)
        data_split.append(trial_split)
    assert len(data_split) == len(label_split)
    return data_split, label_split
