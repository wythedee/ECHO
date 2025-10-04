import numpy as np
from share import THREADS, SRC_FOLDER, META, DATA_FOLDER
from utils.file_loader import load_npz_files
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
#from utils.file_loader2 import load_npz_files
from typing import Tuple,List
import mne

SRC_FOLDER = os.path.join(SRC_FOLDER, 'SLEEP')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'SLEEP')
class SleepData:
    def __init__(self, data: np.ndarray, labels: np.ndarray, sampling_rate: float=100):
        self.data = data
        self.labels = labels
        self.sampling_rate = sampling_rate

CH_NAMES = ['Fp1']
SUBJECTS = [f'S{i:02d}' for i in range(1, 39)]

def _get_datasets(
                  modals: int,
                  data_dir: str,
                  stride: int = 35,
                  two_d: bool = True
) -> List[SleepData]:
    #data 是由num_files个元素的list，每个list都是一个ndarray，每个array形状为(num_epoch,1,1,3000,num_channel),data[0]为(1092,1,1,3000,3)
    data, labels = load_npz_files(
        #glob.glob(os.path.join(data_dir, '*.npz')),！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        glob.glob(os.path.join(data_dir, '*.npz')),
        two_d=two_d
    )
    global SUBJECTS
    SUBJECTS = [f'S{i:02d}' for i in range(1, len(data) + 1)]

    def data_big_group(d: np.ndarray) -> np.ndarray:
        """
        A closure to divide data into big groups to prevent
        data leak in data enhancement.
        """
        return_data = np.array([])
        beg = 0
        while (beg + stride) <= d.shape[0]:
            y = d[beg: beg + stride, ...]
            y = y.reshape((1, 1, stride, 3000, 3))
            # y = y[np.newaxis, ...]
            return_data = y if beg == 0 else np.append(return_data, y, axis=0)
            beg += stride
        return return_data

    def label_big_group(labels: np.ndarray) -> np.ndarray:
        """
        A closure to divide labels into big groups to prevent
        data leak in data enhancement.
        """
        return_labels = np.array([])
        beg = 0
        while (beg + stride) <= len(labels):
            y = labels[beg: beg + stride]
            y = y[np.newaxis, ...]
            return_labels = y if beg == 0 else np.concatenate(
                (return_labels, y),
                axis=0
            )
            beg += stride
        return return_labels  # [:, np.newaxis, ...]

    with ThreadPoolExecutor(max_workers=4) as executor:
        data = executor.map(data_big_group, data)
        labels = executor.map(label_big_group, labels)

    if modals is None:
        datasets = [SleepData(d, l) for d, l in zip(data, labels)]

    elif modals == 3:
        datasets = [SleepData(d[..., :2], l) for d, l in zip(data, labels)]
    elif modals == 4:
        datasets=[]
        for d, l in zip(data, labels):
            d[...,1]=d[..., 0]
            eeg_dataset=SleepData(d[..., :2], l)
            # print(eeg_dataset.data[0,0,0,0,0],eeg_dataset.data[0,0,0,0,1])
            datasets.append(eeg_dataset)
    elif modals == 5:
        datasets=[]
        for d, l in zip(data, labels):
            d[..., 0] = d[..., 1]
            eog_dataset = SleepData(d[..., :2], l)
            # print(eog_dataset.data[0, 0, 0, 0, 0], eog_dataset.data[0, 0, 0, 0, 1])
            datasets.append(eog_dataset)
    else:
        #datasets中包含39个文件，以第一个文件为例datasets[0]，它对应一个SleepData:31，datasets[0]类型为ndarray，形状为（31,1,35,3000,2）,如果单模态就没有最后一个维度
        datasets = [SleepData(d[..., modals], l) for d, l in zip(data, labels)]



    return datasets



get_eeg_datasets = partial(_get_datasets, 0)
get_eog_datasets = partial(_get_datasets,  1)
get_emg_datasets = partial(_get_datasets, 2)
get_eeg_and_eog_datasets = partial(_get_datasets, 3)
get_double_eeg_datasets = partial(_get_datasets, 4)
get_double_eog_datasets = partial(_get_datasets, 5)
get_datasets = partial(_get_datasets, None)

SLEEP_SleepEDF = META("SLEEP_01_SleepEDF", CH_NAMES, SUBJECTS, "SLEEP")

def main():
    datasets = get_eeg_datasets(os.path.join(SRC_FOLDER, "SLEEP_01_SleepEDF-39"))
    for dataset in datasets:
        data = dataset.data
        labels = dataset.labels
        print(data.shape, labels.shape, np.unique(labels, return_counts=True))
        data = data.squeeze([1, -1])
        mne_epo = mne.EpochsArray(data, info=mne.create_info(ch_names=CH_NAMES, sfreq=100, ch_types='eeg'))
        mne_epo.filter(l_freq=0.5, h_freq=50, verbose=False)
        mne_epo.resample(250, verbose=False)
        



if __name__ == "__main__":
    main()