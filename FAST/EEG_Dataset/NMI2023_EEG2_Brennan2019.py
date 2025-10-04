import os
import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
import torch
import multiprocessing as mp
import multiprocessing.dummy as dmp
from functools import partial
import h5py
import pandas as pd
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = "/media/james/public/dataset"
NAME = "NMI2023_EEG2_Brennan2019"
SUBJECTS = [
    'S01', 'S03', 'S04', 'S05', 'S06', 'S08', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
    'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S25', 'S26', 'S34', 'S35', 'S36', 'S37',
    'S38', 'S39', 'S40', 'S41', 'S42', 'S44', 'S45', 'S48'
]

def proc_one(sub):
    tmin = -0.5
    tmax = 2.5

    # Read raw data of the eeg of the subjects
    raw = mne.io.read_raw_brainvision(f'{SRC_FOLDER}/{NAME}/{sub}.vhdr', preload=True)
    Fs = raw.info['sfreq']
    channels_to_drop = ['VEOG', 'Aux5', 'AUD']
    for channel in channels_to_drop:
        if channel in raw.ch_names:
            raw.drop_channels([channel])
    # raw = raw.resample(250)
    eegData = raw.filter(1, 40, verbose=False)
    segments, segment_ids = mne.events_from_annotations(eegData)
    # print(segments)
    # print(eegData.times)
    # print(eegData.get_data().shape)
    # exit()
    
    # print(len(ONSET_TIME[0]),ONSET_TIME[0])
    # print(len(WORD_LISTS[0]), WORD_LISTS[0])
    
    # Slice the raw eegData regarding words
    ref_times = [eegData.times[segments[i][0]] for i in range(1, 13)]
    onset_times = [times + ref_times[i] for i, times in enumerate(ONSET_TIME)]
    # for i in range (12):
    #     print(len(onset_times[i]))
    #     print(len(WORD_LISTS[i]))
    onset_times_index = [np.round(times * Fs).astype(int) for times in onset_times]
    events = np.concatenate([np.column_stack((times, np.zeros(len(times), int), np.arange(len(times)))) for times in onset_times_index])
    word_lists = np.concatenate(WORD_LISTS)
    # print(events.shape, word_lists.shape) 
    epochs = mne.Epochs(eegData, events, event_id=None, tmin=tmin, tmax=tmax, baseline=None, preload=True, event_repeated='drop')
    print(epochs.get_data().shape)

    return epochs, word_lists





if __name__ == "__main__":
    vhdr = f'{SRC_FOLDER}/{NAME}/S01.vhdr'
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    SF = raw.info['sfreq']
    CH_NAMES = raw.ch_names

    df = pd.read_csv(f'{SRC_FOLDER}/{NAME}/AliceChapterOne-EEG.csv')
    # Initialize the word map dictionary
    WORD_MAP = []
    WORD_LISTS = []
    ONSET_TIME = []
    OFFSET_TIME = []

    # Populate the word map
    for i in range(1, 13):
        onset_time = []
        offset_time = []
        word_list = []
        for index, row in df.iterrows():
            word = row['Word']
            if word not in WORD_MAP:
                WORD_MAP.append(word)
            if row['Segment'] == i:
                word_list.append(word)
                # onset_time.append(int(row['onset']*SF))
                # offset_time.append(int(row['offset']*SF))
                onset_time.append(row['onset'])
                offset_time.append(row['offset'])
        ONSET_TIME.append(onset_time)
        OFFSET_TIME.append(offset_time)
        WORD_LISTS.append(word_list)


    WORD_MAP = np.array(WORD_MAP)
    WORD_MAP = np.unique(WORD_MAP)
    WORD_MAP = WORD_MAP.tolist()

    proc_one('S01')