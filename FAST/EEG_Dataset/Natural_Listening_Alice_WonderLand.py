import os
import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
from functools import partial
import h5py
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = "/media/james/public/dataset"

NAME = 'Natural_Listening_Alice_WonderLand'
SUBJECTS = [
    'S01', 'S03', 'S04', 'S05', 'S06', 'S08', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15',
    'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S25', 'S26', 'S34', 'S35', 'S36', 'S37',
    'S38', 'S39', 'S40', 'S41', 'S42', 'S44', 'S45', 'S48'
]

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
            onset_time.append(int(row['onset']*SF))
            offset_time.append(int(row['offset']*SF))
    ONSET_TIME.append(onset_time)
    OFFSET_TIME.append(offset_time)
    WORD_LISTS.append(word_list)


WORD_MAP = np.array(WORD_MAP)
WORD_MAP = np.unique(WORD_MAP)
WORD_MAP = WORD_MAP.tolist()

def proc_one(sub):
    data = mne.io.read_raw_brainvision(f'{SRC_FOLDER}/{NAME}/{sub}.vhdr', preload=True)
    channels_to_drop = ['VEOG', 'Aux5', 'AUD']
    for channel in channels_to_drop:
        if channel in data.ch_names:
            data.drop_channels([channel])
    data = data.filter(1, 40, verbose=False)
    raw = data.get_data().astype(np.float32)
    sfreq = data.info['sfreq']
    events, event_id = mne.events_from_annotations(data)
    events = events[:, 0]

    eegSegments = []
    lables = []

    print(len(raw[0]))
    for i in range(1, 13):
        onset_time = ONSET_TIME[i-1]
        offset_time = OFFSET_TIME[i-1]
        eegSegment = []
        lable = []
        if len(events) > i:
            even_time = events[i]
            for j in range(len(onset_time)):
                start = onset_time[j]
                end = offset_time[j]
                if len(raw[0]) < (end+even_time):
                    continue
                X = raw[:, (start+even_time):(end+even_time)]
                eegSegment.append(X)
                lables.extend([WORD_MAP.index(WORD_LISTS[i-1][j])] * (end-start))
            eegSegment = np.concatenate(eegSegment, axis=1)
            eegSegments.append(eegSegment)

    X = np.concatenate(eegSegments, axis=1)
    print(X.shape, len(lables))

    return sub, X, lables

def proc_all():
    with mp.Pool(24) as pool:
        res = pool.map(proc_one, SUBJECTS)

if __name__ == '__main__':
    # for i in range(len(SUBJECTS)):
    #     sub = SUBJECTS[i]
    #     print("Processing subject: ", sub)
    #     proc_one(sub)
    sub = 'S26'
    proc_one(sub)