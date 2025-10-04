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

NAME = "NMI2023_EEG1_Broderick2019"
SUBJECTS = [
    'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 
    'Subject7', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12', 
    'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject18', 
    'Subject19'
]
CH_NAMES = [
    "Cz", "A2", "CPz", "A4", "P1", "A6", "P3", "A8", "A9", "PO7", 
    "A11", "A12", "A13", "A14", "O1", "A16", "PO3", "CMS", "Pz", 
    "A20", "POz", "A22", "Oz", "A24", "Iz", "A26", "A27", 
    "O2", "A29", "PO4", "DRL", "P2", "B1", "CP2", "B3", "P4", 
    "B5", "B6", "PO8", "B8", "B9", "P10", "P8", "B12", "P6", 
    "TP8", "B15", "CP6", "B17", "CP4", "B19", "C2", "B21", "C4", 
    "B23", "C6", "B25", "T8", "FT8", "B28", "FC6", "B30", "FC4", 
    "B32", "C1", "C2", "C3", "F4", "F6", "C6", "F8", "AF8", "C9", 
    "C10", "FC2", "F2", "C13", "C14", "AF4", "Fp2", "Fpz", "C18", 
    "AFz", "C20", "Fz", "C22", "FCz", "FC1", "F1", "C26", "C27", 
    "AF3", "Fp1", "AF7", "C31", "C32", "D1", "D2", "D3", "F3", 
    "F5", "D6", "F7", "FT7", "D9", "FC5", "D11", "FC3", "D13", 
    "C1", "D15", "CP1", "D17", "D18", "C3", "D20", "C5", "D22", 
    "T7", "TP7", "D25", "CP5", "D27", "CP3", "P5", "D30", "P7", "P9"
]
RUN = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5', 'Run6', 'Run7', 'Run8', 'Run9', 'Run10',
       'Run11', 'Run12', 'Run13', 'Run14', 'Run15', 'Run16', 'Run17', 'Run18', 'Run19', 'Run20',]


# This one loading the entire paragraph of the text
def load_events_private(run):
    # Load and process the json file of the words
    path = Path(SRC_FOLDER) / NAME
    run_id = run[3:]
    with open(path / "Natural_Speech" / "private" / f"align{run_id}.json") as f:
        align = json.load(f)

    wordlist = list()
    for entry in align["words"]:
        entry.pop("endOffset")
        entry.pop("startOffset")
        success = entry.pop("case") == "success"
        if not success:
            continue
        if entry["alignedWord"] == "<unk>":
            success = False
        entry["success"] = success

        txt = entry.pop("word")
        entry["string"] = txt
        entry["duration"] = entry["end"] - entry["start"]

        aligned = entry.pop("alignedWord")
        entry["aligned"] = aligned
        wordlist.append(entry)
        wordlist[-1]["kind"] = "word"

    df = pd.DataFrame(wordlist)
    return df

def load_raw(sub, run):
    path = Path(SRC_FOLDER) / NAME
    eeg_fname = (
        path
        / "Natural_Speech"
        / "EEG"
        / f"{sub}"
        / f"{sub}_{run}.mat"
    )
    mat = scipy.io.loadmat(str(eeg_fname))
    assert mat["fs"][0][0] == 128
    ch_types = ["eeg"] * 128
    montage = mne.channels.make_standard_montage("biosemi128")
    info = mne.create_info(montage.ch_names, 128.0, ch_types)
    eeg = mat["eegData"].T * 1e6
    assert len(eeg) == 128
    raw = mne.io.RawArray(eeg, info)
    raw.set_montage(montage)

    info_mas = mne.create_info(
        ["mastoids1", "mastoids2"], 128.0, ["misc", "misc"]
    )
    mastoids = mne.io.RawArray(mat["mastoids"].T * 1e6, info_mas)
    raw.add_channels([mastoids])

    raw = raw.pick_types(
        meg=False, eeg=True, misc=False, eog=False, stim=False
    
    ) 
    return raw

def proc_one_private(sub, run):
    tmin = -0.5
    tmax = 2.5

    # Getting the labels
    query = f"kind == 'word'"
    events = load_events_private(run).copy()
    events = events.sort_values("start")
    label = events.string.values

    # Process the EEG data
    raw = load_raw(sub, run)
    meta = load_events_private(run).copy().query(query)
    times = meta.start.values

    eegData = raw.filter(l_freq=1, h_freq=40, verbose=False)
    sample_rate = eegData.info['sfreq']
    delta = 0.5 / sample_rate
    mask = np.logical_and(times + tmin >= 0, times + tmax < eegData.times[-1] + delta)

    # Filter the label and map to the WORD_MAP_ALL
    label = label[mask]
    label = np.array([word_map_all.index(word) for word in label])

    # Filter the metadata
    meta = meta.iloc[np.where(mask)].reset_index(drop=True)
    samples = np.round(times[mask] * sample_rate).astype(int)
    event_ids = np.arange(len(samples))
    mne_events = np.column_stack((samples, np.zeros(len(samples), int), event_ids))
    epochs = mne.Epochs(eegData, mne_events, event_id=None, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, event_repeated='drop')

    print(epochs.get_data().shape)
    return epochs, label

def proc_one_wordVec(sub, run):
    tmin = -0.5
    tmax = 2.5

    # Check the onset and offset time, wordVec for a single run
    # print(onset_times[RUN.index(run)].shape)
    # print(offset_times[RUN.index(run)].shape)
    # print(word_vec_part[RUN.index(run)].shape)

    # Load the raw and process
    raw = load_raw(sub, run)
    eegData = raw.filter(l_freq=1, h_freq=40, verbose=False)
    sample_rate = eegData.info['sfreq']

    # Getting mask for a events
    delta = 0.5 / sample_rate
    mask = np.logical_and(onset_times[RUN.index(run)] + tmin >= 0,
                          onset_times[RUN.index(run)] + tmax < eegData.times[-1] + delta)
    
    # Filter the label and map to the WORD_MAP_PART
    label = word_vec_part[RUN.index(run)][mask]
    label = np.array([word_map_part.index(word[0]) for word in label])
    
    # Filter the events
    events = onset_times[RUN.index(run)][mask]
    samples = np.round(events * sample_rate).astype(int)
    event_ids = np.arange(len(samples))
    mne_events = np.column_stack((samples, np.zeros(len(samples), int), event_ids))
    epochs = mne.Epochs(eegData, mne_events, event_id=None, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, event_repeated='drop')
    
    print(epochs.get_data().shape)
    return epochs, label

if __name__ == "__main__":
    word_map_part = set()
    word_map_all = []
    word_vec_part = []
    onset_times = []
    offset_times = []
    fs = 128

    for run in RUN:
        data = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/Natural_Speech/Stimuli/Text/{run}.mat')
        wordVec = data['wordVec']
        onset_time = data['onset_time']
        offset_time = data['offset_time']

        for word in wordVec:
            word_map_part.add(word[0][0])

        # onset_indices = (onset_time * fs).astype(int)
        # offset_indices = (offset_time * fs).astype(int)

        # onset_times.append(onset_indices)
        # offset_times.append(offset_indices)
        onset_times.append(onset_time)
        offset_times.append(offset_time)
        word_vec_part.append(wordVec)

    word_map_part = np.array(list(word_map_part))
    word_map_part = np.unique(word_map_part)
    word_map_part = word_map_part.tolist()

    word_map_all = set()

    for run in RUN:
        df = load_events_private(run)
        for word in df.string:
            word_map_all.add(word)

    word_map_all = np.array(list(word_map_all))
    word_map_all = np.unique(word_map_all)
    word_map_all = word_map_all.tolist()

    # print(len(word_map_part), len(word_map_all))

    # proc_one_private("Subject1", "Run1")
    proc_one_wordVec("Subject1", "Run1")