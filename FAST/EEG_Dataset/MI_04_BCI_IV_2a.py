import mne
import numpy as np
import os
import pickle
import scipy
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')
SRC_NAME = 'MI_BCI_IV_2a'
NAME = 'MI_04_BCI_IV_2a'
SUBJECTS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
CH_NAMES = [
    'Fz',
    'FC3','FC1','FCz','FC2','FC4',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP3','CP1','CPz','CP2','CP4',
    'P1','Pz','P2',
    'POz',
]
MI_BCI_IV_2a = META(NAME, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right', 'MI/BothFeet', 'MI/Tongue'], resample_rate=250, time_length=4)

def load_BCI_IV_IIa_label(label_path):
    labelmat = scipy.io.loadmat(label_path)
    label = np.squeeze(np.array(labelmat['classlabel'])) - 1
    return label

def load_BCI_IV_IIa_data(data_path, tmin=0, tmax=4, baseline=None, filter_or_not=True, standard_ornot=True):
    raw_data = mne.io.read_raw_gdf(data_path)
    fs = raw_data.info.get('sfreq')
    events, event_ids = mne.events_from_annotations(raw_data)
    stimcodes = ('769', '770', '771', '772', '783')
    stims = [value for key, value in event_ids.items() if key in stimcodes]
    epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                        baseline=baseline, preload=True, proj=False, reject_by_annotation=False, verbose=False)
    epochs.filter(l_freq=0.1, h_freq=75, verbose=False)
    channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
    epochs = epochs.drop_channels(channels_to_remove)
    epochs.resample(250, npad='auto', verbose=False)
    x_data = (epochs.get_data(copy=False) * 1e6)[:, :, 1:]
    return x_data

def proc_one(subject):
    x1 = load_BCI_IV_IIa_data(f'{SRC_FOLDER}/{SRC_NAME}/{subject}T.gdf')
    x2 = load_BCI_IV_IIa_data(f'{SRC_FOLDER}/{SRC_NAME}/{subject}E.gdf')
    x = np.concatenate([x1, x2], axis=0)
    y1 = load_BCI_IV_IIa_label(f'{SRC_FOLDER}/{SRC_NAME}/true_labels/{subject}T.mat')
    y2 = load_BCI_IV_IIa_label(f'{SRC_FOLDER}/{SRC_NAME}/true_labels/{subject}E.mat')
    y = np.concatenate([y1, y2], axis=0)
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    # mask = (y == 0) | (y == 1) # take only left and right hand MI
    # x = x[mask]
    # y = y[mask]
    # print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    x = pipeline(x, CH_NAMES)
    return subject, x, y

def proc_all():
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__=='__main__':
    proc_all()
