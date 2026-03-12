import os
import mne
import numpy as np
import scipy
import multiprocessing as mp
from functools import partial
import h5py
import sys
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

SRC_FOLDER = os.path.join(SRC_FOLDER, 'MI')

SRC_NAME = 'MI_PhysioNet'
NAME = 'MI_12_PhysioNet'
# | clsss | Meaning    | Run  ID             | in EDF |
# | ---   | ---------- | ------------------- | ------ |
# | 0     | Left Hand  | 3, 4, 7, 8, 11, 12  | T1     |
# | 1     | Right Hand | 3, 4, 7, 8, 11, 12  | T2     |
# | 2     | Both Fists | 5, 6, 9, 10, 13, 14 | T1     |
# | 3     | Both Feet  | 5, 6, 9, 10, 13, 14 | T2     |
TEXT_LABELS = ['MI/Left', 'MI/Right', 'MI/BothFists', 'MI/BothFeet']
CH_NAMES = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3',
    'CP1','CPz','CP2','CP4','CP6','Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5',
    'F3','F1','Fz','F2','F4','F6','F8','FT7','FT8','T7','T8','T9','T10','TP7','TP8','P7','P5',
    'P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'
]
SUBJECTS = [
    'S001','S002','S003','S004','S005','S006','S007','S008','S009','S010',
    'S011','S012','S013','S014','S015','S016','S017','S018','S019','S020',
    'S021','S022','S023','S024','S025','S026','S027','S028','S029','S030',
    'S031','S032','S033','S034','S035','S036','S037','S038','S039','S040',
    'S041','S042','S043','S044','S045','S046','S047','S048','S049','S050',
    'S051','S052','S053','S054','S055','S056','S057','S058','S059','S060',
    'S061','S062','S063','S064','S065','S066','S067','S068','S069','S070',
    'S071','S072','S073','S074','S075','S076','S077','S078','S079','S080',
    'S081','S082','S083','S084','S085','S086','S087','S088','S089','S090',
    'S091','S092','S093','S094','S095','S096','S097','S098','S099','S100',
    'S101','S102','S103','S104','S105','S106','S107','S108','S109'
]

MI_PhysioNet = META(NAME, CH_NAMES, SUBJECTS, TEXT_LABELS, resample_rate=250, time_length=10)

event_id = {"T1": 1, "T2": 2}
group_hand = {3, 4, 7, 8, 11, 12}
group_both = {5, 6, 9, 10, 13, 14}

def proc_one(sub, tmin=0.0, tmax=4.0):
    all_epochs, all_labels = [], []

    for r in range(3, 15):
        raw = mne.io.read_raw_edf(f"{SRC_FOLDER}/{SRC_NAME}/{sub}/{sub}R{r:02d}.edf", preload=True, verbose="error")
        raw.filter(l_freq=0.3, h_freq=50, verbose=False)
        raw.rename_channels(lambda s: s.rstrip("."))
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

        if len(events) == 0:
            continue

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True, verbose=False)
        y = epochs.events[:, 2]

        if r in group_hand:
            y_new = np.where(y == 1, 0, 1)  # T1 → Left (0), T2 → Right (1)
        elif r in group_both:
            y_new = np.where(y == 1, 2, 3)  # T1 → Both Fists (2), T2 → Both Feet (3)
        else:
            continue

        all_epochs.append(epochs)
        all_labels.append(y_new)

    epochs = mne.concatenate_epochs(all_epochs)
    labels = np.concatenate(all_labels).astype(np.uint8)

    # epochs = epochs.filter(l_freq=Parameters['lp'], h_freq=Parameters['hp'], verbose=False)
    epochs = epochs.resample(250, npad='auto')

    X = epochs.get_data().astype(np.float32)
    X = pipeline(X, CH_NAMES)

    return sub, X, labels

if __name__ == "__main__":
    # sub, X, Y = proc_one('S001')  # Example usage for testing

    with mp.Pool(64) as pool:
        res = pool.map(proc_one, SUBJECTS)

    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

    # with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'r') as f:
    #     for sub in SUBJECTS:
    #         print(sub, f[f'{sub}/X'].shape, f[f'{sub}/Y'].shape, np.unique(f[f'{sub}/Y'], return_counts=True))