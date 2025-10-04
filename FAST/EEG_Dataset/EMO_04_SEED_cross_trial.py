import argparse
import os
import warnings
from typing import Dict, List

import numpy as np
import torch
import einops
import mne
import h5py
import scipy.io as sio
import os.path as osp

# SEED has 15 subjects indexed as 1..15
SUBJECTS = list(range(1, 16))

# Original channel names from SEED files (uppercase, including CB1/CB2)
ORIGINAL_CH_NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
            'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
            'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
            'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5',
            'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

# Pipeline channel names (CamelCase, without CB1/CB2), aligned with template usage
CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
            'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'O1', 'Oz', 'O2']

RESAMPLE_RATE = 250


def get_trial_split(cfg_py: str, ds_name: str) -> Dict[str, List[int]]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("split_cfg", cfg_py)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)  # type: ignore
    attr = f"{ds_name}_trial_split"
    if not hasattr(cfg, attr):
        raise ValueError(f"{attr} not found in {cfg_py}")
    split = getattr(cfg, attr)
    out = {}
    for k in ["train", "val", "test"]:
        v = split.get(k, [])
        out[k] = [int(i) for i in list(v)]
    return out


def load_pipeline(share_dir: str):
    import importlib.util, sys
    sys.path.append(share_dir)
    spec = importlib.util.spec_from_file_location("share", os.path.join(share_dir, "share.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.pipeline


def load_one_session(file_to_load, feature_key):
    data = sio.loadmat(file_to_load, verify_compressed_data_integrity=False)
    keys_to_select = [k for k in data.keys() if feature_key in k]
    data_session = [data[k] for k in keys_to_select]
    min_length = min([item.shape[1] for item in data_session])
    data_session = [sess[:, :min_length] for sess in data_session]
    data_session = np.array(data_session)
    return data_session


def process_subject(sub_idx: int, raw_root: str, trial_split: Dict[str, List[int]], pipeline_fn, time_len_sec: int):
    data_folder = osp.join(raw_root, 'EMO_04_SEED', 'SEED', 'Preprocessed_EEG')
    label = sio.loadmat(osp.join(data_folder, 'label.mat'))['label']
    label += 1
    label = np.squeeze(label)

    files_this_subject = []
    for root, dirs, files in os.walk(data_folder, topdown=False):
        for name in files:
            if sub_idx < 10:
                sub_code = name[:2]
            else:
                sub_code = name[:3]
            if f'{sub_idx}_' == sub_code:
                files_this_subject.append(name)
    files_this_subject = sorted(files_this_subject)

    Xspl = {"train": [], "val": [], "test": []}
    Yspl = {"train": [], "val": [], "test": []}

    for file in files_this_subject:
        sess = load_one_session(osp.join(data_folder, file), feature_key='eeg')
        # keep all labels; do not drop Neutral or remap classes
        label_selected = label

        # iterate trials
        for i, x in enumerate(sess, start=1):
            # x: (C, T), original sfreq ~1000, channel names in ORIGINAL_CH_NAMES
            raw = mne.io.RawArray(x, mne.create_info(ch_names=ORIGINAL_CH_NAMES, sfreq=1000, ch_types='eeg'), verbose=False)
            # drop CB1/CB2 to align with pipeline CH_NAMES
            drop_list = [ch for ch in ['CB1','CB2'] if ch in raw.ch_names]
            if drop_list:
                raw.drop_channels(drop_list)
            raw.resample(RESAMPLE_RATE, npad='auto')
            raw.filter(l_freq=1, h_freq=40, verbose=False)
            x = raw.get_data().astype(np.float32)
            # window to non-overlap segments of time_len_sec seconds
            x = torch.tensor(x).unfold(1, RESAMPLE_RATE*time_len_sec, RESAMPLE_RATE*time_len_sec)
            x = einops.rearrange(x, 'C N T -> N C T').numpy()
            y = np.array(label_selected[i-1]).repeat(x.shape[0]).astype(np.uint8)

            trial_idx_rel = i - 1
            split_name = 'train'
            for k in ['train','val','test']:
                if trial_idx_rel in trial_split.get(k, []):
                    split_name = k
                    break
            Xspl[split_name].append(x)
            Yspl[split_name].append(y)

    out = {}
    for k in ['train','val','test']:
        if Xspl[k]:
            X = np.concatenate(Xspl[k], axis=0)
            X = pipeline_fn(X, CH_NAMES)
            Y = np.concatenate(Yspl[k], axis=0)
        else:
            X = np.zeros((0, len(CH_NAMES), RESAMPLE_RATE*time_len_sec), dtype=np.float32)
            Y = np.zeros((0,), dtype=np.uint8)
        out[k] = {'X': X, 'Y': Y}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="/path/to/your/dataset_root/EMO")
    parser.add_argument("--cfg_py", type=str, default="/path/to/your/ECHO/FAST/dataset_split_config.py")
    parser.add_argument("--share_dir", type=str, default="/path/to/your/ECHO/FAST/EEG_Dataset")
    parser.add_argument("--out_h5", type=str, default="/path/to/your/dataset_root/EMO/EMO_04_SEED.h5")
    parser.add_argument("--time_len", type=int, default=1)
    args = parser.parse_args()

    split = get_trial_split(args.cfg_py, "EMO_04_SEED")
    pipeline_fn = load_pipeline(args.share_dir)

    if os.path.exists(args.out_h5):
        os.remove(args.out_h5)
    with h5py.File(args.out_h5, 'w') as f:
        for sub in SUBJECTS:
            per_split = process_subject(sub, args.raw_root, split, pipeline_fn, time_len_sec=args.time_len)
            for split_name, idx_prefix in [("train",1),("val",2),("test",3)]:
                key = f"{idx_prefix}_{sub}"
                X = per_split[split_name]['X']
                Y = per_split[split_name]['Y']
                grp = f.create_group(key)
                grp.create_dataset('X', data=X, compression='gzip')
                grp.create_dataset('Y', data=Y, compression='gzip')
                print(key, X.shape, Y.shape, np.unique(Y, return_counts=True))
    print(f"Wrote H5: {args.out_h5}")


if __name__ == '__main__':
    main()
