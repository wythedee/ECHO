import argparse
import os
import warnings
from typing import Dict, List

import numpy as np
import torch
import einops
import mne
import h5py
import scipy.io

SUBJECTS = ['1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_', '11_', '12_', '13_', '14_', '15_']

ORIGINAL_CH_NAMES = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3','POz',
    'PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

# remove CB1 and CB2
CH_NAMES = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3','POz',
    'PO4','PO6','PO8','O1','Oz','O2']

SRC_RATE = 200
L_FREQ = 0.3
H_FREQ = 50
RESAMPLE_RATE = 250
TIME_LENGTH = 4

session_labels = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}


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


def process_subject(subject_tag: str, raw_root: str, trial_split: Dict[str, List[int]], pipeline_fn, time_len_sec: int, resample_rate: int):
    per_split = {"train": {"X": [], "Y": []}, "val": {"X": [], "Y": []}, "test": {"X": [], "Y": []}}
    import glob
    for session in [1, 2, 3]:
        pat = os.path.join(raw_root, "EMO_02_SEED_IV", "eeg_raw_data", str(session), f"{subject_tag}*.mat")
        files = list(glob.glob(pat))
        if not files:
            continue
        fn = files[0]
        data = scipy.io.loadmat(fn, squeeze_me=True)
        prefix = list(data.keys())[-1].split('_')[0]
        labels = session_labels[session]
        for i in range(1, 24+1):
            x = data[f'{prefix}_eeg{i}']
            raw = mne.io.RawArray(x, mne.create_info(ch_names=ORIGINAL_CH_NAMES, sfreq=SRC_RATE, ch_types='eeg'))
            raw.drop_channels(['CB1','CB2'])
            raw = raw.resample(resample_rate)
            raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
            x = raw.get_data().astype(np.float32)
            x = torch.tensor(x).unfold(1, resample_rate*time_len_sec, resample_rate*time_len_sec)
            x = einops.rearrange(x, 'C N T -> N C T').numpy()
            y = np.array(labels[i-1]).repeat(x.shape[0]).astype(np.uint8)

            trial_idx_rel = i - 1
            split_name = 'train'
            for k in ['train','val','test']:
                if trial_idx_rel in trial_split.get(k, []):
                    split_name = k
                    break
            per_split[split_name]['X'].append(x)
            per_split[split_name]['Y'].append(y)

    out = {}
    for k in ['train','val','test']:
        if per_split[k]['X']:
            X = np.concatenate(per_split[k]['X'], axis=0)
            X = pipeline_fn(X, CH_NAMES)
            Y = np.concatenate(per_split[k]['Y'], axis=0)
        else:
            X = np.zeros((0, len(CH_NAMES), resample_rate*time_len_sec), dtype=np.float32)
            Y = np.zeros((0,), dtype=np.uint8)
        out[k] = {'X': X, 'Y': Y}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="/path/to/your/dataset_root/EMO")
    parser.add_argument("--cfg_py", type=str, default="/path/to/your/ECHO/FAST/dataset_split_config.py")
    parser.add_argument("--share_dir", type=str, default="/path/to/your/ECHO/FAST/EEG_Dataset")
    parser.add_argument("--out_h5", type=str, default="/path/to/your/dataset_root/EMO/EMO_02_SEED_IV.h5")
    parser.add_argument("--time_len", type=int, default=1)
    parser.add_argument("--sfreq", type=int, default=250)
    args = parser.parse_args()

    split = get_trial_split(args.cfg_py, "EMO_02_SEED_IV")
    pipeline_fn = load_pipeline(args.share_dir)

    if os.path.exists(args.out_h5):
        os.remove(args.out_h5)
    with h5py.File(args.out_h5, 'w') as f:
        for sub in SUBJECTS:
            per_split = process_subject(sub, args.raw_root, split, pipeline_fn, time_len_sec=args.time_len, resample_rate=args.sfreq)
            subj_num = int(sub.rstrip('_'))
            for split_name, idx_prefix in [("train",1),("val",2),("test",3)]:
                key = f"{idx_prefix}_{subj_num}"
                X = per_split[split_name]['X']
                Y = per_split[split_name]['Y']
                grp = f.create_group(key)
                grp.create_dataset('X', data=X, compression='gzip')
                grp.create_dataset('Y', data=Y, compression='gzip')
                print(key, X.shape, Y.shape, np.unique(Y, return_counts=True))
    print(f"Wrote H5: {args.out_h5}")


if __name__ == '__main__':
    main()
