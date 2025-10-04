import argparse
import os
import warnings
from typing import Dict, List

import numpy as np
import torch
import einops
import mne
import h5py

# =================== Embedded SEED-V constants (no external import) =================== #
# subject 7 has issues in data; exclude it as in original pipeline
SUBJECTS = ['1_', '2_', '3_', '4_', '5_', '6_', '8_', '9_', '10_', '11_', '12_', '13_', '14_', '15_', '16_']

CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
            'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'O1', 'Oz', 'O2']

session_labels = {
    1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

time_stamp = {
    1: {
        'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
        'end':   [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
    },
    2: {
        'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
        'end':   [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]
    },
    3: {
        'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
        'end':   [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
    },
}


def get_trial_split(cfg_py: str, ds_name: str) -> Dict[str, List[int]]:
    """Load <DS>_split from a python config as per-trial indices (relative to a file)."""
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
    if not hasattr(mod, "pipeline"):
        raise RuntimeError("share.py does not define pipeline")
    return mod.pipeline


def process_subject_into_splits(subject_tag: str, raw_root: str, trial_split: Dict[str, List[int]], pipeline_fn, time_len_sec: int = 10, sfreq_out: int = 250):
    per_split = {"train": {"X": [], "Y": []}, "val": {"X": [], "Y": []}, "test": {"X": [], "Y": []}}
    import glob
    for session in [1, 2, 3]:
        pat = os.path.join(raw_root, "EMO_03_SEED_V", "EEG_raw", f"{subject_tag}{session}*.cnt")
        files = list(glob.glob(pat))
        if not files:
            continue
        fn = files[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            eeg_raw = mne.io.read_raw_cnt(fn)

        useless_ch = ['M1', 'M2', 'VEO', 'HEO', 'CB1', 'CB2']
        to_drop = [ch for ch in useless_ch if ch in eeg_raw.ch_names]
        if to_drop:
            eeg_raw.drop_channels(to_drop)
        sfreq = int(eeg_raw.info['sfreq'])

        ts_start = time_stamp[session]['start']
        ts_end   = time_stamp[session]['end']
        data_matrix = eeg_raw.get_data()

        for i in range(1, 15+1):
            trial = data_matrix[:, ts_start[i-1]*sfreq : ts_end[i-1]*sfreq]
            raw = mne.io.RawArray(trial, mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types='eeg', verbose=False), verbose=False)
            raw = raw.resample(sfreq_out)
            raw.filter(l_freq=1, h_freq=40, verbose=False)
            x = raw.get_data().astype(np.float32)
            x = torch.tensor(x).unfold(1, sfreq_out*time_len_sec, sfreq_out*time_len_sec)
            x = einops.rearrange(x, 'C N T -> N C T').numpy()

            label = session_labels[session][i-1]
            y = np.array(label).repeat(x.shape[0]).astype(np.uint8)

            trial_idx_rel = i - 1
            split_name = 'train'
            for k in ['train', 'val', 'test']:
                if trial_idx_rel in trial_split.get(k, []):
                    split_name = k
                    break

            per_split[split_name]['X'].append(x)
            per_split[split_name]['Y'].append(y)

    out = {}
    for k in ['train', 'val', 'test']:
        if per_split[k]['X']:
            X = np.concatenate(per_split[k]['X'], axis=0)
            X = pipeline_fn(X, CH_NAMES)
            Y = np.concatenate(per_split[k]['Y'], axis=0)
        else:
            X = np.zeros((0, len(CH_NAMES), sfreq_out*time_len_sec), dtype=np.float32)
            Y = np.zeros((0,), dtype=np.uint8)
        out[k] = {'X': X, 'Y': Y}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="/path/to/your/dataset_root/EMO")
    parser.add_argument("--cfg_py", type=str, default="/path/to/your/ECHO/FAST/dataset_split_config.py")
    parser.add_argument("--share_dir", type=str, default="/path/to/your/ECHO/FAST/EEG_Dataset")
    parser.add_argument("--out_h5", type=str, default="/path/to/your/dataset_root/EMO/EMO_03_SEED_V.h5")
    parser.add_argument("--time_len", type=int, default=1)
    parser.add_argument("--sfreq", type=int, default=250)
    args = parser.parse_args()

    split = get_trial_split(args.cfg_py, "EMO_03_SEED_V")
    pipeline_fn = load_pipeline(args.share_dir)

    if os.path.exists(args.out_h5):
        os.remove(args.out_h5)
    with h5py.File(args.out_h5, 'w') as f:
        for sub in SUBJECTS:
            per_split = process_subject_into_splits(sub, args.raw_root, split, pipeline_fn, time_len_sec=args.time_len, sfreq_out=args.sfreq)
            subj_num = int(sub.rstrip('_'))
            for split_name, idx_prefix in [("train", 1), ("val", 2), ("test", 3)]:
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
