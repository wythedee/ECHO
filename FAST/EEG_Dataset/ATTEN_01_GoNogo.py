import os
import re
import traceback
import mne
import h5py
import numpy as np
import multiprocessing as mp
import time
from typing import Dict, List, Tuple

# Project-wide helpers
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline, split_trial, find_available_path

# ATTEN_ICA=0 ATTEN_WORKERS=2 python EEG_Dataset/ATTEN_01_GoNogo.py
SRC_FOLDER = os.path.join(SRC_FOLDER, 'ATTEN')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'ATTEN')
NAME = 'ATTEN_01_GoNogo'

# We output 4s windows at 250 Hz, mapped to the unified 75-ch template in pipeline
RESAMPLE_RATE = 250
TIME_LENGTH = 4

# Two classes: Attention vs Rest
CLASSES = ['ATTEN/Rest', 'ATTEN/Attention']


# Register META (subjects will be filled after scanning)
ATTEN_GoNogo = META(NAME, [], [], CLASSES, resample_rate=RESAMPLE_RATE, time_length=TIME_LENGTH)

# Suppress verbose logs from MNE to keep console clean
mne.set_log_level('ERROR')

# Runtime toggles (environment-controlled)
ENABLE_ICA = os.environ.get('ATTEN_ICA', '1') == '1'  # set ATTEN_ICA=0 to skip ICA entirely
ICA_DECIM = int(os.environ.get('ATTEN_ICA_DECIM', '5'))  # decimate factor used during ICA fitting
ICA_MAX_T = float(os.environ.get('ATTEN_ICA_MAX_S', '0'))  # crop duration in seconds for ICA training (0 means full)
RESAMPLE_BEFORE_ICA = os.environ.get('ATTEN_RESAMPLE_BEFORE_ICA', '1') == '1'  # set to 0 to keep original sfreq for ICA
WORKERS = int(os.environ.get('ATTEN_WORKERS', '8'))


def _dataset_root() -> str:
    # Try several common layouts the user may have
    # 1) {SRC_FOLDER}/ATTEN/ATTEN_01_GoNogo/VPxxx-EEG
    # 2) {SRC_FOLDER}/ATT/ATTEN_01_GoNogo/VPxxx-EEG
    # 3) {SRC_FOLDER}/ATTEN/ThreeCognitiveTasks/VPxxx
    candidates = [
        os.path.join(SRC_FOLDER, 'ATTEN', 'ATTEN_01_GoNogo'),
        os.path.join(SRC_FOLDER, 'ATTEN_01_GoNogo'),
        os.path.join(SRC_FOLDER, 'ATTEN', 'ThreeCognitiveTasks'),
    ]
    root = find_available_path(candidates)
    print(f"[ATTEN][root] candidates={candidates}, chosen={root}")
    return root


def _list_subjects(root: str) -> List[str]:
    subs: List[str] = []
    for d in os.listdir(root):
        # Only consider directories, accept VPxxx or VPxxx-EEG
        if not os.path.isdir(os.path.join(root, d)):
            continue
        if re.match(r'^VP0?\d+(-EEG)?$', d):
            subs.append(d)
    subs.sort(key=lambda s: int(re.findall(r'\d+', s)[0]))
    print(f"[ATTEN][list_subjects] scanned '{root}', found {len(subs)} subjects")
    return subs


def _safe_read_brainvision(vhdr_path: str) -> mne.io.BaseRaw:
    print(f"[ATTEN][read] reading BrainVision: {vhdr_path}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose='ERROR')
    print(f"[ATTEN][read] loaded: sfreq={raw.info.get('sfreq')}, n_ch={len(raw.ch_names)}")
    # Normalize channel types for EOG if present
    eog_map = {}
    for k in ['HEOG', 'VEOG']:
        if k in raw.ch_names:
            eog_map[k] = 'eog'
    if eog_map:
        raw.set_channel_types(eog_map)
    # Usual capitalization for Fp1/Fp2 that appears as FP1/FP2 in some recordings
    rename = {}
    if 'FP1' in raw.ch_names:
        rename['FP1'] = 'Fp1'
    if 'FP2' in raw.ch_names:
        rename['FP2'] = 'Fp2'
    if rename:
        raw.rename_channels(rename)
    try:
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
    except Exception:
        pass
    print(f"[ATTEN][read] post-setup: n_ch={len(raw.ch_names)} (EOG may be present)")
    return raw


def _ica_eog_removal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    print(f"[ATTEN][ica] start: sfreq={raw.info.get('sfreq')}, n_ch={len(raw.ch_names)}, ENABLE_ICA={ENABLE_ICA}, DECIM={ICA_DECIM}, MAX_T={ICA_MAX_T}, RESAMPLE_BEFORE_ICA={RESAMPLE_BEFORE_ICA}")
    t0 = time.time()
    raw.load_data()

    if not ENABLE_ICA:
        print("[ATTEN][ica] skipping ICA per ATTEN_ICA=0; applying light filters only")
        try:
            cleaned = raw.copy()
            cleaned.filter(0.5, None, method='iir', verbose='ERROR')
            cleaned.filter(None, 50.0, method='iir', verbose='ERROR')
            print(f"[ATTEN][ica] skip done in {time.time()-t0:.2f}s")
            return cleaned
        except Exception as e:
            print(f"[ATTEN][ica] skip filters failed or skipped: {e}")
            return raw

    # Prepare data for ICA: filtered, optionally resampled and cropped
    work = raw.copy()
    if RESAMPLE_BEFORE_ICA and abs(work.info.get('sfreq', RESAMPLE_RATE) - RESAMPLE_RATE) > 1e-3:
        work.resample(RESAMPLE_RATE, npad='auto')
        print(f"[ATTEN][ica] resampled for ICA to {RESAMPLE_RATE} Hz")
    work = work.filter(l_freq=1.0, h_freq=None, verbose='ERROR')
    if ICA_MAX_T and ICA_MAX_T > 0:
        tmax = min(ICA_MAX_T, work.times[-1])
        work.crop(tmin=0.0, tmax=tmax, include_tmax=False)
        print(f"[ATTEN][ica] cropped for ICA to first {tmax:.2f}s")

    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
    t_fit0 = time.time()
    ica.fit(work, decim=max(1, ICA_DECIM), verbose='ERROR')
    print(f"[ATTEN][ica] fitted ICA with {ica.n_components_ if hasattr(ica, 'n_components_') else 'unknown'} components in {time.time()-t_fit0:.2f}s (total {time.time()-t0:.2f}s)")

    # Find EOG-related ICs if EOG channels exist
    try:
        eog_indices, _ = ica.find_bads_eog(raw, verbose='ERROR')
        ica.exclude = eog_indices
    except Exception as e:
        print(f"[ATTEN][ica] find_bads_eog failed: {e}")
        ica.exclude = []

    cleaned = raw.copy()
    t_apply0 = time.time()
    ica.apply(cleaned, verbose='ERROR')
    print(f"[ATTEN][ica] applied, excluded={ica.exclude}, apply_time={time.time()-t_apply0:.2f}s, total={time.time()-t0:.2f}s")

    # Gentle highpass and notch at 50 Hz
    try:
        cleaned.filter(0.5, None, method='iir', verbose='ERROR')
        cleaned.filter(None, 50.0, method='iir', verbose='ERROR')
    except Exception as e:
        print(f"[ATTEN][ica] post filters failed or skipped: {e}")
    return cleaned


def _extract_epochs_att_rest(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray, float]:
    # We expect blocks starting with code 48 (dataset-specific). We'll robustly locate it.
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    print(f"[ATTEN][events] got {len(events)} events, ids={list(event_id.keys())}")
    codes = events[:, 2]

    # Try to map a key that ends with ' 48' or equals '48'
    target_codes: List[int] = []
    for k, v in event_id.items():
        ks = str(k)
        if ks.endswith(' 48') or ks == '48' or ks.endswith('/S 48'):
            target_codes.append(v)
    if not target_codes:
        # Fallback: some BrainVision exporters write raw code 48 directly in events
        target_codes = [48]
    print(f"[ATTEN][events] target_codes={target_codes}")

    # Build attention and rest epochs around each start code (20s each)
    # Attention: [0, 20]s after the cue; Rest: [-20, 0]s before the next cue
    # We emulate the original script using the same 20s windows.
    # First, construct an events array that contains only the target code
    att_events = events[np.isin(codes, target_codes)]
    if len(att_events) == 0:
        print(f"[ATTEN][epochs] no attention events found")
        return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0, 0), dtype=np.float32), raw.info['sfreq']

    # For rest, use the same onsets but take the preceding 20s
    rest_events = att_events.copy()
    # Disable the very first trial's rest (no preceding window)
    rest_events[0, 2] = 0
    # Add a synthetic end mark to safely bound the final window
    last = events[-1, 0]
    rest_events = np.concatenate([rest_events, np.array([[last + int(20 * raw.info['sfreq']), 0, target_codes[0]]])])

    epochs_att = mne.Epochs(raw, att_events, event_id=dict(att=target_codes[0]), tmin=0.0, tmax=20.0,
                            baseline=None, preload=True, reject_by_annotation=False, verbose='ERROR')
    epochs_rest = mne.Epochs(raw, rest_events, event_id=dict(rest=target_codes[0]), tmin=-20.0, tmax=0.0,
                             baseline=None, preload=True, reject_by_annotation=False, verbose='ERROR')
    print(f"[ATTEN][epochs] att={len(epochs_att)} rest={len(epochs_rest)}")

    # Resample to project standard
    epochs_att.resample(RESAMPLE_RATE, npad='auto')
    epochs_rest.resample(RESAMPLE_RATE, npad='auto')

    X_att = epochs_att.get_data(copy=False).astype(np.float32)  # (N, C, T)
    X_rest = epochs_rest.get_data(copy=False).astype(np.float32)
    print(f"[ATTEN][epochs] X_att={X_att.shape} X_rest={X_rest.shape}")
    return X_att, X_rest, RESAMPLE_RATE


def _load_cnt_mrk_mat(sub_dir_path: str, task: str = 'wg') -> Tuple[np.ndarray, List[str], int, np.ndarray, np.ndarray]:
    """Load .mat continuous signals and markers.
    Returns: X (T, C), ch_names, fs, event_times (sample idx), event_labels (0/1 where 1=WG)
    """
    raise NotImplementedError("MAT processing has been removed; use BrainVision inputs only.")


def _epoch_mat(X_tc: np.ndarray, fs: int, times: np.ndarray, labels: np.ndarray, win_sec: int = 20) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Epoch 20s windows after each event time for both classes.
    Returns X (N, C, T), Y (N,), ch_names forwarded externally.
    """
    raise NotImplementedError("MAT processing has been removed; use BrainVision inputs only.")


def _segment_and_pipeline(X: np.ndarray, Y: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # X: (N, C, T) at RESAMPLE_RATE
    if X.size == 0:
        return np.zeros((0, 75, TIME_LENGTH * RESAMPLE_RATE), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    print(f"[ATTEN][segment] input X={X.shape}, Y={Y.shape}, n_ch={len(ch_names)}")
    trials = [X[i].T for i in range(X.shape[0])]  # (T, C)
    labels = Y.tolist()
    X_split, Y_split = split_trial(trials, labels, segment_length=TIME_LENGTH, overlap=0.0,
                                   sampling_rate=RESAMPLE_RATE, sub_segment=0, sub_overlap=0.0)
    seg_X, seg_Y = [], []
    for trial_segments, trial_labels in zip(X_split, Y_split):
        for seg, lab in zip(trial_segments, trial_labels):
            seg_X.append(seg.T)  # (C, T)
            seg_Y.append(lab)
    if len(seg_X) == 0:
        return np.zeros((0, 75, TIME_LENGTH * RESAMPLE_RATE), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    X_arr = np.array(seg_X, dtype=np.float32)
    Y_arr = np.array(seg_Y, dtype=np.uint8)
    # Map to template and robust-scale
    print(f"[ATTEN][segment] segments before map: X={X_arr.shape}, Y={Y_arr.shape}")
    X_arr = pipeline(X_arr, ch_names)
    print(f"[ATTEN][segment] after pipeline map: X={X_arr.shape}")
    return X_arr, Y_arr


def _find_session_vhdr(sub_dir_path: str, ses: int) -> str:
    # Search any BrainVision header file that corresponds to the given session index
    # Common patterns: gonogo1.vhdr, gonogo_1.vhdr, GoNogo1.vhdr
    pat = re.compile(r'(?i)gonogo\D*0?%d\.vhdr$' % ses)
    # search in the subject folder recursively (some exports nest files)
    print(f"[ATTEN][vhdr] searching session {ses} under {sub_dir_path}")
    for root, _, files in os.walk(sub_dir_path):
        for fn in files:
            if fn.lower().endswith('.vhdr') and pat.search(fn):
                path = os.path.join(root, fn)
                print(f"[ATTEN][vhdr] found session {ses}: {path}")
                return path
    # Fallback: if there is exactly one .vhdr and ses==1, take it
    vhdrs = []
    for root, _, files in os.walk(sub_dir_path):
        for fn in files:
            if fn.lower().endswith('.vhdr'):
                vhdrs.append(os.path.join(root, fn))
    if len(vhdrs) == 1 and ses == 1:
        print(f"[ATTEN][vhdr] fallback single vhdr for ses=1: {vhdrs[0]}")
        return vhdrs[0]
    print(f"[ATTEN][vhdr] no vhdr for session {ses}")
    return ''


def _proc_one_subject(sub_dir: str) -> Tuple[str, np.ndarray, np.ndarray]:
    # sub_dir: e.g., VP001
    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []

    sub_path = os.path.join(_dataset_root(), sub_dir)
    print(f"{sub_dir}: sub_path {sub_path}")
    # BrainVision path only (MAT branch removed)
    for ses in (1, 2, 3):
        try:
            print(f"[ATTEN][subject {sub_dir}] processing session {ses}")
            vhdr = _find_session_vhdr(sub_path, ses)
            if not vhdr:
                print(f"[ATTEN][subject {sub_dir}] session {ses} skipped: vhdr not found")
                continue
            raw = _safe_read_brainvision(vhdr)
            raw = _ica_eog_removal(raw)

            X_att, X_rest, _ = _extract_epochs_att_rest(raw)
            if X_att.size == 0:
                print(f"[ATTEN][subject {sub_dir}] session {ses}: no epochs extracted")
                continue
            X = np.concatenate([X_att, X_rest], axis=0)
            Y = np.array([1] * len(X_att) + [0] * len(X_rest), dtype=np.uint8)
            print(f"[ATTEN][subject {sub_dir}] session {ses}: concatenated X={X.shape}, Y={Y.shape}")

            eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
            ch_names = [raw.ch_names[p] for p in eeg_picks]
            if len(ch_names) != X.shape[1]:
                X = X[:, eeg_picks, :]
                print(f"[ATTEN][subject {sub_dir}] session {ses}: picked EEG channels -> X={X.shape}")

            X_seg, Y_seg = _segment_and_pipeline(X, Y, ch_names)
            if X_seg.size:
                print(f"{sub_dir}: vhdr segments {X_seg.shape}, labels {np.unique(Y_seg, return_counts=True)}")
                X_all.append(X_seg)
                Y_all.append(Y_seg)
            else:
                print(f"[ATTEN][subject {sub_dir}] session {ses}: no segments after pipeline")
        except Exception as e:
            print(f"[ATTEN][subject {sub_dir}] session {ses} ERROR: {e}\n{traceback.format_exc()}")

    if not X_all:
        return sub_dir, np.zeros((0, 75, TIME_LENGTH * RESAMPLE_RATE), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    X_cat = np.concatenate(X_all, axis=0)
    Y_cat = np.concatenate(Y_all, axis=0)
    return sub_dir, X_cat, Y_cat


def proc_all():
    root = _dataset_root()
    print(f"[ATTEN] root: {root}")
    subjects_src = _list_subjects(root)
    print(f"[ATTEN] subjects: {len(subjects_src)} -> {subjects_src[:5]}{'...' if len(subjects_src)>5 else ''}")
    # Keys: 001..N sorted by numeric order
    keys = [f'{i+1:03d}' for i in range(len(subjects_src))]
    sub_map = {src: key for src, key in zip(subjects_src, keys)}
    ATTEN_GoNogo.subjects = keys

    print(f"[ATTEN] starting multiprocessing pool with {WORKERS} workers for {len(subjects_src)} subjects")
    with mp.Pool(WORKERS) as pool:
        results = pool.map(_proc_one_subject, subjects_src)

    os.makedirs(DATA_FOLDER, exist_ok=True)
    h5_path = os.path.join(DATA_FOLDER, f'{NAME}.h5')
    with h5py.File(h5_path, 'w') as f:
        for src, X, Y in results:
            if X is None or X.size == 0:
                print(f"[ATTEN] skip {src}: no usable segments")
                continue
            key = sub_map[src]
            f.create_dataset(f'{key}/X', data=X)
            f.create_dataset(f'{key}/Y', data=Y)
            print(key, X.shape, Y.shape, np.unique(Y, return_counts=True))
    print(f"[ATTEN] write done -> {h5_path}")


if __name__ == '__main__':
    proc_all()
