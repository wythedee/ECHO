import os
import pickle
import numpy as np
import h5py
import multiprocessing as mp
import time
import pyedflib.highlevel as hl
from mne.filter import resample as mne_resample
from collections import defaultdict
from typing import Dict, List, Tuple

# Keep framework style
from share import META, SRC_FOLDER, DATA_FOLDER, pipeline


# Source/Output roots (follow project convention)
SRC_FOLDER = os.path.join(SRC_FOLDER, 'EP')
DATA_FOLDER = os.path.join(DATA_FOLDER, 'EP')
NAME = 'EP_01_CHBMIT'

# Expect raw data layout:
# {SRC_FOLDER}/{NAME}/chbXX/chbXX_YY.edf and chbXX-summary.txt
SIGNALS_PATH = os.path.join(SRC_FOLDER, NAME)
PROCESSED_PATH = os.path.join(DATA_FOLDER, f'{NAME}_processed')
TMP_SEG_DIR = os.path.join(DATA_FOLDER, f'{NAME}_tmp_segments')

# Logging and multiprocessing controls
QUIET = bool(int(os.environ.get('EP_QUIET', '0')))  # 1 to suppress noisy skips
MAX_WORKERS_CLEAN = int(os.environ.get('EP_MAX_WORKERS_CLEAN', '8'))
MAX_WORKERS_SEG = int(os.environ.get('EP_MAX_WORKERS_SEG', '8'))


# 16 bipolar channels present in CHB-MIT (order is fixed for data loading)
CH_NAMES_BIPOLAR = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
]

# Map each bipolar derivation to a unique 10-10 scalp name accepted by pipeline
# Note: this is a pragmatic spatial mapping, not a physical reconstruction of referential signals
CHANNEL_NAME_MAP = {
    'FP1-F7': 'Fp1',
    'F7-T7': 'FT7',
    'T7-P7': 'TP7',
    'P7-O1': 'PO7',
    'FP2-F8': 'Fp2',
    'F8-T8': 'FT8',
    'T8-P8': 'TP8',
    'P8-O2': 'PO8',
    'FP1-F3': 'F3',
    'F3-C3': 'C3',
    'C3-P3': 'CP3',
    'P3-O1': 'PO3',
    'FP2-F4': 'F4',
    'F4-C4': 'C4',
    'C4-P4': 'CP4',
    'P4-O2': 'PO4',
}

# The channel names passed to pipeline should be these mapped names (length must match data channels)
CH_NAMES_MAPPED = [CHANNEL_NAME_MAP[k] for k in CH_NAMES_BIPOLAR]

TEXT_LABELS = ['EP/NonSeizure', 'EP/Seizure']
SAMPLING_RATE = 256
TIME_LENGTH = 10  # seconds

# Default subjects (dataset has 24 subjects). H5 may include a subset depending on availability.
SUBJECTS = [f'chb{i:02d}' for i in range(1, 25)]


EP_CHBMIT = META(NAME, CH_NAMES_MAPPED, SUBJECTS, TEXT_LABELS, resample_rate=250, time_length=TIME_LENGTH)


# -------------------------------
# Step 1: Clean EDFs and unify channels per subject (from process1.py)
# -------------------------------

def _compressed_pickle(path: str, data: dict) -> None:
    # Atomic write to avoid partial files across processes
    tmp_path = f"{path}.{os.getpid()}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _process_metadata(summary_path: str, filename: str) -> Dict:
    # Parse summary to get seizures (count and (start,end) in samples)
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    metadata = {}
    times: List[Tuple[int, int]] = []
    seizures = 0
    for i in range(len(lines)):
        parts = lines[i].split()
        if len(parts) == 3 and parts[2] == filename:
            j = i + 1
            processed = False
            while not processed:
                if lines[j].split()[0] == 'Number':
                    seizures = int(lines[j].split()[-1])
                    processed = True
                j += 1
            if seizures > 0:
                j = i + 1
                for _ in range(seizures):
                    processed = False
                    while not processed:
                        l = lines[j].split()
                        if len(l) > 0 and l[0] == 'Seizure' and 'Start' in l:
                            start = int(l[-2]) * SAMPLING_RATE - 1
                            end = int(lines[j + 1].split()[-2]) * SAMPLING_RATE - 1
                            processed = True
                        j += 1
                    times.append((start, end))
            break
    metadata['seizures'] = seizures
    metadata['times'] = times
    return metadata


def _drop_channels(edf_source: str, to_keep: List[int]) -> Dict[str, np.ndarray]:
    signals, signal_headers, _ = hl.read_edf(edf_source, ch_nrs=to_keep, digital=False)
    clean_file: Dict[str, np.ndarray] = {}
    for signal, header in zip(signals, signal_headers):
        channel = header.get('label')
        if channel in clean_file:
            channel = channel + '-2'
        clean_file[channel] = signal.astype(float)
    return clean_file


def _move_channels(clean_dict: Dict[str, np.ndarray], channels: Dict[str, List[str]], target: str) -> None:
    keys_to_delete = []
    for key in clean_dict:
        if key != 'metadata' and key not in channels:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del clean_dict[key]

    size = 0
    for item in clean_dict.keys():
        if item != 'metadata':
            size = len(clean_dict[item])
            break

    for k in channels.keys():
        if k not in clean_dict:
            clean_dict[k] = np.zeros(size, dtype=float)

    _compressed_pickle(target + '.pkl', clean_dict)


def _process_files(p: str, valid_channels: List[str], channels: Dict[str, List[str]], start: int, end: int) -> None:
    for num in range(start, end + 1):
        to_keep: List[int] = []
        n_str = f'{num:02d}'
        filename = f"{SIGNALS_PATH}/chb{p}/chb{p}_{n_str}.edf"
        try:
            _, signal_headers, _ = hl.read_edf(filename, digital=False)
            n = 0
            for h in signal_headers:
                if h.get('label') in valid_channels:
                    if n not in to_keep:
                        to_keep.append(n)
                n += 1
        except OSError:
            if not QUIET:
                print(f"[Skip] File not found: {filename}")
            continue

        if len(to_keep) == 0:
            if not QUIET:
                print(f"[Skip] No valid channel for {filename}")
            continue

        try:
            clean_dict = _drop_channels(filename, to_keep=to_keep)
        except AssertionError:
            if not QUIET:
                print(f"[Skip] Assertion reading: {filename}")
            continue

        metadata = _process_metadata(
            f"{SIGNALS_PATH}/chb{p}/chb{p}-summary.txt",
            f"chb{p}_{n_str}.edf",
        )
        metadata['channels'] = valid_channels
        clean_dict['metadata'] = metadata
        target = f"{PROCESSED_PATH}/chb{p}/chb{p}_{n_str}.edf"
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _move_channels(clean_dict, channels, target)


def _start_process(p: str, ref_num: str, start: int, end: int, sum_ind: int) -> None:
    # Read summary to build consistent channels
    with open(f"{SIGNALS_PATH}/chb{p}/chb{p}-summary.txt", 'r') as f:
        lines = f.readlines()

    channels: Dict[str, List[str]] = defaultdict(list)
    valid_channels: List[str] = []
    to_keep: List[int] = []
    channel_index = 1
    summary_index = 0

    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue
        if parts[0] == 'Channels' and parts[1] == 'changed:':
            summary_index += 1
        if parts[0] == 'Channel' and summary_index == sum_ind and (parts[2] != '-' and parts[2] != '.'):
            name = parts[2] if parts[2] not in channels else parts[2] + '-2'
            channels[name].append(str(channel_index))
            channel_index += 1
            valid_channels.append(name)
            to_keep.append(int(parts[1][:-1]) - 1)

    ref_num_str = f'{int(ref_num):02d}'
    ref_filename = f"{SIGNALS_PATH}/chb{p}/chb{p}_{ref_num_str}.edf"
    target = f"{PROCESSED_PATH}/chb{p}/chb{p}_{ref_num_str}.edf"
    os.makedirs(os.path.dirname(target), exist_ok=True)

    clean_dict = _drop_channels(ref_filename, to_keep=to_keep)
    metadata = _process_metadata(
        f"{SIGNALS_PATH}/chb{p}/chb{p}-summary.txt",
        f"chb{p}_{ref_num_str}.edf",
    )
    metadata['channels'] = valid_channels
    clean_dict['metadata'] = metadata
    _compressed_pickle(target + '.pkl', clean_dict)

    _process_files(p, valid_channels, channels, start, end)


# Patient-specific parameters (from process1.py)
_PARAMETERS = [
    ('01', '01', 2, 46, 0),
    ('02', '01', 2, 35, 0),
    ('03', '01', 2, 38, 0),
    ('05', '01', 2, 39, 0),
    ('06', '01', 2, 24, 0),
    ('07', '01', 2, 19, 0),
    ('08', '02', 3, 29, 0),
    ('10', '01', 2, 89, 0),
    ('11', '01', 2, 99, 0),
    ('14', '01', 2, 42, 0),
    ('20', '01', 2, 68, 0),
    ('21', '01', 2, 33, 0),
    ('22', '01', 2, 77, 0),
    ('23', '06', 7, 20, 0),
    ('24', '01', 3, 21, 0),
    ('04', '07', 1, 43, 1),
    ('09', '02', 1, 19, 1),
    ('15', '02', 1, 63, 1),
    ('16', '01', 2, 19, 0),
    ('18', '02', 1, 36, 1),
    ('19', '02', 1, 30, 1),
]


def run_clean_stage(parallel: bool = True) -> None:
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    if parallel:
        ctx = mp.get_context('spawn')
        workers = max(1, min(MAX_WORKERS_CLEAN, mp.cpu_count()))
        with ctx.Pool(workers) as pool:
            pool.starmap(_start_process, _PARAMETERS)
    else:
        for args in _PARAMETERS:
            _start_process(*args)


# -------------------------------
# Step 2: Segment into 10s windows and label (from process2.py)
# -------------------------------

def _intersects(seg_start: int, seg_end: int, seizure_start: int, seizure_end: int) -> bool:
    # Intersection test
    return not (seg_end <= seizure_start or seg_start >= seizure_end)


def _normalize_clip_iqr(X: np.ndarray) -> np.ndarray:
    # Robust clip + IQR normalization
    lo, hi = np.percentile(X, 0.1), np.percentile(X, 99.9)
    X = np.clip(X, lo, hi)
    median = np.median(X)
    q25, q75 = np.percentile(X, 25), np.percentile(X, 75)
    iqr = q75 - q25
    if iqr == 0:
        iqr = 1.0
    return (X - median) / iqr


def _load_record_to_array(record: dict, bipolar_order: List[str]) -> np.ndarray:
    xs: List[np.ndarray] = []
    for ch in bipolar_order:
        if ch in record:
            xs.append(np.asarray(record[ch], dtype=np.float32))
        else:
            raise ValueError(f'Missing channel {ch} in record')
    return np.stack(xs, axis=0)


def _safe_pickle_load(path: str, retries: int = 3, base_delay: float = 0.2):
    """Robust pickle loader with retries for transient I/O errors."""
    attempt = 0
    last_err = None
    while attempt <= retries:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
            last_err = e
            if attempt == retries:
                break
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            attempt += 1
    if not QUIET:
        print(f"[Skip] Failed to read pickle after retries: {path} ({type(last_err).__name__}: {last_err})")
    return None


def _segment_subject(sub_folder: str) -> Tuple[str, str, int]:
    # sub_folder: path to processed/chbXX
    sub = os.path.basename(sub_folder)
    files = [f for f in os.listdir(sub_folder) if f.endswith('.pkl')]
    files.sort()

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for f in files:
        record = _safe_pickle_load(os.path.join(sub_folder, f))
        if record is None:
            continue
        signal = _load_record_to_array(record, CH_NAMES_BIPOLAR)  # (C, T) in bipolar order
        seizure_times = record.get('metadata', {}).get('times', [])

        # Regular 10s stride 10s
        step = SAMPLING_RATE * TIME_LENGTH
        for i in range(0, signal.shape[1], step):
            seg = signal[:, i:i + step]
            if seg.shape[1] != step:
                continue
            # Resample 2560 -> 2500 on time axis
            seg64 = seg.astype(np.float64, copy=False)
            seg = mne_resample(seg64, down=128, up=125, axis=1).astype(np.float32, copy=False)
            label = 0
            for (ss, ee) in seizure_times:
                if _intersects(i, i + step, ss, ee):
                    label = 1
                    break
            X_list.append(seg)
            y_list.append(label)

        # Extra positive sampling around seizures (5s step)
        for idx, (ss, ee) in enumerate(seizure_times):
            start_i = max(0, ss - SAMPLING_RATE)
            end_i = min(ee + SAMPLING_RATE, signal.shape[1])
            for i in range(start_i, end_i, 5 * SAMPLING_RATE):
                seg = signal[:, i:i + step]
                if seg.shape[1] != step:
                    continue
                # Resample 2560 -> 2500 on time axis
                seg64 = seg.astype(np.float64, copy=False)
                seg = mne_resample(seg64, down=128, up=125, axis=1).astype(np.float32, copy=False)
                X_list.append(seg)
                y_list.append(1)

    os.makedirs(TMP_SEG_DIR, exist_ok=True)
    tmp_path = os.path.join(TMP_SEG_DIR, f'{sub}.h5')

    if len(X_list) == 0:
        # write empty datasets for consistency (250 Hz * 10 s = 2500)
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('X', data=np.empty((0, len(CH_NAMES_MAPPED), 250 * TIME_LENGTH), dtype=np.float32))
            f.create_dataset('Y', data=np.empty((0,), dtype=np.uint8))
        return sub, tmp_path, 0

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.uint8)

    # Map to template space using pipeline (expects recognized 10-10 names)
    X = pipeline(X, CH_NAMES_MAPPED)

    with h5py.File(tmp_path, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=y)
    return sub, tmp_path, int(X.shape[0])


def run_segment_stage(write_h5: bool = True) -> Tuple[List[Tuple[str, str, int]], str]:
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f'Processed path not found: {PROCESSED_PATH}. Run run_clean_stage() first.')

    sub_folders = [os.path.join(PROCESSED_PATH, d) for d in os.listdir(PROCESSED_PATH) if os.path.isdir(os.path.join(PROCESSED_PATH, d))]
    sub_folders.sort()

    ctx = mp.get_context('spawn')
    workers = max(1, min(MAX_WORKERS_SEG, mp.cpu_count()))
    with ctx.Pool(workers) as pool:
        res = pool.map(_segment_subject, sub_folders)

    if write_h5:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        h5_path = os.path.join(DATA_FOLDER, f'{NAME}.h5')
        with h5py.File(h5_path, 'w') as fout:
            for sub, tmp_path, n in res:
                if n == 0:
                    continue
                with h5py.File(tmp_path, 'r') as fin:
                    X = fin['X'][()]
                    Y = fin['Y'][()]
                    fout.create_dataset(f'{sub}/X', data=X)
                    fout.create_dataset(f'{sub}/Y', data=Y)
                    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
        # optional: cleanup tmp files
        try:
            for sub, tmp_path, _ in res:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            if os.path.isdir(TMP_SEG_DIR) and len(os.listdir(TMP_SEG_DIR)) == 0:
                os.rmdir(TMP_SEG_DIR)
        except Exception:
            pass
        return res, h5_path
    else:
        return res, ''


def proc_all() -> None:
    # Full pipeline
    run_clean_stage(parallel=True)
    run_segment_stage(write_h5=True)


if __name__ == '__main__':
    proc_all()


