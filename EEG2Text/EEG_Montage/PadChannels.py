import numpy as np
import mne

EXCLUDE_REMAP = {
    'T7': 'T3',
    'T8': 'T4',
    'P7': 'T5',
    'P8': 'T6',
}

TEMPLATE_CH = [
    'A1','A2', 
    'TP9', 'TP10', 'F9', 'F10',
    'AFp1', 'AFp2', 'AFF1h', 'AFF2h',
    'PPO1h', 'PPO2h', 'POO1', 'POO2',
    'Fp1','Fp2','Fpz','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
    'M1','T3','C3','Cz','C4','T4','M2','CP5','CP1','CP2','CP6','T5','P3',
    'Pz','P4','T6','POz','O1','Oz','O2','AF7','AF3','AF4','AF8','F5','F1',
    'F2','F6','FC3','FCz','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P5',
    'P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8',
    'FT9','FT10','TPP9h','TPP10h','PO9','PO10','P9','P10','AFF1','AFz','AFF2',
    'FFC5h','FFC3h','FFC4h','FFC6h','FCC5h','FCC3h','FCC4h','FCC6h','CCP5h','CCP3h',
    'CCP4h','CCP6h','CPP5h','CPP3h','CPP4h','CPP6h','PPO1','PPO2','I1','Iz','I2','AFp3h',
    'AFp4h','AFF5h','AFF6h','FFT7h','FFC1h','FFC2h','FFT8h','FTT9h','FTT7h','FCC1h',
    'FCC2h','FTT8h','FTT10h','TTP7h','CCP1h','CCP2h','TTP8h','TPP7h','CPP1h','CPP2h',
    'TPP8h','PPO9h','PPO5h','PPO6h','PPO10h','POO9h','POO3h','POO4h','POO10h','OI1h','OI2h'
]
assert not any([ch in EXCLUDE_REMAP.keys() for ch in TEMPLATE_CH]), f'not include {EXCLUDE_REMAP.keys()}'

def rename_ch(ch_names):
    return [EXCLUDE_REMAP[ch] if ch in EXCLUDE_REMAP else ch for ch in ch_names]

def map_to_template(X, ch_names):
    ch_names = rename_ch(ch_names)
    newX = np.zeros((X.shape[0], len(TEMPLATE_CH), X.shape[2]), dtype=np.float32)
    for ch in ch_names:
        if ch not in TEMPLATE_CH:
            raise ValueError(f'not in TEMPLATE_CH: {ch}')
        
    bads = []
    for i, ch in enumerate(TEMPLATE_CH):
        if ch not in ch_names:
            bads.append(ch)
            continue
        idx = ch_names.index(ch)
        newX[:, i] = X[:, idx]
        
    return newX, bads

def interpolate_EEG(X, bads):
    info = mne.create_info(ch_names=TEMPLATE_CH, sfreq=250, ch_types='eeg')
    info['bads'] = bads
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage)
    epochs = mne.EpochsArray(X, info)
    epochs = epochs.interpolate_bads(reset_bads=True)
    return epochs.get_data().astype(np.float32)