import json
import numpy as np

ch75 = {
    "A1":  ["T9","M1","A1"],
    "A2":  ["T10","M2","A2"],
    "TP9": ["TP9","TTP9h","TTP9","TPP9"],
    "TP10":["TP10","TTP10h","TTP10","TPP10"],
    "F9":  ["AF9","F9","F9h","AFF9"],
    "F10": ["AF10","F10","AFF10"],
    "Fp1": ["Fp1","AFp7h","AFp5h","AFp3h","Fp1h","AFp5","AFp3"],
    "Fp2": ["Fp2","AFp6h","AFp8h","Fp2h","AFp4","AFp6"],
    "Fpz": ["Fpz","AFp1h","AFp2h","AFp1","AFp2"],
    "F7":  ["F7","AFF9h","FFT7h"],
    "F3":  ["F3","F5h","F3h","AFF3","FFC3"],
    "Fz":  ["Fz","FFC1h","F1h"],
    "F4":  ["F4","AFF6h","F6h","AFF4","FFC4"],
    "F8":  ["F8","AFF10h","FFT8h","F10h"],
    "FC5": ["FC5","FT7h","FC5h"],
    "FC1": ["FC1","FCC1"],
    "FC2": ["FC2","FFC2h","FCC4h","FC2h","FC4h","FFC2","FCC2"],
    "FC6": ["FC6","FCC6"],
    "T3":  ["T7","T9h","FTT7","T3"],
    "C3":  ["C3","FCC3h","C5h","FCC3","CCP3"],
    "Cz":  ["Cz","C1h","CCPz"],
    "C4":  ["C4","C6h","CCP4"],
    "T4":  ["T8","FTT10h","T8h","T10h","TTP8","T4"],
    "CP5": ["CP5","CCP5h"],
    "CP1": ["CP1","CCP1h"],
    "CP2": ["CP2","CP2h","CP4h"],
    "CP6": ["CP6","CCP6h","CP6h","TP8h","CPP6"],
    "T5":  ["P7","TPP7h","P9h","TPP7","T5"],
    "P3":  ["P3","P5h","P3h","CPP3","PPO3"],
    "Pz":  ["Pz","CPP1h","P1h","CPPz"],
    "P4":  ["P4","CPP4h","PPO4h","P4h","P6h","CPP4","PPO4"],
    "T6":  ["P8","TPP8h","P8h","P10h","T6"],
    "POz": ["PO1","POz","PPO1h","PPO2h","PO1h","PO2h","PPOz"],
    "O1":  ["O1","POO7h","POO5h","OI1h","O1h","I1h","POO7","POO5","OI1"],
    "Oz":  ["Oz","Iz","POO1h","POO2h","O2h","POO1","POOz","POO2","OIz"],
    "O2":  ["O2","POO6h","POO8h","OI2h","I2h","POO6","POO8","OI2"],
    "AF7": ["AF7","AF5","AFp9h","AFF7h","AF9h","AF7h","AFp9","AFp7","AFF7"],
    "AF3": ["AF3","AFF5h","AFF3h","AF5h","AF3h"],
    "AF4": ["AF2","AF4","AFp4h","AFF4h","AF4h","AF6h"],
    "AF8": ["AF6","AF8","AFp10h","AFF8h","AF8h","AF10h","AFp8","AFp10","AFF8"],
    "F5":  ["F5","FFC5h","F7h","AFF5","FFC5"],
    "F1":  ["F1","FFC3h","AFF1","FFC1"],
    "F2":  ["F2","AFF2h","FFC4h","F2h","F4h","AFF2"],
    "F6":  ["F6","FFC6h","F8h","AFF6","FFC6"],
    "FC3": ["FC3","FC3h"],
    "FCz": ["FCz","FCC2h","FC1h","FFCz","FCCz"],
    "FC4": ["FC4","FC6h","FCC4"],
    "C5":  ["C5","FCC5h","T7h","FCC5","CCP5"],
    "C1":  ["C1","FCC1h","C3h","CCP1"],
    "C2":  ["C2","CCP2h","C2h","C4h","CCP2"],
    "C6":  ["C6","FCC6h","CCP6"],
    "CP3": ["CP3","CCP3h","CP5h","CP3h"],
    "CPz": ["CPz","CP1h"],
    "CP4": ["CP4","CCP4h"],
    "P5":  ["P5","CPP5h","P7h","CPP5"],
    "P1":  ["P1","CPP3h","CPP1","PPO1"],
    "P2":  ["P2","CPP2h","P2h","CPP2","PPO2"],
    "P6":  ["P6","CPP6h","PPO6"],
    "PO5": ["PO5","PPO5h","PO7h","PO5h","PPO5"],
    "PO3": ["PO3","PPO3h","POO3h","PO3h","POO3"],
    "PO4": ["PO2","PO4","POO4h","PO4h","POO4"],
    "PO6": ["PO6","PPO6h","PO6h"],
    "FT7": ["FT7","FTT9h","FTT7h","FFT7"],
    "FT8": ["FT8","FFT10h","FTT8h","FT8h","FT10h","FFT8","FTT8"],
    "TP7": ["TP7","TTP7h","TP9h","TP7h","TTP7"],
    "TP8": ["TP8","TTP8h","TP10h","TPP8"],
    "PO7": ["PO7","PPO7h","POO9h","PO9h","PPO7"],
    "PO8": ["PO8","PPO8h","POO10h","PO8h","PO10h","PPO8"],
    "FT9": ["FT9","FFT9h","FT9h","FFT9","FTT9"],
    "FT10":["FT10","FFT10","FTT10"],
    "PO9": ["PO9","I1","PPO9h","PPO9","POO9"],
    "PO10":["PO10","I2","PPO10h","POO10"],
    "P9":  ["P9","TPP9h"],
    "P10": ["P10","TPP10h","PPO10"],
    "AFz": ["AF1","AFz","AFF1h","AF1h","AF2h","AFpz","AFFz"]
}

class AdaptiveGrouping:
    def __init__(self, name):
        templates = {'ch75': ch75}
        self.template_dict = templates[name]
        self.template_channels = list(self.template_dict.keys())
        self.lookup_template_ch_by_src = self.create_lookup()

    def create_lookup(self):
        lookup = {}
        for k, v in self.template_dict.items():
            for ch in v:
                lookup[ch] = k
        return lookup

    def map_trial(self, X, ch_names):
        assert len(ch_names) == X.shape[0], f'{len(ch_names)} != {X.shape[0]}'
        newX = np.zeros((len(self.template_channels), X.shape[1]), dtype=np.float32)
        num = np.ones(len(self.template_channels), dtype=np.float32)
        for x, ch in zip(X, ch_names):
            idx = self.template_channels.index(self.lookup_template_ch_by_src[ch])
            newX[idx] += x
            num[idx] += 1
        return newX / num[:, np.newaxis]

    def map_to_template(self, X, ch_names):
        # 将多个试验映射到模板
        # new_ch_names = list(self.template_dict.keys())
        newX = np.array([self.map_trial(x, ch_names) for x in X])
        return newX

if __name__ == '__main__':
    adaptive_group = AdaptiveGrouping('ch75')
    X_example = np.random.randn(10, 100)  # 10 channels, 100 samples
    channel_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10']
    mapped_X = adaptive_group.map_to_template(X_example, channel_names)
