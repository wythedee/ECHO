import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
torch.set_num_threads(16)
import h5py
import torch.nn as nn

def bold(x):       return '\033[1m'  + str(x) + '\033[0m'
def dim(x):        return '\033[2m'  + str(x) + '\033[0m'
def italicized(x): return '\033[3m'  + str(x) + '\033[0m'
def underline(x):  return '\033[4m'  + str(x) + '\033[0m'
def blink(x):      return '\033[5m'  + str(x) + '\033[0m'
def inverse(x):    return '\033[7m'  + str(x) + '\033[0m'
def gray(x):       return '\033[90m' + str(x) + '\033[0m'
def red(x):        return '\033[91m' + str(x) + '\033[0m'
def green(x):      return '\033[92m' + str(x) + '\033[0m'
def yellow(x):     return '\033[93m' + str(x) + '\033[0m'
def blue(x):       return '\033[94m' + str(x) + '\033[0m'
def magenta(x):    return '\033[95m' + str(x) + '\033[0m'
def cyan(x):       return '\033[96m' + str(x) + '\033[0m'
def white(x):      return '\033[97m' + str(x) + '\033[0m'

class H5Dataset:
    def __init__(self, root, name):
        self.root = root
        self.sfreq, self.X, self.Y = self.load(name)
        self.n_subjects = self.X.shape[0]

    def load(self, name):
        X, Y = [], []
        with h5py.File(f'{self.root}/{name}.h5', 'r') as f:
            for sub in f.keys():
                X.append(f[sub]['X'][()])
                Y.append(f[sub]['Y'][()])
        return 200, np.array(X), np.array(Y)
    
    def get_folds(self, train_idx, test_idx):
        train_X = np.concatenate([self.X[i] for i in train_idx])
        train_Y = np.concatenate([self.Y[i] for i in train_idx])
        test_X = [self.X[i] for i in test_idx]
        test_Y = [self.Y[i] for i in test_idx]
        return train_X, train_Y, test_X, test_Y

    def get_sub(self, idx):
        return self.X[idx], self.Y[idx]
    
def freeze(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

def all_exist(files):
    return all(os.path.exists(str(f)) for f in files)

class SimpleDataset(Dataset):
    def __init__(self, data, label):
        if len(data.shape) == 4:
            data, label = np.concatenate(data, axis=0), np.concatenate(label, axis=0).astype(np.int64)
        label = np.array(label, dtype=np.int64)
        self.data, self.labels = torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        return sample, label.long()

def cosine_lr(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def constant_lr(base_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    total_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters, endpoint=False)
    else:
        warmup_schedule = np.array([], dtype=float)

    const_iters = total_iters - warmup_iters
    constant_schedule = np.full(const_iters, base_value, dtype=float)

    schedule = np.concatenate((warmup_schedule, constant_schedule))
    assert len(schedule) == total_iters, "Schedule length mismatch."
    return schedule
