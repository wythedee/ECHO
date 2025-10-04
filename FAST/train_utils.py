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
    return all(os.path.exists(f) for f in files)

class SimpleDataset(Dataset):
    def __init__(self, data, label):
        if len(data.shape) == 4:
            data, label = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
        self.data, self.labels = torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        return sample, label

def convert_epoch_acc_to_npy(all_csv, npy_path):
    if all(os.path.exists(f) for f in all_csv):
        np.save(npy_path, np.array([np.loadtxt(f, delimiter=',') for f in all_csv]).astype(np.float32))
        for f in all_csv:
            os.remove(f)

def convert_logits_to_npy(all_logits, npy_path):
    if all(os.path.exists(f) for f in all_logits):
        np.save(npy_path, np.concatenate([np.load(f) for f in all_logits]))
        for f in all_logits:
            os.remove(f)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def inference_on_loaders(model, loaders, forward_mode):
    model.eval()
    model.cuda()
    with torch.no_grad():
        Pred, Real = [], []
        for loader in loaders:
            _pred, _true = [], []
            for x, y in loader:
                preds = torch.argmax(model(x.cuda(), forward_mode)[0], dim=1).cpu()
                _pred.append(preds)
                _true.append(y)
            Pred.append(torch.cat(_pred).cpu())
            Real.append(torch.cat(_true).cpu())
    return Pred, Real