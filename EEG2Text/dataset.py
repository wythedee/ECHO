import numpy as np
import h5py
import os
class DSO_Dataset:
    def __init__(self, name = 'CS-rmEMG=k10-rmEOG=t3.5', root = '/media/james/public/DSO_PP2025'):
        self.name = name
        with np.load(root + f'/{name}.npz') as d:
            X, Y = d['X'], d['Y']
        self.sfreq = 200
        self.n_subjects = 57
        self.X = np.array(np.array_split(X, self.n_subjects))
        self.Y = np.array(np.array_split(Y, self.n_subjects))

    def __getitem__(self, idx):       # one subject
        return self.X[idx], self.Y[idx]

    def get_folds(self, train_idx, test_idx):
        train_X = np.concatenate([self.X[i] for i in train_idx])
        train_Y = np.concatenate([self.Y[i] for i in train_idx])
        test_X  = [self.X[i] for i in test_idx]
        test_Y  = [self.Y[i] for i in test_idx]
        return train_X, train_Y, test_X, test_Y
    
class h5Dataset:
    def __init__(self, name, classes, root='/home/workspace/EEG_Standardized_Group', pad=False, pad_length=None):
        self.sfreq, self.X, self.Y = self.load(name, root)
        if pad:
            self.pad_x(pad_length)
        self.n_subjects = len(self.X)
        self.classes = classes

    def load(self, name, root):
        X, Y = [], []
        with h5py.File(f'{root}/{name}.h5', 'r') as f:
            for sub in f.keys():
                X.append(f[sub]['X'][()])
                Y.append(f[sub]['Y'][()])
        return 250, X, Y
    
    def get_folds(self, train_idx, test_idx):
        train_X = np.concatenate([self.X[i] for i in train_idx])
        train_Y = np.concatenate([self.Y[i] for i in train_idx])
        test_X = np.concatenate([self.X[i] for i in test_idx])
        test_Y = np.concatenate([self.Y[i] for i in test_idx])
        
        # 创建subject映射 (key: subject_id, value: (start, end) range)
        train_subject_map = {}
        test_subject_map = {}
        
        # 记录train X中每个subject对应的范围
        current_idx = 0
        for subject_id in train_idx:
            n_trials = self.X[subject_id].shape[0]  # 该subject的trial数
            train_subject_map[subject_id] = (current_idx, current_idx + n_trials)
            current_idx += n_trials
        
        # 记录test X中每个subject对应的范围
        current_idx = 0
        for subject_id in test_idx:
            n_trials = self.X[subject_id].shape[0]  # 该subject的trial数
            test_subject_map[subject_id] = (current_idx, current_idx + n_trials)
            current_idx += n_trials
        
        return train_X, train_Y, test_X, test_Y, train_subject_map, test_subject_map

    def get_sub(self, idx):
        return self.X[idx], self.Y[idx]

    def pad_x(self, pad_length=None):
        """Zero-pad each subject's X in-place.

        Args:
            pad_length (int | None): target length for the last dimension.
                If None, use the maximum length among all subjects.
        """
        # determine target length
        if pad_length is None:
            pad_length = max(x.shape[2] for x in self.X)

        # pad each subject's trials to pad_length
        for i in range(len(self.X)):
            cur_len = self.X[i].shape[2]
            if cur_len < pad_length:
                self.X[i] = np.pad(self.X[i],
                                    ((0, 0), (0, 0), (0, pad_length - cur_len)),
                                    mode='constant',
                                    constant_values=0)

class SHHS_Dataset:
    def __init__(self, name = 'SLEEP_03_SHHS', root = '/home/workspace/dataset'):
        self.name = name
        self.path = root + f'/{name}' + '/shhs_npz'
        self.n_subjects = 0
        self.X = []
        self.Y = []
        self.subject_list = []
        self._load_data(self.path)

    def _load_data(self, dir_path):
        count = 0
        if self.n_subjects != 0:
            return self.n_subjects
        for file in os.listdir(dir_path):
            if file.endswith('.npz'):
                with np.load(self.path + f'/{file}') as d:
                    try:
                        X, Y = d['x'], d['y']
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue
                    self.X.append(X[:,:,:3000])
                    self.Y.append(Y)
                    print(f"{file} loaded, X shape: {X.shape}, Y shape: {Y.shape}, classes: {np.unique(Y)}")
                count += 1
            # if count == 10:
            #     break
        self.n_subjects = count
        return self.n_subjects

    def get_folds(self, train_idx, test_idx):
        train_X = np.concatenate([self.X[i] for i in train_idx])
        train_Y = np.concatenate([self.Y[i] for i in train_idx])
        test_X = [self.X[i] for i in test_idx]
        test_Y = [self.Y[i] for i in test_idx]
        return train_X, train_Y, test_X, test_Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class EMO_3_Dataset:
    def __init__(self, name, root = '/home/workspace/EEG_Standardized_Group'):
        self.sfreq, self.X, self.Y = self.load(name, root)
        self.name = name
        self.n_subjects = self.X.shape[0]

    def load(self, name, root):
        X, Y = [], []
        with h5py.File(f'{root}/{name}.h5', 'r') as f:
            for sub in f.keys():
                X.append(f[sub]['X'][()])
                Y.append(f[sub]['Y'][()])
        return 250, np.array(X), np.array(Y)
    
    def get_folds(self, train_idx, test_idx):
        train_X = np.concatenate([self.X[i] for i in train_idx])
        train_Y = np.concatenate([self.Y[i] for i in train_idx])
        test_X = [self.X[i] for i in test_idx]
        test_Y = [self.Y[i] for i in test_idx]
        return train_X, train_Y, test_X, test_Y

    def get_sub(self, idx):
        return self.X[idx], self.Y[idx]
    
class EMO_Dataset:
    def __init__(self, name, classes, root = '/home/workspace/EEG_Standardized_Group', pad=False, pad_length=None):
        self.sfreq, self.X, self.Y = self.load(name, root)
        if pad:
            self.pad_x(pad_length)
        self.name = name
        self.n_subjects = len(self.X)
        self.classes = classes

    def load(self, name, root):
        X, Y = [], []
        with h5py.File(f'{root}/{name}.h5', 'r') as f:
            for sub in f.keys():
                X.append(f[sub]['X'][()])
                Y.append(f[sub]['Y'][()])
        return 250, X, Y
    
    def get_folds(self, train_idx, test_idx):
        train_X = np.concatenate([self.X[i] for i in train_idx])
        train_Y = np.concatenate([self.Y[i] for i in train_idx])
        test_X  = np.concatenate([self.X[i] for i in test_idx])
        test_Y  = np.concatenate([self.Y[i] for i in test_idx])
        
        # 创建subject映射 (key: subject_id, value: (start, end) range)
        train_subject_map = {}
        test_subject_map = {}
        
        # 记录train X中每个subject对应的范围
        current_idx = 0
        for subject_id in train_idx:
            n_trials = self.X[subject_id].shape[0]  # 该subject的trial数
            train_subject_map[subject_id] = (current_idx, current_idx + n_trials)
            current_idx += n_trials
        
        # 记录test X中每个subject对应的范围
        current_idx = 0
        for subject_id in test_idx:
            n_trials = self.X[subject_id].shape[0]  # 该subject的trial数
            test_subject_map[subject_id] = (current_idx, current_idx + n_trials)
            current_idx += n_trials
        
        return train_X, train_Y, test_X, test_Y, train_subject_map, test_subject_map

    def get_sub(self, idx):
        return self.X[idx], self.Y[idx]
    
    def pad_x(self, pad_length=None):
        """Zero-pad each subject's samples to `pad_length` (time dimension)."""
        if pad_length is None:
            pad_length = max(x.shape[2] for x in self.X)

        for i in range(len(self.X)):
            cur_len = self.X[i].shape[2]
            if cur_len < pad_length:
                self.X[i] = np.pad(
                    self.X[i],
                    ((0, 0), (0, 0), (0, pad_length - cur_len)),
                    mode='constant',
                    constant_values=0,
                )

class MultiSourceDataset:
    def __init__(self, udas, resample_rate, time_length, classes):
        self.udas = udas
        self.resample_rate = resample_rate
        self.time_length = time_length  
        self.classes = classes
        self.X = []
        self.Y = []
        self.n_classes = 0
        for i, uda in enumerate(udas):
            # (n_subjects, n_trials, n_channels, n_samples), elements: (n_trials, n_channels, n_samples)
            self.X.extend(uda.X)
            # (n_subjects, n_trials), elements: (n_trials, )
            new_y = self.reindex_y(uda.Y, i)
            self.Y.extend(new_y)
        self.pad_x()
        self.n_subjects = len(self.X)
    
    def pad_x(self):
        max_length = max([x.shape[2] for x in self.X])  # 改为shape[2]
        for i in range(len(self.X)):
            if self.X[i].shape[2] < max_length:  # 改为shape[2]
                self.X[i] = np.pad(self.X[i], 
                                ((0, 0), (0, 0), (0, max_length - self.X[i].shape[2])),  # 添加第三个维度的填充
                                'constant', 
                                constant_values=0)
        return self.X
    
    def reindex_y(self, y, index):
        y_new = []
        # y: (n_subjects, n_trials), elements: (n_trials, )
        for i in range(len(y)):
            y_new.append(self.n_classes + y[i])
        self.n_classes += len(self.classes[index])
        return y_new
    
    def get_folds(self, train_idx, test_idx):
        train_X = np.concatenate([self.X[i] for i in train_idx])
        train_Y = np.concatenate([self.Y[i] for i in train_idx])
        test_X = np.concatenate([self.X[i] for i in test_idx])
        test_Y = np.concatenate([self.Y[i] for i in test_idx])
        
        # 创建subject映射 (key: subject_id, value: (start, end) range)
        train_subject_map = {}
        test_subject_map = {}
        
        # 记录train X中每个subject对应的范围
        current_idx = 0
        for subject_id in train_idx:
            n_trials = self.X[subject_id].shape[0]  # 该subject的trial数
            train_subject_map[subject_id] = (current_idx, current_idx + n_trials)
            current_idx += n_trials
        
        # 记录test X中每个subject对应的范围
        current_idx = 0
        for subject_id in test_idx:
            n_trials = self.X[subject_id].shape[0]  # 该subject的trial数
            test_subject_map[subject_id] = (current_idx, current_idx + n_trials)
            current_idx += n_trials
        
        return train_X, train_Y, test_X, test_Y, train_subject_map, test_subject_map
    
    def get_sub(self, idx):
        return self.X[idx], self.Y[idx]
    
if __name__ == '__main__':
    dataset = EMO_3_Dataset('EMO_03_SEED_V')
    print(dataset.X.shape)
    print(dataset.Y.shape)
    print(dataset.get_folds([0, 1, 2], [3, 4, 5]))
