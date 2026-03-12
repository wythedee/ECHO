import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .utils import pad_trial
import random as _rnd
from typing import List as _List, Tuple as _Tuple
from collections import OrderedDict
import warnings

class ICLWithSupportDataset_Test(Dataset):
    def __init__(self, EEG, Text, tokenizer, trials_per_subject: int = 100, k_per_class: int = 3, max_len: int = 256, subject_map: dict = None, fixed_support_num: int = None):
        """
        EEG test dataset, including ICL support and SOT markers
        
        Parameters:
            EEG_data: EEG signal data, shape (N, C, T)
            labels: List of labels
            tokenizer: Whisper tokenizer object
            k_per_class: Number of samples to take for each label
            max_len: Maximum sequence length
            subject_map: Subject mapping dictionary
            fixed_support_num: Fixed number of support samples
        """
        super().__init__()
        assert EEG.shape[0] == len(Text)
        self.k_per_class = k_per_class
        self.max_len = max_len
        self.trials_per_subject = trials_per_subject
        self.fixed_support_num = fixed_support_num
        
        self.total_trials = EEG.shape[0]
        self.unique_labels = list(set(Text))
        
        if subject_map:
            # Use subject_map mode
            self.n_subjects = len(subject_map)
            self.EEG = [None] * (max(subject_map.keys()) + 1)  # Ensure index is large enough
            self.Text = [None] * (max(subject_map.keys()) + 1)
            self.subject_map = subject_map
            
            for subject_id, (start, end) in subject_map.items():
                self.EEG[subject_id] = torch.from_numpy(EEG[start:end])
                self.Text[subject_id] = Text[start:end]
        else:
            self.n_subjects = EEG.shape[0] // trials_per_subject
            self.subject_map = None
            self.EEG = torch.from_numpy(np.array(np.split(EEG, self.n_subjects)))  # (S, T, C, L)
            self.Text = np.array(np.split(Text, self.n_subjects))                  # (S, T)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return self.total_trials

    def get_subject_by_idx(self, idx):
        if self.subject_map:
            for subject_id, (start, end) in self.subject_map.items():
                if idx >= start and idx < end:
                    return subject_id, idx - start
        return None
    
    def _smart_sampling(self, query_label, label_dict, target_num):
        """
        Balanced sampling strategy:
        1. Ensure query category has at least 1 sample
        2. Ensure each category has at least 1 sample  
        3. Remaining slots are randomly assigned, avoiding category quantity deviation
        """
        support_ids = []
        
        available_labels = [label for label in label_dict.keys() if label_dict[label]]
        if not available_labels:
            return support_ids
        
        label_candidates = {}
        for label in available_labels:
            label_candidates[label] = label_dict[label].copy()
            random.shuffle(label_candidates[label])
        
        remaining_slots = target_num
        
        if query_label in available_labels and remaining_slots > 0:
            if label_candidates[query_label]:
                support_ids.append(label_candidates[query_label].pop(0))
                remaining_slots -= 1
        
        other_labels = [label for label in available_labels if label != query_label]
        for label in other_labels:
            if remaining_slots <= 0:
                break
            if label_candidates[label]:
                support_ids.append(label_candidates[label].pop(0))
                remaining_slots -= 1
        
        while remaining_slots > 0:
            available_for_sampling = []
            for label in available_labels:
                if label_candidates[label]:
                    available_for_sampling.append(label)
                else:
                    label_candidates[label] = label_dict[label].copy()
                    random.shuffle(label_candidates[label])
                    if label_candidates[label]:
                        available_for_sampling.append(label)
            
            if not available_for_sampling:
                break
                
            chosen_label = random.choice(available_for_sampling)
            support_ids.append(label_candidates[chosen_label].pop(0))
            remaining_slots -= 1

        random.shuffle(support_ids)
        return support_ids
    
    def __getitem__(self, idx):
        if self.subject_map:
            sid, qid = self.get_subject_by_idx(idx)
        else:
            sid = idx // self.trials_per_subject
            qid = idx % self.trials_per_subject
        
        eeg_query = self.EEG[sid][qid]
        txt_query = self.Text[sid][qid]
        
        subject_texts = self.Text[sid]
        support_ids = []
        label_dict = {}
        
        for label in self.unique_labels:
            label_indices = [i for i in range(len(subject_texts)) if subject_texts[i] == label and i != qid]
            if label_indices:
                label_dict[label] = label_indices
        
        if self.fixed_support_num is not None:
            support_ids = self._smart_sampling(txt_query, label_dict, self.fixed_support_num)
        else:
            # Original k_per_class strategy
            for label in self.unique_labels:
                if label in label_dict and label_dict[label]:
                    k_for_label = min(self.k_per_class, len(label_dict[label]))
                    support_ids.extend(random.sample(label_dict[label], k_for_label))
            random.shuffle(support_ids)
        eeg_support = self.EEG[sid][support_ids] if support_ids else torch.zeros((0, *eeg_query.shape))
        txt_support = subject_texts[support_ids] if support_ids else []
        
        if len(support_ids) > 0:
            sep0_len = len(self.tokenizer.encode('<'))
            sep1_len = len(self.tokenizer.encode('>'))
            sot_len = len(list(self.tokenizer.sot_sequence_including_notimestamps))
            
            tok_lens = [len(self.tokenizer.encode(txt)) + sep0_len + sep1_len for txt in txt_support]
            cum_lens = np.cumsum(tok_lens)
            
            if cum_lens[-1] + sot_len > self.max_len:
                idx = np.searchsorted(cum_lens, self.max_len - sot_len)
                idx = idx // len(self.unique_labels) * len(self.unique_labels)
                if idx == 0:
                    raise ValueError("Max Length is too short for 1 sample each class.")
                support_ids = support_ids[:idx]
                eeg_support = self.EEG[sid][support_ids]
                txt_support = subject_texts[support_ids]
        
        tok_support = [self.tokenizer.encode(txt) for txt in txt_support]
        
        return {
            "EEG": eeg_query,
            "support_EEG": eeg_support,
            "support_text": txt_support,
            "support_tokens": tok_support,
            "target_text": txt_query
        }

class WhisperICLCollator_Test:
    def __init__(self, tokenizer, sot_first: bool = True, support: bool = True):
        self.tokenizer = tokenizer
        self.tok_start = list(self.tokenizer.sot_sequence_including_notimestamps)
        self.tok_end = [self.tokenizer.eot]
        self.support_sep0 = self.tokenizer.encode('<')
        self.support_sep1 = self.tokenizer.encode('>')
        self.sot_first = sot_first
        self.support = support

    @staticmethod
    def _pad(seq_list, pad_val, padding_side='left'):
        max_len = max(len(s) for s in seq_list)
        if padding_side == 'left':
            # Left padding: reverse each sequence, pad, then reverse back
            reversed_seqs = [seq.flip(0) for seq in seq_list]
            padded = pad_sequence(reversed_seqs, batch_first=True, padding_value=pad_val)[:, :max_len]
            return padded.flip(1)  # flip sequence dimension
        else:
            # Right padding (original logic)
            return pad_sequence(seq_list, batch_first=True, padding_value=pad_val)[:, :max_len]

    def __call__(self, features):
        batch_eeg, batch_inp, batch_text = [], [], []

        for feat in features:
            supp_eeg = feat["support_EEG"]
            query_eeg = feat["EEG"].unsqueeze(0)
            eeg_stack = torch.cat([supp_eeg, query_eeg], dim=0)
            batch_eeg.append(eeg_stack)

            if self.support:
                supp_input_tokens = []
                for supp_toks in feat["support_tokens"]:
                    seg = self.support_sep0 + supp_toks + self.support_sep1
                    supp_input_tokens += seg

                if self.sot_first:
                    input_tokens = self.tok_start + supp_input_tokens
                else:
                    input_tokens = supp_input_tokens + self.tok_start
                
                target_text = f"<{feat['target_text']}>"
            else:
                input_tokens = self.tok_start

                expected_sequence = []
                for support_text in feat["support_text"]:
                    expected_sequence.append(f"<{support_text}>")
                expected_sequence.append(f"<{feat['target_text']}>")
                target_text = "".join(expected_sequence)
            input_tokens = torch.tensor(input_tokens, dtype=torch.long)

            batch_inp.append(input_tokens)
            batch_text.append(target_text)

        # if len(batch_inp) > 0:
        #     max_len = max(len(inp) for inp in batch_inp)
        #     min_len = min(len(inp) for inp in batch_inp)
        #     assert max_len == min_len, "Input length is not consistent, max_len: {max_len}, min_len: {min_len}"
        
        # dec_in = self._pad(batch_inp, self.tok_start[0], padding_side='left')
        dec_in = torch.stack(batch_inp)
        eeg = torch.stack(batch_eeg)

        return {"eeg": eeg, "input_tokens": dec_in, "target_text": batch_text}

###############################################################
#                         Lazy Version                        #
###############################################################

class ICLWithSupportDataset_Test_Lazy(Dataset):
    """Lazy ICL dataset for testing phase."""
    def __init__(
        self,
        udas: _List,
        tokenizer,
        classes: _List[str],
        subject_map: dict,
        k_per_class: int = 3,
        max_len: int = 256,
        fixed_support_num: int = None,
        support: bool = True,
        time_len: int = 30,
    ):
        super().__init__()
        self.udas = udas
        self.tokenizer = tokenizer
        self.classes = classes
        self.k_per_class = k_per_class
        self.max_len = max_len
        self.fixed_support_num = fixed_support_num
        self.support = support
        self.subject_map = subject_map
        self.time_len = time_len

        self.trial_indices: _List[_Tuple[int, int]] = []

        count = 0
        self.uda2class_idx = [count]
        for uda in udas:
            count += len(uda.classes)
            self.uda2class_idx.append(count)
            
        for sub_id, (st, ed) in subject_map.items():
            for lt in range(ed - st):
                self.trial_indices.append((sub_id, lt))

        # ----- Build fixed support pool per subject and remove them from query indices -----
        self.fixed_support_pool_size = self.fixed_support_num if self.fixed_support_num is not None else self.k_per_class * len(self.classes)
        self.fixed_support_local_indices = {}
        for sub_id, (st, ed) in self.subject_map.items():
            n_trials = ed - st
            k = min(self.fixed_support_pool_size, n_trials)
            if k == 0:
                self.fixed_support_local_indices[sub_id] = []
                continue

            label_to_indices = {}
            for lt in range(n_trials):
                y = self._load_label(sub_id, lt)
                label = self._get_class_label(y)
                label_to_indices.setdefault(label, []).append(lt)

            for idx_list in label_to_indices.values():
                _rnd.shuffle(idx_list)

            support_local = []
            label_keys = list(label_to_indices.keys())
            while len(support_local) < k and label_to_indices:
                _rnd.shuffle(label_keys)
                new_keys = []
                for label in label_keys:
                    idx_list = label_to_indices.get(label, [])
                    if not idx_list:
                        continue
                    support_local.append(idx_list.pop(0))
                    if len(support_local) >= k:
                        break
                    if idx_list:
                        new_keys.append(label)
                label_keys = new_keys if new_keys else [
                    key for key, vals in label_to_indices.items() if vals
                ]
                if not label_keys:
                    break

            if len(support_local) < k:
                remaining = [
                    lt
                    for lt in range(n_trials)
                    if lt not in support_local
                ]
                _rnd.shuffle(remaining)
                support_local.extend(remaining[: k - len(support_local)])

            self.fixed_support_local_indices[sub_id] = sorted(support_local)

        support_sets = {sid: set(lst) for sid, lst in self.fixed_support_local_indices.items()}
        self.trial_indices = [
            (sid, ltid)
            for (sid, ltid) in self.trial_indices
            if ltid not in support_sets.get(sid, set())
        ]

        # Preload fixed support data for each subject for efficient reuse during inference
        self.fixed_support_cache = {}
        for sub_id, local_list in self.fixed_support_local_indices.items():
            supp_eeg, supp_toks, supp_text = [], [], []
            for ltid in local_list:
                x_s, y_s = self._load_trial(sub_id, ltid)
                t = self._get_class_label(y_s)
                supp_eeg.append(x_s)
                supp_toks.append(self.tokenizer.encode(t))
                supp_text.append(t)
            if supp_eeg:
                supp_eeg = torch.stack(supp_eeg)
            else:
                supp_eeg = None  # lazily handle empty case in __getitem__ based on query shape
            self.fixed_support_cache[sub_id] = {
                "eeg": supp_eeg,
                "tokens": supp_toks,
                "text": supp_text,
            }

    def __len__(self):
        return len(self.trial_indices)

    def _smart_sampling(self, query_label, label_dict, target_num):
        support_ids = []
        available_labels = [lab for lab, tids in label_dict.items() if tids]
        if not available_labels or target_num <= 0:
            return support_ids

        if query_label in label_dict and label_dict[query_label]:
            support_ids.append(_rnd.choice(label_dict[query_label]))

        for lab in available_labels:
            if lab == query_label:
                continue
            if len(support_ids) >= target_num:
                break
            support_ids.append(_rnd.choice(label_dict[lab]))

        if len(support_ids) < target_num:
            pool = [tid for tids in label_dict.values() for tid in tids if tid not in support_ids]
            _rnd.shuffle(pool)
            support_ids.extend(pool[: target_num - len(support_ids)])

        if len(support_ids) > target_num:
            support_ids = _rnd.sample(support_ids, target_num)

        _rnd.shuffle(support_ids)
        return support_ids

    def _load_trial(self, global_sub_idx: int, local_trial_idx: int):
        cum = 0
        uda_idx = None
        for u_id, uda in enumerate(self.udas):
            if global_sub_idx < cum + uda.n_subjects:
                uda_idx = u_id
                local_sub_idx = global_sub_idx - cum
                break
            cum += uda.n_subjects
        assert uda_idx is not None

        uda = self.udas[uda_idx]
        x_np, y = uda.get_trial(local_sub_idx, local_trial_idx)
        y = self.uda2class_idx[uda_idx] + y
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*",
            )
            x_tensor = torch.from_numpy(pad_trial(x_np[None, ...], uda.sfreq * self.time_len)[0])
        return x_tensor, y

    def _get_class_label(self, y):
        return self.classes[y]
    
    def _load_label(self, global_sub_idx: int, local_trial_idx: int):
        """Load label without reading EEG data."""
        cum = 0
        uda_idx = None
        for u_id, uda in enumerate(self.udas):
            if global_sub_idx < cum + uda.n_subjects:
                uda_idx = u_id
                local_sub_idx = global_sub_idx - cum
                break
            cum += uda.n_subjects
        assert uda_idx is not None

        uda = self.udas[uda_idx]
        if hasattr(uda, "get_label"):
            y = uda.get_label(local_sub_idx, local_trial_idx)
        else:
            _, y = uda.get_trial(local_sub_idx, local_trial_idx)
        y = self.uda2class_idx[uda_idx] + y
        return y

    def __getitem__(self, idx):
        gsid, ltid = self.trial_indices[idx]
        eeg_q, y_q = self._load_trial(gsid, ltid)
        txt_q = self._get_class_label(y_q)

        # ---------- fixed support from precomputed pool ----------
        cache = self.fixed_support_cache.get(gsid, None)
        if cache is not None:
            cached_text = cache["text"]
            cached_toks = cache["tokens"]
            cached_eeg = cache["eeg"]

            if cached_text:
                target_num = self.fixed_support_num if self.fixed_support_num is not None else len(cached_text)
                target_num = min(target_num, len(cached_text))
                label_dict = {}
                for idx_c, label in enumerate(cached_text):
                    label_dict.setdefault(label, []).append(idx_c)
                selected_idx = self._smart_sampling(txt_q, label_dict, target_num)
                if not selected_idx:
                    selected_idx = list(range(target_num))
                selected_idx = selected_idx[:target_num]
                _rnd.shuffle(selected_idx)
                supp_text = [cached_text[i] for i in selected_idx]
                supp_toks = [cached_toks[i] for i in selected_idx]
                if cached_eeg is not None:
                    supp_eeg = cached_eeg[selected_idx]
                else:
                    supp_eeg = torch.zeros((0, *eeg_q.shape))
            else:
                supp_text, supp_toks = [], []
                supp_eeg = torch.zeros((0, *eeg_q.shape))
        else:
            supp_text, supp_toks = [], []
            supp_eeg = torch.zeros((0, *eeg_q.shape))

        return {
            "EEG": eeg_q,
            "support_EEG": supp_eeg,
            "support_text": supp_text,
            "support_tokens": supp_toks,
            "target_text": txt_q,
        }
