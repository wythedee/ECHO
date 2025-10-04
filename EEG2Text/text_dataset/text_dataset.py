import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .utils import pad_trial
import random as _rnd
from typing import List as _List, Tuple as _Tuple
import warnings

class ICLWithSupportDataset_TestSub(Dataset):
    def __init__(self, EEG_subject, Text_subject, tokenizer, k_support: int = 5):
        super().__init__()
        assert EEG_subject.shape[0] == len(Text_subject)
        self.k = k_support
        self.EEG = torch.from_numpy(EEG_subject)   # (N, C, T)
        self.Text = Text_subject
        self.tokenizer = tokenizer
        self.num_trials = len(Text_subject)

    def __len__(self):
        return self.num_trials

    def __getitem__(self, qid):
        # Get query EEG and text
        eeg_query = self.EEG[qid]                  # (C, T)
        txt_query = self.Text[qid]
        tok_query = self.tokenizer.encode(txt_query)

        # Get k support samples (exclude query index)
        candidates = list(range(self.num_trials))
        candidates.remove(qid)
        support_ids = random.sample(candidates, self.k)

        eeg_support = self.EEG[support_ids]        # (k, C, T)
        txt_support = [self.Text[i] for i in support_ids]
        tok_support = [self.tokenizer.encode(txt) for txt in txt_support]

        return {
            "EEG": eeg_query,               # (C, T)
            "tokens": tok_query,            # list[int]
            "support_EEG": eeg_support,     # (k, C, T)
            "support_tokens": tok_support   # list[k x (Li,)]
        }

class ICLWithSupportDataset(Dataset):
    def __init__(self, EEG, Text, tokenizer, trials_per_subject: int = 100, k_per_class: int = 2, max_len: int = 256, subject_map: dict = None, fixed_support_num: int = None):
        super().__init__()
        assert EEG.shape[0] == len(Text)
        self.k_per_class = k_per_class
        self.max_len = max_len
        self.trials_per_subject = trials_per_subject
        self.fixed_support_num = fixed_support_num  # fixed support number
        
        self.total_trials = EEG.shape[0]
        self.unique_labels = list(set(Text))
        self.tokenizer = tokenizer
        
        if subject_map:
            # use subject_map mode
            self.n_subjects = len(subject_map)
            self.EEG = [None] * max(subject_map.keys()) + [None]  # ensure index large enough
            self.Text = [None] * max(subject_map.keys()) + [None]
            self.subject_map = subject_map
            
            for subject_id, (start, end) in subject_map.items():
                self.EEG[subject_id] = torch.from_numpy(EEG[start:end])
                self.Text[subject_id] = Text[start:end]
        else:
            # use trials_per_subject mode
            self.n_subjects = EEG.shape[0] // trials_per_subject
            self.subject_map = None
            self.EEG = torch.from_numpy(np.array(np.split(EEG, self.n_subjects)))  # (S, T, C, L)
            self.Text = np.array(np.split(Text, self.n_subjects))                  # (S, T)
            

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
        balanced sampling strategy:
        1. ensure query category has at least 1 sample
        2. ensure each category has at least 1 sample  
        3. remaining slots are randomly assigned, avoiding category quantity deviation
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
        tok_query = self.tokenizer.encode(txt_query)
        
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
        txt_support = subject_texts[support_ids] if support_ids else []
        eeg_support = self.EEG[sid][support_ids] if support_ids else torch.zeros((0, *eeg_query.shape))
        
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

        # label list for each support sample (use original text label for simplicity)
        support_labels = list(txt_support)  # same length as tok_support

        return {
            "EEG": eeg_query,             # (C, T)
            "tokens": tok_query,          # (L,)
            "support_EEG": eeg_support,   # (k, C, T)
            "support_tokens": tok_support,  # list of k (L_i,)
            "support_labels": support_labels  # list of k labels (str)
        }
    
class WhisperICLCollator:
    def __init__(self, tokenizer, k_is_random: bool = False, sot_first: bool = True, supp_loss: bool = True):
        self.tokenizer = tokenizer
        self.tok_start = list(self.tokenizer.sot_sequence_including_notimestamps)
        self.tok_end = [self.tokenizer.eot]
        self.support_sep0 = self.tokenizer.encode('<')
        self.support_sep1 = self.tokenizer.encode('>')
        self.k_is_random = k_is_random
        self.sot_first = sot_first
        self.supp_loss = supp_loss
        
    @staticmethod
    def _pad(seq_list, pad_val, padding_side='left'):
        max_len = max(len(s) for s in seq_list)
        if padding_side == 'left':
            # left padding: reverse each sequence, pad, then reverse back
            reversed_seqs = [seq.flip(0) for seq in seq_list]
            padded = pad_sequence(reversed_seqs, batch_first=True, padding_value=pad_val)[:, :max_len]
            return padded.flip(1)  # flip sequence dimension
        else:
            # right padding (original logic)
            return pad_sequence(seq_list, batch_first=True, padding_value=pad_val)[:, :max_len]

    def __call__(self, features):
        batch_eeg, batch_inp, batch_lbl = [], [], []
        max_support = len(features[0]["support_EEG"])

        # Decide support count: 0 or >= number of unique classes in support.
        if self.k_is_random and max_support > 0:
            max_uniq_labels = max(len(set(feat.get("support_labels", None))) for feat in features)
            possible_counts = [0] + list(range(max_uniq_labels, max_support + 1))
            actual_k = random.choice(possible_counts)
        else:
            actual_k = max_support

        for feat in features:
            supp_eeg_full = feat["support_EEG"]              # (k, C, T)
            supp_tokens_full = feat["support_tokens"]
            supp_labels_full = feat.get("support_labels", None)
            if supp_labels_full is None:
                supp_labels_full = [tuple(t) for t in supp_tokens_full]

            # Random pick and keep alignment
            if actual_k == 0:
                supp_eeg = supp_eeg_full[:0]  # empty tensor
                support_tokens_to_use = []
            elif self.k_is_random:
                # ensure selected indices cover all classes
                label_to_indices = {}
                for i, lab in enumerate(supp_labels_full):
                    label_to_indices.setdefault(lab, []).append(i)

                labels_set = list(label_to_indices.keys())
                # if actual_k < len(labels_set) (possible when uniq_cnt < max_support and random picks smaller?)
                if actual_k < len(labels_set):
                    actual_k = len(labels_set)

                selected_idx = []
                for lab in labels_set:
                    selected_idx.append(random.choice(label_to_indices[lab]))

                remaining_needed = actual_k - len(selected_idx)
                if remaining_needed > 0:
                    remaining_pool = [i for i in range(len(supp_labels_full)) if i not in selected_idx]
                    extra = random.sample(remaining_pool, min(remaining_needed, len(remaining_pool)))
                    selected_idx.extend(extra)

                selected_idx = sorted(selected_idx)

                supp_eeg = supp_eeg_full[selected_idx]
                support_tokens_to_use = [supp_tokens_full[i] for i in selected_idx]
            else:
                supp_eeg = supp_eeg_full
                support_tokens_to_use = supp_tokens_full

            query_eeg = feat["EEG"].unsqueeze(0)        # (1, C, T)
            eeg_stack = torch.cat([supp_eeg, query_eeg], dim=0)  # (k+1, C, T)
            batch_eeg.append(eeg_stack)

            supp_input_tokens, supp_target_tokens = [], []
            for toks in support_tokens_to_use:
                seg = self.support_sep0 + toks + self.support_sep1
                supp_input_tokens  += seg
                supp_target_tokens += [-100] * len(seg)


            query_seg = self.support_sep0 + feat["tokens"] + self.support_sep1
            
            if self.sot_first:
                input_tokens = self.tok_start + supp_input_tokens + query_seg
                if self.supp_loss:
                    target_tokens = self.tok_start[1:] + supp_input_tokens + query_seg + self.tok_end
                else:
                    target_tokens = self.tok_start[1:] + supp_target_tokens + [-100] * len(self.support_sep0) + feat["tokens"] + [-100] * len(self.support_sep1) + self.tok_end
            else:
                input_tokens = supp_input_tokens + self.tok_start + query_seg
                if self.supp_loss:
                    target_tokens = supp_input_tokens[1:] + self.tok_start[1:] + query_seg + self.tok_end
                else:
                    target_tokens = (supp_target_tokens[1:] + self.tok_start if len(supp_target_tokens) > 1 else self.tok_start[1:]) + [-100] * len(self.support_sep0) + feat["tokens"] + [-100] * len(self.support_sep1) + self.tok_end

            input_tokens  = torch.tensor(input_tokens, dtype=torch.long)
            target_tokens = torch.tensor(target_tokens, dtype=torch.long)

            batch_inp.append(input_tokens)
            batch_lbl.append(target_tokens)

        dec_in = self._pad(batch_inp, pad_val=self.tok_end[0], padding_side='right')
        labels = self._pad(batch_lbl, pad_val=-100, padding_side='right')
        eeg = torch.stack(batch_eeg)  # (B, k+1, C, T)

        return {"eeg": eeg, "input_tokens": dec_in, "target_tokens": labels}

###############################################################
#                         Lazy Version                        #
###############################################################

class ICLWithSupportDataset_Lazy(Dataset):
    """Same interface as `ICLWithSupportDataset`, but loads trials lazily from disk."""
    def __init__(
        self,
        udas: _List,
        tokenizer,
        classes: _List[str],
        subject_map: dict,
        k_per_class: int = 2,
        max_len: int = 256,
        fixed_support_num: int = None,
        time_len: int = 30,
    ):
        super().__init__()
        self.udas = udas
        self.tokenizer = tokenizer
        self.classes = classes
        self.k_per_class = k_per_class
        self.max_len = max_len
        self.time_len = time_len
        self.fixed_support_num = fixed_support_num
        self.subject_map = subject_map  # global_sub -> (start, end)

        # Pre-build a mapping: global_trial_idx -> (global_subject_id, local_trial_id)
        self.trial_indices: _List[_Tuple[int, int]] = []

        count = 0
        self.uda2class_idx = [count]
        for uda in udas:
            count += len(uda.classes)
            self.uda2class_idx.append(count)
            
        for sub_id, (st, ed) in subject_map.items():
            for lt in range(ed - st):
                self.trial_indices.append((sub_id, lt))

    def __len__(self):
        return len(self.trial_indices)

    def _smart_sampling(self, query_label, label_dict, target_num):
        support_ids = []
        available_labels = [lab for lab, tids in label_dict.items() if tids]
        if not available_labels:
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
        """Directly fetch a single trial from the underlying H5 dataset (no caching)."""
        cum = 0
        uda_idx = None
        for u_id, uda in enumerate(self.udas):
            if global_sub_idx < cum + uda.n_subjects:
                uda_idx = u_id
                local_sub_idx = global_sub_idx - cum
                break
            cum += uda.n_subjects
        assert uda_idx is not None, "decode subject failed"

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
    
    def _load_label(self, global_sub_idx: int, local_trial_idx: int):
        """Load label only (no EEG reading)."""
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
        txt_q = self.classes[y_q]
        tok_q = self.tokenizer.encode(txt_q)

        # Construct support pool
        s_start, s_end = self.subject_map[gsid]
        all_trials = list(range(s_start, s_end))
        abs_q = s_start + ltid
        all_trials.remove(abs_q)

        label_dict = {}
        for abs_tid in all_trials:
            lt = abs_tid - s_start
            y = self._load_label(gsid, lt)
            label_dict.setdefault(y, []).append(abs_tid)

        support_ids = []
        if self.fixed_support_num is not None:
            support_ids = self._smart_sampling(y_q, label_dict, self.fixed_support_num)
        else:
            for lab, tids in label_dict.items():
                k = min(len(tids), self.k_per_class)
                support_ids.extend(_rnd.sample(tids, k))
            _rnd.shuffle(support_ids)

        supp_eeg, supp_toks, supp_labels = [], [], []
        for abs_tid in support_ids:
            lt = abs_tid - s_start
            x_s, y_s = self._load_trial(gsid, lt)
            supp_eeg.append(x_s)
            supp_toks.append(self.tokenizer.encode(self.classes[y_s]))
            supp_labels.append(self.classes[y_s])

        if supp_eeg:
            supp_eeg = torch.stack(supp_eeg)
        else:
            supp_eeg = torch.zeros((0, *eeg_q.shape))

        return {
            "EEG": eeg_q,
            "tokens": tok_q,
            "support_EEG": supp_eeg,
            "support_tokens": supp_toks,
            "support_labels": supp_labels,
        }