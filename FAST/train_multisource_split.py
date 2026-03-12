import os
import argparse
import h5py
import numpy as np
import torch
import json
import shutil

torch.set_num_threads(8)
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold  # KFold 已移除
from transformers import PretrainedConfig
import lightning as pl
import logging
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, f1_score, confusion_matrix
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)

from FAST_v2 import FAST as Tower
from EEG_Montage.template_ch74 import Electrodes, Zones
from EEG_Dataset.share import template_ch_names, template_zones
# from EEG_Dataset.MI_01_SSVEP_KoreaU import CH_NAMES, ZONES
from train_utils import seed_all, cosine_scheduler, all_exist, yellow, green, freeze
from train_utils import convert_epoch_acc_to_npy
from EEG_Dataset import DATASET_BASIC_TASKS

DATASET_ROOT = os.environ.get('ECHO_DATASET_ROOT', '/path/to/EEG_Standardized_Group')
PreTrainStage1 = 5
PreTrainStage2 = 45
PreTrainEpochs = PreTrainStage1 + PreTrainStage2
FineTuneEpochs = 20
CheckValEveryN = 1

# 尝试导入固定 subject 划分配置；失败时在运行时回退到默认 7/1.5/1.5
try:
    import dataset_split_config as split_cfg  # 可能包含用户自定义的 subject 划分
except Exception as _e:
    split_cfg = None

def is_rank0():
    """Return True if current process is global rank 0.
    Fallback to True when rank env is absent (single process)."""
    rank = os.environ.get("RANK") or os.environ.get("GLOBAL_RANK")
    try:
        return int(rank) == 0
    except Exception:
        return True

def find_dataset(h5_name):
    for _, task in DATASET_BASIC_TASKS.items():
        if task.h5_name == h5_name:
            return task
    raise ValueError(f"Dataset {h5_name} not found")

# ===== Lazy trial-level dataset =====
class TrialDataset(Dataset):
    """Read one trial on demand from underlying h5 files."""
    def __init__(self, multisrc, trial_index_list):
        self.multisrc = multisrc            # MultiSourceDataset
        self.trials = trial_index_list      # list of (uda_idx, local_sub_idx, trial_idx)
        self.handles = {}

    def _get_h5(self, uda_idx):
        if uda_idx not in self.handles:
            uda = self.multisrc.udas[uda_idx]
            self.handles[uda_idx] = h5py.File(uda.file_path, 'r')
        return self.handles[uda_idx]

    def __getitem__(self, idx):
        uda_idx, local_sub_idx, trial_idx = self.trials[idx]
        h5 = self._get_h5(uda_idx)
        uda = self.multisrc.udas[uda_idx]
        sub_key = uda.sub_keys[local_sub_idx]
        x = h5[sub_key]['X'][trial_idx]
        y = h5[sub_key]['Y'][trial_idx]
        # pad / clip
        x = self.multisrc._pad_trial(x[np.newaxis, ...])[0]
        # label remap
        y = self.multisrc.remap_labels_single(np.array([y]), uda_idx)[0]
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.trials)

class EEG_Encoder_Module(pl.LightningModule):
    def __init__(self, config, lr_mode, niter_per_ep, dataset_names=None, seed: int = 42):
        super().__init__()
        seed_all(seed)
        self.config = config
        self.lr_mode = lr_mode
        self.n_val_loaders = len(dataset_names) if dataset_names is not None else 1
        self.train_acc = []
        self.train_loss = []
        self.epoch_acc = []
        self.epoch_loss = []
        self.dataset_names = dataset_names or []
        self.loss = nn.CrossEntropyLoss()
        self.model = Tower(config)
        # ===== Debug: register hooks to locate NaN/Inf =====
        self._register_nan_hooks()
        # ===== End debug =====
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes = config.n_classes)
        # ===== Test step outputs for metrics calculation =====
        self.test_step_outputs = []
        self.forward_mode = 'default'
        if lr_mode == 'pre':
            self.lr_list = np.concatenate([
                cosine_scheduler(1, 0.01, PreTrainStage1, niter_per_ep, warmup_epochs=0),
                cosine_scheduler(1, 0.01, PreTrainStage2, niter_per_ep, warmup_epochs=10)
            ])
        elif lr_mode == 'tune':
            self.lr_list = cosine_scheduler(1, 0.1, FineTuneEpochs, niter_per_ep, warmup_epochs=10)

    def lr_lambda(self, step):
        assert self.global_step == step
        idx = self.global_step - 1
        if idx >= len(self.lr_list):
            idx = -1  # use last value if exceeded
        lr_value = self.lr_list[idx]  # print(f'{step} LR: {lr_value:.6f}')
        return lr_value

    def configure_optimizers(self):
        if self.lr_mode == 'pre':
            self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
            return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]
        elif self.lr_mode == 'tune':
            self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr * 2)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
            return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def _balanced_accuracy(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = int(self.config.n_classes)
        recalls = []
        for c in range(num_classes):
            mask = (target == c)
            denom = mask.sum()
            if denom > 0:
                correct = (preds[mask] == c).sum()
                recalls.append(correct.float() / denom.float())
        if len(recalls) == 0:
            return torch.tensor(0.0, device=target.device, dtype=torch.float32)
        return torch.stack(recalls).mean()

    def on_train_epoch_start(self):
        if self.lr_mode == 'pre':
            if self.current_epoch < PreTrainStage1:
                self.forward_mode = 'train_head'
            else:
                self.forward_mode = 'train_transformer'
        elif self.lr_mode == 'tune':
            self.forward_mode = 'default'

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.forward(x, self.forward_mode)
        preds = logits.argmax(dim=1)
        bacc = self._balanced_accuracy(preds, y)
        self.log("train/acc", bacc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        loss = self.loss(logits, y)
        self.log("train/loss", loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        # ===== Debug: check raw input =====
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[BAD X] batch={batch_idx} min/max: {x.min().item()} {x.max().item()}", flush=True)
            raise SystemExit
        # ===== End debug =====
        logits = self.model.forward(x, self.forward_mode)

        # 1️⃣ 先看 logits 本身有没有 NaN / Inf，幅值是否异常
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[NAN LOGITS] batch={batch_idx}", flush=True)
            print("  logits min/max:", logits.min().item(), logits.max().item())
            # 逐层定位：打印每个参数有没有 NaN / Inf
            for n, p in self.model.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"  ⚠️ param {n} has NaN/Inf")
            raise SystemExit  # 直接停掉方便定位

        # 2️⃣ 再确认标签是否越界
        if y.min() < 0 or y.max() >= self.config.n_classes:
            print(f"[BAD LABEL] batch={batch_idx}", y.min().item(), y.max().item(), flush=True)
            raise SystemExit

        # 3️⃣ 计算 loss
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)
        acc = self._balanced_accuracy(preds, y)

        # loss 为 NaN 时打印更多信息
        if torch.isnan(loss):
            print(f"[NAN LOSS] batch={batch_idx}", flush=True)
            print("  logits min/max:", logits.min().item(), logits.max().item())
            print("  y unique:", torch.unique(y))
            raise SystemExit

        # 正常训练流程
        self.log("train/loss", loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero and self.current_epoch % 5 == 0:
            acc = self.trainer.callback_metrics["train/acc"]
            loss = self.trainer.callback_metrics["train/loss"]
            acc = acc.item() if torch.is_tensor(acc) else float(acc)
            loss = loss.item() if torch.is_tensor(loss) else float(loss)
            self.train_acc.append(acc)
            self.train_loss.append(loss)
            # 验证指标在 on_validation_epoch_end 再打印；此处仅打印训练
            print(f"Epoch {self.current_epoch} Train acc: {acc:.4f} Train loss: {loss:.4f}")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.model.forward(x)
        preds = logits.argmax(dim=1)
        acc = self._balanced_accuracy(preds, y)
        loss = self.loss(logits, y)
        if dataloader_idx + 1 > self.n_val_loaders:
            self.n_val_loaders = dataloader_idx + 1

        # log individual dataloader acc
        self.log(f"acc", acc, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(f"loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Gather accuracy for each validation dataloader
        acc_list = []
        loss_list = []
        if len(self.dataset_names) > 1:
            for i in range(self.n_val_loaders):
                key = f"acc/dataloader_idx_{i}"
                if key in self.trainer.callback_metrics:
                    val = self.trainer.callback_metrics[key].item() if torch.is_tensor(self.trainer.callback_metrics[key]) else float(self.trainer.callback_metrics[key])
                    acc_list.append(val)
                key = f"loss/dataloader_idx_{i}"
                if key in self.trainer.callback_metrics:
                    val = self.trainer.callback_metrics[key].item() if torch.is_tensor(self.trainer.callback_metrics[key]) else float(self.trainer.callback_metrics[key])
                    loss_list.append(val)
            if acc_list:
                # print each dataset acc
                acc_text = " | ".join([f"{self.dataset_names[i] if i < len(self.dataset_names) else f'ds_{i}'}:{acc_list[i]:.3f}" for i in range(len(acc_list))])
                mean_acc = np.mean(acc_list)
                self.log("mean_acc", mean_acc, on_epoch=True, prog_bar=False, sync_dist=True, add_dataloader_idx=False)
                if self.trainer.is_global_zero and self.current_epoch % 5 == 0:
                    print(f"Epoch {self.current_epoch}, Acc: {acc_text}, Mean: {mean_acc:.3f}")
                self.epoch_acc.append(np.array(acc_list))
            if loss_list:
                loss_text = " | ".join([f"{self.dataset_names[i] if i < len(self.dataset_names) else f'ds_{i}'}:{loss_list[i]:.3f}" for i in range(len(loss_list))])
                if self.trainer.is_global_zero and self.current_epoch % 5 == 0:
                    print(f"Epoch {self.current_epoch} Loss: {loss_text}")
                self.epoch_loss.append(np.array(loss_list))
        else:
            # single dataloader case
            if "acc" in self.trainer.callback_metrics:
                val_acc = self.trainer.callback_metrics["acc"].item() if torch.is_tensor(self.trainer.callback_metrics["acc"]) else float(self.trainer.callback_metrics["acc"])
                val_loss = self.trainer.callback_metrics["loss"].item() if torch.is_tensor(self.trainer.callback_metrics["loss"]) else float(self.trainer.callback_metrics["loss"])
                if self.trainer.is_global_zero and self.current_epoch % 5 == 0:
                    print(f"Epoch {self.current_epoch}: val_acc: {val_acc:.3f}, val_loss: {val_loss:.3f}")
                self.epoch_acc.append(np.array([val_acc]))
                self.epoch_loss.append(np.array([val_loss]))
                self.log("mean_acc", val_acc, on_epoch=True, prog_bar=False, sync_dist=True, add_dataloader_idx=False)
                self.log("mean_loss", val_loss, on_epoch=True, prog_bar=False, sync_dist=True, add_dataloader_idx=False)

    # ===== debug hooks =====
    def _register_nan_hooks(self):
        """Attach forward hooks to detect NaN/Inf in model outputs and terminate immediately for inspection."""
        # Only let rank-0 print, avoid重复输出 in DDP
        if hasattr(self, 'global_rank') and getattr(self, 'global_rank', 0) != 0:
            return
        layer_names = {id(m): n for n, m in self.model.named_modules()}
        def _has_nan(t):
            return torch.isnan(t).any() or torch.isinf(t).any()
        def hook_fn(module, inp, out):
            bad = False
            if torch.is_tensor(out):
                bad = _has_nan(out)
            elif isinstance(out, (tuple, list)):
                bad = any(torch.is_tensor(t) and _has_nan(t) for t in out)
            if bad:
                w = getattr(module, 'weight', None)
                if w is not None:
                    print("  weight nan?", torch.isnan(w).any().item(), "inf?", torch.isinf(w).any().item(),
                          "min/max:", w.min().item(), w.max().item(), flush=True)
                x_in = inp[0] if isinstance(inp, (list, tuple)) else inp
                if torch.is_tensor(x_in):
                    print("  input min/max:", x_in.min().item(), x_in.max().item(), flush=True)
                    print("  input mean/std:", x_in.mean().item(), x_in.std().item(), flush=True)
                    print("  input has nan?", torch.isnan(x_in).any().item(),
                          "inf?", torch.isinf(x_in).any().item(), flush=True)
                print(f"\n[NaN DETECTED] after layer: {layer_names.get(id(module), module.__class__.__name__)}", flush=True)
                raise SystemExit
        # 只在常见算子上挂 hook，减少开销
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.MultiheadAttention, nn.GELU)):
                m.register_forward_hook(hook_fn)
    # ===== end debug hooks =====

    # ===== test loop =====
    def on_test_epoch_start(self):
        # Reset container every test run
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.model.forward(x)
        preds = logits.argmax(dim=1)
        acc = self._balanced_accuracy(preds, y)
        loss = self.loss(logits, y)
        if dataloader_idx + 1 > self.n_val_loaders:
            self.n_val_loaders = dataloader_idx + 1
        self.log("test_acc", acc, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True, prog_bar=False)
        # Collect predictions and targets for metrics calculation
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = y.cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        output = {
            "predictions": preds,
            "targets": targets,
            "probs": probs,
            "dataloader_idx": dataloader_idx
        }
        self.test_step_outputs.append(output)



    def on_test_epoch_end(self):
        acc_list = []
        loss_list = []
        # Calculate additional metrics from collected outputs
        self.test_metrics = self._calculate_test_metrics()

        if len(self.dataset_names) > 1:
            for i in range(self.n_val_loaders):
                key = f"test_acc/dataloader_idx_{i}"
                if key in self.trainer.callback_metrics:
                    val = self.trainer.callback_metrics[key]
                    val = val.item() if torch.is_tensor(val) else float(val)
                    acc_list.append(val)
                key = f"test_loss/dataloader_idx_{i}"
                if key in self.trainer.callback_metrics:
                    val = self.trainer.callback_metrics[key]
                    val = val.item() if torch.is_tensor(val) else float(val)
                    loss_list.append(val)
            if acc_list and self.trainer.is_global_zero:
                acc_text = " | ".join([f"{self.dataset_names[i] if i < len(self.dataset_names) else f'ds_{i}'}:{acc_list[i]:.3f}" for i in range(len(acc_list))])
                print(f"[TEST] Acc: {acc_text}")
                # Print additional metrics
                if hasattr(self, 'test_metrics') and self.test_metrics:
                    self._print_additional_metrics()
            if loss_list and self.trainer.is_global_zero:
                loss_text = " | ".join([f"{self.dataset_names[i] if i < len(self.dataset_names) else f'ds_{i}'}:{loss_list[i]:.3f}" for i in range(len(loss_list))])
                print(f"[TEST] Loss: {loss_text}")
        else:
            if "test_acc" in self.trainer.callback_metrics and self.trainer.is_global_zero:
                val_acc = self.trainer.callback_metrics["test_acc"]
                val_loss = self.trainer.callback_metrics.get("test_loss", 0.0)
                val_acc = val_acc.item() if torch.is_tensor(val_acc) else float(val_acc)
                val_loss = val_loss.item() if torch.is_tensor(val_loss) else float(val_loss)
                print(f"[TEST] acc: {val_acc:.3f}, loss: {val_loss:.3f}")
                # Print additional metrics
                if hasattr(self, 'test_metrics') and self.test_metrics:
                    self._print_additional_metrics()

        if self.trainer.is_global_zero:
            self._print_confusion_matrices()

    def _calculate_test_metrics(self):
        """Calculate ROC AUC, PR AUC, Kappa, and Weighted F1 metrics from test outputs"""
        if not self.test_step_outputs:
            return {}

        # Aggregate all outputs
        all_predictions = []
        all_targets = []
        all_probs = []

        # Group by dataloader for multi-dataset case
        dataloader_groups = {}
        for output in self.test_step_outputs:
            dl_idx = output.get("dataloader_idx", 0)
            if dl_idx not in dataloader_groups:
                dataloader_groups[dl_idx] = {"predictions": [], "targets": [], "probs": []}
            dataloader_groups[dl_idx]["predictions"].extend(output["predictions"])
            dataloader_groups[dl_idx]["targets"].extend(output["targets"])
            dataloader_groups[dl_idx]["probs"].append(output["probs"])

        metrics = {}
        for dl_idx, data in dataloader_groups.items():
            ds_name = self.dataset_names[dl_idx] if dl_idx < len(self.dataset_names) else f"dataset_{dl_idx}"

            y_true = np.array(data["targets"])
            y_pred = np.array(data["predictions"])
            y_scores = np.concatenate(data["probs"], axis=0) if data["probs"] else None

            if y_scores is None or len(y_true) == 0:
                continue

            num_classes = self.config.n_classes
            ds_metrics = {}

            try:
                # Balanced accuracy from confusion matrix (macro recall)
                try:
                    labels = list(range(int(num_classes)))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    row_sums = cm.sum(axis=1)
                    recalls = np.divide(np.diag(cm), row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums != 0)
                    valid = row_sums != 0
                    bacc = float(recalls[valid].mean()) if np.any(valid) else 0.0
                    ds_metrics['accuracy'] = bacc
                except Exception:
                    pass

                # Binary classification metrics
                if num_classes == 2:
                    # Use positive class (class 1) probabilities
                    y_score_pos = y_scores[:, 1]
                    try:
                        roc_auc = roc_auc_score(y_true, y_score_pos)
                        ds_metrics['roc_auc'] = float(roc_auc)
                    except Exception:
                        pass
                    try:
                        pr_auc = average_precision_score(y_true, y_score_pos)
                        ds_metrics['pr_auc'] = float(pr_auc)
                    except Exception:
                        pass

                # Multiclass metrics (for num_classes > 2)
                if num_classes > 2:
                    # Create one-hot encoding for multiclass ROC/PR AUC
                    y_true_onehot = np.zeros((len(y_true), num_classes), dtype=np.int32)
                    y_true_onehot[np.arange(len(y_true)), y_true] = 1

                    # Calculate macro-averaged ROC AUC and PR AUC
                    roc_list = []
                    pr_list = []
                    for c in range(num_classes):
                        # Check if class has both positive and negative samples
                        if np.sum(y_true_onehot[:, c]) > 0 and np.sum(1 - y_true_onehot[:, c]) > 0:
                            try:
                                roc_list.append(roc_auc_score(y_true_onehot[:, c], y_scores[:, c]))
                            except Exception:
                                pass
                            try:
                                pr_list.append(average_precision_score(y_true_onehot[:, c], y_scores[:, c]))
                            except Exception:
                                pass

                    if len(roc_list) > 0:
                        ds_metrics['roc_auc_macro'] = float(np.mean(roc_list))
                    if len(pr_list) > 0:
                        ds_metrics['pr_auc_macro'] = float(np.mean(pr_list))

                    # Cohen's Kappa and Weighted F1 for multiclass
                    try:
                        kappa = cohen_kappa_score(y_true, y_pred)
                        ds_metrics['kappa'] = float(kappa)
                    except Exception:
                        pass
                    try:
                        wf1 = f1_score(y_true, y_pred, average='weighted')
                        ds_metrics['f1_weighted'] = float(wf1)
                    except Exception:
                        pass

            except Exception as e:
                print(f"Warning: Error calculating metrics for {ds_name}: {e}")

            if ds_metrics:
                metrics[ds_name] = ds_metrics

        return metrics

    def _print_additional_metrics(self):
        """Print additional test metrics"""
        if not hasattr(self, 'test_metrics') or not self.test_metrics:
            return

        print("\n[TEST] Additional Metrics:")
        for ds_name, metrics in self.test_metrics.items():
            metric_strs = []
            for metric_name, value in metrics.items():
                metric_strs.append(f"{metric_name}:{value:.4f}")
            if metric_strs:
                print(f"  {ds_name}: {' | '.join(metric_strs)}")

    def _print_confusion_matrices(self):
        """Print confusion matrix per dataset using collected test outputs."""
        if not self.test_step_outputs:
            return
        # Group predictions and targets by dataloader index
        dataloader_groups = {}
        for output in self.test_step_outputs:
            dl_idx = output.get("dataloader_idx", 0)
            if dl_idx not in dataloader_groups:
                dataloader_groups[dl_idx] = {"predictions": [], "targets": []}
            dataloader_groups[dl_idx]["predictions"].extend(output["predictions"])
            dataloader_groups[dl_idx]["targets"].extend(output["targets"])

        print("\n[TEST] Confusion Matrix:")
        for dl_idx, data in sorted(dataloader_groups.items(), key=lambda x: x[0]):
            ds_name = self.dataset_names[dl_idx] if dl_idx < len(self.dataset_names) else f"dataset_{dl_idx}"
            y_true = np.array(data["targets"], dtype=np.int64)
            y_pred = np.array(data["predictions"], dtype=np.int64)
            num_classes = int(self.config.n_classes)
            labels = list(range(num_classes))
            try:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
            except Exception:
                # Fallback to numpy bincount if sklearn unavailable at runtime
                cm = np.zeros((num_classes, num_classes), dtype=np.int64)
                for t, p in zip(y_true, y_pred):
                    if 0 <= t < num_classes and 0 <= p < num_classes:
                        cm[t, p] += 1
            print(f"  {ds_name}:")
            print(cm)

            # Balanced Accuracy from confusion matrix
            recalls = []
            for c in range(num_classes):
                denom = cm[c, :].sum()
                if denom > 0:
                    recalls.append(cm[c, c] / float(denom))
            if len(recalls) > 0:
                bacc = float(np.mean(recalls))
                print(f"    Balanced Acc: {bacc:.4f}")


def output_acc_chart(epoch_acc, train_acc, train_loss, save_path, dataset_names=None):
    """Draw accuracy (validation & training) and training loss in two subplots."""
    epochs = range(1, len(epoch_acc) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # ---------- Accuracy subplot ---------- #
    if dataset_names and len(dataset_names) > 1:
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i in range(len(epoch_acc[0])):
            dataset_acc = [acc[i] for acc in epoch_acc]
            color = colors[i % len(colors)]
            label = dataset_names[i] if i < len(dataset_names) else f'Dataset_{i}'
            ax_acc.plot(epochs, dataset_acc, color=color, linewidth=2, label=label, marker='o', markersize=3)
        mean_acc = [np.mean(acc) for acc in epoch_acc]
        ax_acc.plot(epochs, mean_acc, color='black', linewidth=3, label='Mean', linestyle='--', marker='s', markersize=4)
        ax_acc.set_title('Validation Accuracy - Multi-Dataset', fontsize=16)
    else:
        mean_acc = [np.mean(acc) for acc in epoch_acc]
        ax_acc.plot(epochs, mean_acc, 'b-', linewidth=2, label='Validation Accuracy', marker='o', markersize=3)
        ax_acc.set_title('Validation Accuracy', fontsize=16)

    ax_acc.plot(epochs[:len(train_acc)], train_acc, 'r-', linewidth=2, label='Training Accuracy', marker='o', markersize=3)
    ax_acc.set_ylabel('Accuracy', fontsize=14)
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(True, linestyle='--', alpha=0.7)
    ax_acc.legend(fontsize=12)

    # ---------- Loss subplot ---------- #
    if train_loss is not None and len(train_loss) == len(epochs):
        ax_loss.plot(epochs, train_loss, 'g-', linewidth=2, label='Training Loss', marker='o', markersize=3)
        ax_loss.set_ylabel('Loss', fontsize=14)
        ax_loss.grid(True, linestyle='--', alpha=0.7)
        ax_loss.legend(fontsize=12)
    else:
        ax_loss.text(0.5, 0.5, 'Training loss unavailable', horizontalalignment='center', verticalalignment='center', transform=ax_loss.transAxes)
        ax_loss.set_axis_off()

    ax_loss.set_xlabel('Epoch', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class h5Dataset:
    """
    Only read meta information at initialization, actual data is read on demand, significantly reducing CPU memory usage.
    """
    def __init__(self, name, classes):
        self.name = name
        self.file_path = f"{DATASET_ROOT}/{name}.h5"
        # Keep file handle open to avoid repeated open/close
        self.h5file = h5py.File(self.file_path, 'r')
        self.sub_keys = list(self.h5file.keys())
        self.n_subjects = len(self.sub_keys)
        self.classes = classes
        self.sfreq = 250  # Unified sampling rate, hardcoded as in original logic

    def _load_subject(self, sub_idx):
        """Return X, Y for a single subject, only load current subject into memory"""
        sub_key = self.sub_keys[sub_idx]
        X = self.h5file[sub_key]['X'][()]  # (n_trials, n_channels, n_samples)
        Y = self.h5file[sub_key]['Y'][()]
        return X.astype(np.float32), Y

    # Interface for MultiSourceDataset, keep signature unchanged
    def get_sub(self, idx):
        return self._load_subject(idx)

    def __del__(self):
        try:
            self.h5file.close()
        except Exception:
            pass

class MultiSourceDataset:
    """
    Changed to lazy loading, does not load all data into memory at initialization.
    Interface remains the same, no need to modify external training code.
    """
    def __init__(self, udas, resample_rate, time_length, classes, class_to_label_map=None):
        self.udas = udas
        self.resample_rate = resample_rate
        self.time_length = time_length  # seconds
        self.classes = classes

        # ---------- Build / adopt global class mapping ---------- #
        if class_to_label_map is not None:
            # Start from existing mapping (e.g., from pretraining ckpt)
            self.class_to_label_map = class_to_label_map.copy()
        else:
            self.class_to_label_map = {}

        # Ensure all classes appearing in current datasets are covered
        for cls_list in classes:
            for cls_name in cls_list:
                if cls_name not in self.class_to_label_map:
                    self.class_to_label_map[cls_name] = len(self.class_to_label_map)
        self.n_classes = len(self.class_to_label_map)

        # ---------- Build subject <-> dataset mapping ---------- #
        self.subject_to_dataset = []  # [(uda_idx, local_sub_idx), ...]
        for uda_idx, uda in enumerate(udas):
            self.subject_to_dataset.extend([(uda_idx, s_idx) for s_idx in range(uda.n_subjects)])

        self.n_subjects = len(self.subject_to_dataset)

    # ------------------ Utility functions ------------------ #
    def _pad_trial(self, x):
        """Pad a single trial to a uniform length, avoid padding all at once"""
        target_len = self.time_length * self.resample_rate
        cur_len = x.shape[-1]
        if cur_len < target_len:
            pad_width = target_len - cur_len
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        elif cur_len > target_len:
            x = x[..., :target_len]
        return x

    def remap_labels(self, y, dataset_index):
        """Keep interface, not used by external code, can be left empty or simply implemented"""
        return self.remap_labels_single(y, dataset_index)

    def remap_labels_single(self, y, dataset_index):
        dataset_classes = self.classes[dataset_index]
        return np.array([self.class_to_label_map[dataset_classes[label]] for label in y])

    # ------------------ New: Generate trial indices for given subject list ------------------ #
    def build_trial_indices(self, subject_indices):
        idx_list = []
        for global_sub_idx in subject_indices:
            uda_idx, local_sub_idx = self.subject_to_dataset[global_sub_idx]
            uda = self.udas[uda_idx]
            sub_key = uda.sub_keys[local_sub_idx]
            n_trials = uda.h5file[sub_key]['X'].shape[0]
            idx_list.extend([(uda_idx, local_sub_idx, t) for t in range(n_trials)])
        return idx_list

    # ------------------ Compatible with old interface ------------------ #
    def get_folds(self, train_idx, test_idx):
        """Read specified subjects' data on demand and concatenate, avoid loading all data at once."""
        train_X, train_Y = [], []
        test_X, test_Y = [], []
        test_dataset_indices = []

        # Process training set
        for global_sub_idx in train_idx:
            uda_idx, local_sub_idx = self.subject_to_dataset[global_sub_idx]
            X_sub, Y_sub = self.udas[uda_idx].get_sub(local_sub_idx)
            # Pad length & label remap
            X_sub = self._pad_trial(X_sub)
            Y_sub = self.remap_labels_single(Y_sub, uda_idx)
            train_X.append(X_sub)
            train_Y.append(Y_sub)

        # Process test set
        for global_sub_idx in test_idx:
            uda_idx, local_sub_idx = self.subject_to_dataset[global_sub_idx]
            X_sub, Y_sub = self.udas[uda_idx].get_sub(local_sub_idx)
            X_sub = self._pad_trial(X_sub)
            Y_sub = self.remap_labels_single(Y_sub, uda_idx)

            test_X.append(X_sub)
            test_Y.append(Y_sub)
            test_dataset_indices.append(np.full(len(Y_sub), uda_idx))

        # Concatenate to numpy arrays, keep consistent with original logic
        train_X = np.concatenate(train_X)
        train_Y = np.concatenate(train_Y)
        test_X = np.concatenate(test_X)
        test_Y = np.concatenate(test_Y)
        test_dataset_indices = np.concatenate(test_dataset_indices)

        return train_X, train_Y, test_X, test_Y, test_dataset_indices

    def get_sub(self, idx):
        """Return data for a single subject, interface unchanged"""
        uda_idx, local_sub_idx = self.subject_to_dataset[idx]
        X_sub, Y_sub = self.udas[uda_idx].get_sub(local_sub_idx)
        X_sub = self._pad_trial(X_sub)
        Y_sub = self.remap_labels_single(Y_sub, uda_idx)
        return X_sub, Y_sub, uda_idx

    def output_class_mapping(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.class_to_label_map, f, indent=4)

# ===== Pretrain with DataLoader (lazy) =====

def Pretrain_dl(config, preCkpt, preLog, train_loader, val_loader, test_loader, device_ids, seed, encoder_dir=None, debug_one_step=False):
    if isinstance(val_loader, list):
        cb = [ModelCheckpoint(dirpath=f"{RunFolder}/ckpt", filename='{epoch:02d}-{mean_acc:.3f}', monitor='mean_acc', mode='max', save_top_k=1, save_last=True, every_n_epochs=1, save_on_train_epoch_end=False)]
    else:
        cb = [ModelCheckpoint(dirpath=f"{RunFolder}/ckpt", filename='{epoch:02d}-{acc:.3f}', monitor='acc', mode='max', save_top_k=1, save_last=True, every_n_epochs=1, save_on_train_epoch_end=False)]

    model = EEG_Encoder_Module(config, 'pre', len(train_loader), dataset_names=datasets, seed=seed)
    strategy = 'ddp_find_unused_parameters_true' if isinstance(device_ids, (list, tuple)) and len(device_ids) > 1 else 'auto'
    devices = device_ids if isinstance(device_ids, (list, tuple)) and len(device_ids) > 0 else 1
    trainer_kwargs = dict(
        strategy=strategy, accelerator='gpu', devices=devices, max_epochs=PreTrainEpochs,
        callbacks=cb, enable_progress_bar=False, enable_checkpointing=True,
        precision='bf16-mixed', logger=False, benchmark=True, deterministic=False,
        enable_model_summary=False, check_val_every_n_epoch=CheckValEveryN,
        sync_batchnorm=True, use_distributed_sampler=True
    )
    if debug_one_step:
        # Run only 1 step for train/val/test
        trainer_kwargs.update(dict(limit_train_batches=1, limit_val_batches=1, limit_test_batches=1, max_steps=1, num_sanity_val_steps=0))
    trainer = pl.Trainer(**trainer_kwargs)


    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


    # 训练曲线与日志输出
    if trainer.is_global_zero:
        output_acc_chart(model.epoch_acc, model.train_acc, model.train_loss, f"{RunFolder}/pre_acc_chart.png", dataset_names=datasets)
        epoch_acc = np.array(model.epoch_acc).T
        if isinstance(val_loader, list):
            for logf, acc in zip(preLog, epoch_acc):
                acc_interp = np.interp(np.linspace(0,1,PreTrainEpochs), np.linspace(0,1,len(acc)), acc)
                np.savetxt(logf, acc_interp, delimiter=',', fmt='%.4f')
        else:
            for logf, acc in zip(preLog, epoch_acc):
                acc_interp = np.interp(np.linspace(0,1,PreTrainEpochs), np.linspace(0,1,len(acc)), acc)
                np.savetxt(logf, acc_interp, delimiter=',', fmt='%.4f')

    # 使用最佳权重在测试集上评估
    try:
        trainer.test(model=None, dataloaders=test_loader, ckpt_path='best')
    except Exception:
        trainer.test(model=model, dataloaders=test_loader)

    # 收集测试准确率
    test_acc_values = []
    if isinstance(test_loader, list):
        for i in range(len(test_loader)):
            key = f"test_acc/dataloader_idx_{i}"
            if key in trainer.callback_metrics:
                v = trainer.callback_metrics[key]
                v = v.item() if torch.is_tensor(v) else float(v)
                test_acc_values.append(v)
    else:
        v = trainer.callback_metrics.get("test_acc", None)
        if v is not None:
            v = v.item() if torch.is_tensor(v) else float(v)
            test_acc_values.append(v)

    if encoder_dir is not None and trainer.is_global_zero:
        os.makedirs(encoder_dir, exist_ok=True)

        best_ckpt_path = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if hasattr(callback, 'best_model_path') and callback.best_model_path:
                    best_ckpt_path = callback.best_model_path
                    break

        # 复制最佳模型
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            dest_path = os.path.join(encoder_dir, "fixed_split.ckpt")
            shutil.copy2(best_ckpt_path, dest_path)
            print(f"Best model copied to: {dest_path}")
        else:
            print("Warning: Could not find best model checkpoint to copy")

    # 保存最终权重
    torch.save(model.model.state_dict(), preCkpt + '_last.ckpt')
    # Return test accuracy and additional metrics
    test_metrics = getattr(model, 'test_metrics', {})
    return np.array(test_acc_values, dtype=np.float32), test_metrics

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--ds', type=str, default='CS')
    args.add_argument('--time_length', type=int, default=10)
    args.add_argument('--dim1', type=int, default=32)
    args.add_argument('--dim2', type=int, default=96)
    args.add_argument('--head', type=str, default='V1')
    args.add_argument('--win', type=int, default=100)
    args.add_argument('--step', type=int, default=100)
    args.add_argument('--lay', type=str, default='4')
    args.add_argument('--bs', type=int, default=300)
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU ids, e.g. "0" or "0,1"')
    args.add_argument('--workers', type=int, default=4, help='DataLoader num_workers')
    args.add_argument('--persistent_workers', action='store_true', help='Enable DataLoader persistent_workers')
    args.add_argument('--encoder_dir', type=str, default=None, help='Directory to save the best encoder model')
    args.add_argument('--seed', type=int, default=0, help='Random seed')
    args = args.parse_args()

    try:
        seed_all(args.seed)
    except Exception:
        pass

    datasets = args.ds.split(',')
    udas = []
    classes = []
    CH_NAMES, ZONES = template_ch_names, template_zones
    # CH_NAMES, ZONES = CH_NAMES, ZONES
    for ds in datasets:
        task_info = find_dataset(ds)
        uda = h5Dataset(ds, task_info.classes)
        udas.append(uda)
        classes.append(task_info.classes)

    resample_rate = 250
    time_length = args.time_length
    print("Time length:", time_length)
    uda = MultiSourceDataset(udas, resample_rate, time_length, classes)
    n_classes = uda.n_classes  # 使用MultiSourceDataset计算的类别数

    # Debug output: show class mapping
    print("=== Class Mapping Info ===")
    print(f"Total number of classes: {n_classes}")
    print("Class name to label mapping:")
    for class_name, label in sorted(uda.class_to_label_map.items(), key=lambda x: x[1]):
        print(f"  {class_name} -> {label}")
    print("=================")

    print("Channel Length: ", len(CH_NAMES), "Zone Dict:", {k: len(v) for k, v in ZONES.items() if v})
    config = PretrainedConfig(
        electrodes=CH_NAMES,
        zone_dict=ZONES,
        dim_cnn=args.dim1,
        dim_token=args.dim2,
        head=args.head,
        seq_len=time_length*resample_rate,
        window_len=int(args.win),
        slide_step=int(args.step),
        num_layers=int(args.lay),
        n_classes=n_classes,
        num_heads=8,
        dropout=0.2,
        cross_subject=True,
        lr=args.lr
    )

    RunFolder = f"channels_average/{Tower.name}-{time_length}-{args.ds if len(datasets) < 10 else len(datasets)}-{args.seed}-ch:{len(CH_NAMES)}/{config.head}-{config.dim_cnn}-{config.dim_token}-{config.window_len}-{config.slide_step}-{config.num_layers}-{config.lr}-bs={args.bs}--cs={config.cross_subject}"
    os.makedirs(f"{RunFolder}/ckpt", exist_ok=True)
    os.makedirs(f"{RunFolder}/ckpt-tune", exist_ok=True)
    uda.output_class_mapping(f"{RunFolder}/class_mapping.json")
    # ---------- 使用固定 subject 划分（优先从 dataset_split_config 读取；否则默认 7/1.5/1.5） ---------- #
    dataset_subject_ranges = []
    start_idx = 0
    for u in udas:
        n_subjects = u.n_subjects
        dataset_subject_ranges.append((start_idx, start_idx + n_subjects))
        start_idx += n_subjects

    def _safe_get_split(ds_name: str, n_subj: int):
            # 1) 尝试从配置读取
            split = None
            if split_cfg is not None:
                try:
                    attr_name = f"{ds_name}_split"
                    if hasattr(split_cfg, attr_name):
                        split = getattr(split_cfg, attr_name)
                    else:
                        print(f"[split-config] 读取 {ds_name} 失败. 使用默认比例划分。", flush=True)
                except Exception as e:
                    print(f"[split-config] 读取 {ds_name} 失败: {e}. 使用默认比例划分。", flush=True)
                    split = None
            # 2) 若无配置或失败，使用 7/1.5/1.5
            if split is None:
                train_n = int(np.floor(0.70 * n_subj))
                val_n = int(np.floor(0.15 * n_subj))
                test_n = n_subj - train_n - val_n
                return {
                    'train': list(range(0, train_n)),
                    'val': list(range(train_n, train_n + val_n)),
                    'test': list(range(train_n + val_n, train_n + val_n + test_n))
                }
            # 3) 规范化配置中的索引（避免越界），并转为 list
            out = {}
            for k in ['train', 'val', 'test']:
                v = split.get(k, [])
                v = list(v)  # 兼容 range / list
                v = [i for i in v if 0 <= int(i) < n_subj]
                out[k] = v
            return out

    # 按每个数据集单独划分，再映射到全局 subject 索引
    all_train_idx, all_val_idx, all_test_idx = [], [], []
    for ds_idx, (ds_name, u) in enumerate(zip(datasets, udas)):
        local_n = u.n_subjects
        local_split = _safe_get_split(ds_name, local_n)
        start, end = dataset_subject_ranges[ds_idx]
        all_train_idx.extend([start + i for i in local_split['train']])
        all_val_idx.extend([start + i for i in local_split['val']])
        all_test_idx.extend([start + i for i in local_split['test']])
    # print(f"All train idx: {all_train_idx}")
    # print(f"All val idx: {all_val_idx}")
    # print(f"All test idx: {all_test_idx}")

    # 构建训练/验证 dataloader（验证即 val split）
    ckpt = f"{RunFolder}/ckpt/split"
    if len(datasets) > 1:
        # 多数据集：分别为每个数据集构建各自的 val dataloader，以便分别统计/展示
        logf_list = [f"{RunFolder}/pre-dataset-{ds}.csv" for ds in datasets]
        logitsf_list = [f"{RunFolder}/pre-dataset-{ds}.npy" for ds in datasets]
    else:
        logf_list = [f"{RunFolder}/pre.csv"]
        logitsf_list = [f"{RunFolder}/pre.npy"]

    # 若日志文件均已存在且已聚合，可跳过（与原逻辑一致）
    test_acc_values = None
    if not all_exist(logf_list) and not os.path.exists(f"{RunFolder}/pre.npz"):
        train_indices = uda.build_trial_indices(all_train_idx)
        val_indices = uda.build_trial_indices(all_val_idx)
        test_indices = uda.build_trial_indices(all_test_idx)

        train_ds = TrialDataset(uda, train_indices)
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.persistent_workers)

        if len(datasets) > 1:
            val_loader_list = []
            test_loader_list = []
            for ds_idx in range(len(datasets)):
                ds_val = [ti for ti in val_indices if ti[0] == ds_idx]
                if ds_val:
                    tl = DataLoader(TrialDataset(uda, ds_val), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.persistent_workers)
                    val_loader_list.append(tl)
                ds_test = [ti for ti in test_indices if ti[0] == ds_idx]
                if ds_test:
                    tl2 = DataLoader(TrialDataset(uda, ds_test), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.persistent_workers)
                    test_loader_list.append(tl2)
            val_loader = val_loader_list if len(val_loader_list) > 1 else val_loader_list[0]
            test_loader = test_loader_list if len(test_loader_list) > 1 else test_loader_list[0]
        else:
            val_loader = DataLoader(TrialDataset(uda, val_indices), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.persistent_workers)
            test_loader = DataLoader(TrialDataset(uda, test_indices), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.persistent_workers)

        # 训练 + 测试
        try:
            device_ids = [int(x) for x in str(args.gpus).split(',') if x.strip() != '']
        except Exception:
            device_ids = [0]
        result = Pretrain_dl(config, ckpt, logf_list, train_loader, val_loader, test_loader, device_ids, args.seed, args.encoder_dir)

        # Handle return value (could be tuple or just array for backward compatibility)
        if isinstance(result, tuple):
            test_acc_values, test_metrics = result
        else:
            test_acc_values, test_metrics = result, {}

        # 保存测试准确率中间结果
        if test_acc_values is not None and len(test_acc_values) > 0:
            np.save(f"{RunFolder}/test_acc.npy", test_acc_values)

        # 保存额外指标
        if test_metrics:
            import json
            with open(f"{RunFolder}/test_metrics.json", 'w') as f:
                json.dump(test_metrics, f, indent=4)
    else:
        print(f"Skipping fixed-split training because outputs already exist: {logf_list}")
        # 若跳过训练，尝试加载既有测试结果
        try:
            test_acc_values = np.load(f"{RunFolder}/test_acc.npy")
        except Exception:
            test_acc_values = None

        # 尝试加载已有的额外指标
        test_metrics = {}
        try:
            import json
            with open(f"{RunFolder}/test_metrics.json", 'r') as f:
                test_metrics = json.load(f)
        except Exception:
            test_metrics = {}

    pfn_epoacc = f"{RunFolder}/pre_epoacc.npy"
    pfn_final = f"{RunFolder}/pre.npz"
    # 固定划分：多数据集按数据集各一份日志；单数据集一份日志
    if len(datasets) > 1:
        csv_files = [f"{RunFolder}/pre-dataset-{ds}.csv" for ds in datasets]
        npy_files = [f"{RunFolder}/pre-dataset-{ds}.npy" for ds in datasets]
    else:
        csv_files = [f"{RunFolder}/pre.csv"]
        npy_files = [f"{RunFolder}/pre.npy"]

    if is_rank0():
        convert_epoch_acc_to_npy(csv_files, pfn_epoacc)
        # 尝试在固定划分路径下获取 test_acc（若上面变量未定义或为空，则从文件加载）
        try:
            _ta = test_acc_values if ('test_acc_values' in locals() and test_acc_values is not None and len(test_acc_values) > 0) else np.load(f"{RunFolder}/test_acc.npy")
        except Exception:
            _ta = np.array([])

        # 尝试获取额外指标
        _tm = test_metrics if 'test_metrics' in locals() and test_metrics else {}
        if not _tm:
            try:
                import json
                with open(f"{RunFolder}/test_metrics.json", 'r') as f:
                    _tm = json.load(f)
            except Exception:
                _tm = {}

        # 保存最终 npz，包含 epoacc（验证曲线）、test_acc（测试集柱状图使用）、datasets 名称和额外指标
        save_dict = {
            'epoacc': np.load(pfn_epoacc),
            'test_acc': _ta,
            'datasets': np.array(datasets)
        }

        # 添加额外指标到 npz 文件中
        if _tm:
            # 为每个数据集的指标创建独立的数组
            for ds_name, metrics in _tm.items():
                for metric_name, value in metrics.items():
                    key = f"{ds_name}_{metric_name}"
                    save_dict[key] = np.array([value])  # 保存为数组格式

            # 同时保存原始字典格式（作为字符串）
            save_dict['test_metrics_json'] = np.array([json.dumps(_tm)])

        np.savez(pfn_final, **save_dict)
        os.remove(pfn_epoacc)
