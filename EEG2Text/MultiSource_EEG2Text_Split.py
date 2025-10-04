import argparse
import torch
import pickle
import time
import os
import wandb
import gc
import einops
from pathlib import Path

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
)

# https://github.com/openai/whisper/discussions/64
import whisper

from train_utils import seed_all, cosine_lr
from FAST_v2 import FAST as Tower
from EEG_dataset_config import configs, Split_Config
from lazy_dataset import h5Dataset as LazyH5Dataset, DATASET_ROOT
from text_dataset import ICLWithSupportDataset_Lazy, ICLWithSupportDataset_Test_Lazy
from text_dataset import WhisperICLCollator, WhisperICLCollator_Test
import numpy as np

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings(
    "ignore", message=".*find_unused_parameters=True was specified in DDP constructor.*"
)
warnings.filterwarnings(
    "ignore", message="No device id is provided via `init_process_group` or `barrier `."
)


def find_config_by_name(name):
    for config in configs:
        if config["name"] == name:
            return config
    raise ValueError(f"Config with name {name} not found")


def safe_load_state_dict(model, checkpoint_state_dict, skip_key_substrings=None):
    """Load checkpoint_state_dict into model but skip keys that match any substring in
    skip_key_substrings or whose shapes don't match the current model. This avoids
    size-mismatch errors when the checkpoint's head/last-layer size differs.

    Args:
        model: torch.nn.Module to load weights into.
        checkpoint_state_dict: dict from checkpoint['state_dict']
        skip_key_substrings: list of substrings; if any substring is in a key, that key is skipped.
    """
    if skip_key_substrings is None:
        skip_key_substrings = []
    model_state = model.state_dict()
    filtered = {}
    skipped_keys = []
    for k, v in checkpoint_state_dict.items():
        # skip if user requested
        if any(sub in k for sub in skip_key_substrings):
            skipped_keys.append(k)
            continue
        # skip if key not present in model
        if k not in model_state:
            skipped_keys.append(k)
            continue
        # skip if shape mismatch
        if v.shape != model_state[k].shape:
            skipped_keys.append(k)
            continue
        filtered[k] = v

    # Load the filtered dict with strict=False so missing keys in the checkpoint are tolerated
    model.load_state_dict(filtered, strict=False)
    if skipped_keys:
        # keep the message concise
        short = (
            skipped_keys
            if len(skipped_keys) <= 10
            else skipped_keys[:10] + [f"...(+{len(skipped_keys)-10} more)"]
        )
        print(f"safe_load_state_dict: skipped {len(skipped_keys)} keys: {short}")


class WhisperModelModule(LightningModule):
    def __init__(
        self,
        niter_per_ep,
        tokenizer,
        model_name="tiny",
        dataset_name="",
        config=None,
        wd=None,
        seed=0,
    ):
        super().__init__()
        seed_all(seed)
        self.tokenizer = tokenizer
        self.whisper = whisper.load_model(model_name)
        self.whisper.requires_grad_(False)
        self.wd = wd

        # class cfg:
        #     head = 'D0'
        #     n_channels = 63
        #     dim_cnn = 32
        #     dim_token = 96
        #     seq_len = 12*200
        #     window_len = 100
        #     slide_step = 100
        #     n_classes = 5
        #     num_layers = 4
        #     num_attention_heads = 8
        #     dropout = 0.2
        #     num_layers = 4
        #     num_attention_heads = 8
        #     dropout = 0.2
        if config is None:
            config = find_config_by_name(dataset_name)

        class cfg:
            head = config["head"]
            n_channels = config["n_channels"]
            electrodes = config["electrodes"]
            zone_dict = config["zones"]
            dim_cnn = config["dim_cnn"]
            dim_token = config["dim_token"]
            seq_len = config["seq_len"]
            window_len = config["window_len"]
            slide_step = config["slide_step"]
            n_classes = config["n_classes"]
            num_layers = config["num_layers"]
            num_heads = config["num_heads"]
            dropout = config["dropout"]

        print(
            f"head={cfg.head}, n_channels={cfg.n_channels}, dim_cnn={cfg.dim_cnn}, dim_token={cfg.dim_token}, seq_len={cfg.seq_len}, window_len={cfg.window_len}, slide_step={cfg.slide_step}, n_classes={cfg.n_classes}, num_layers={cfg.num_layers}, num_heads={cfg.num_heads}, dropout={cfg.dropout}"
        )

        self.fast = Tower(cfg)
        if "," in dataset_name:
            ds_names = dataset_name.split(",")
            if len(ds_names) > 10:
                encoder_path = (
                    f'eeg_encoders/ds_num={len(ds_names)}/{config["fold_ckpt"]}.ckpt'
                )
            else:
                encoder_path = (
                    f'eeg_encoders/{",".join(ds_names)}/{config["fold_ckpt"]}.ckpt'
                )
        else:
            encoder_path = f'eeg_encoders/{dataset_name}/{config["fold_ckpt"]}.ckpt'
        encoder_path = DATASET_ROOT + "/" + encoder_path
        if config["fold_ckpt"] == "finetune":
            print("Skip encoder loading for finetune")
        else:
            self._load_fast_checkpoint(encoder_path)
            self.fast.requires_grad_(False)
        self.fast_dim_token = self.fast.cls_token.shape[-1]
        self.connect = nn.Sequential(
            nn.Linear(self.fast_dim_token, self.whisper.dims.n_audio_state), nn.GELU()
        )

        self.eeg_start = nn.Parameter(torch.randn(self.fast_dim_token))
        self.eeg_end = nn.Parameter(torch.randn(self.fast_dim_token))
        self.position_embeddings = nn.Embedding(15, self.fast_dim_token)  # 最多10个位置
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.lr_list = cosine_lr(1, 0.5, 100, niter_per_ep, warmup_epochs=1)
        self._tloss, self._vloss = [], []
        self.save_hyperparameters(ignore=["tokenizer"])
        # dataset name for saving predictions
        self.test_ds_name = ""
        self.test_step_outputs = []
        self.stage = 1
        # classes used for scoring metrics (deduplicate while preserving order)
        classes_in = config.get("classes", [])
        classes_in = list(dict.fromkeys(classes_in)) if classes_in else []
        self.classes = [f"<{c}>" for c in classes_in]

    def compute_balanced_accuracy_from_indices(
        self, y_true_idx, y_pred_idx, num_classes
    ):
        # Macro average of per-class recall, only over classes that appear in y_true
        y_true_idx = np.asarray(y_true_idx)
        y_pred_idx = np.asarray(y_pred_idx)
        recalls = []
        for c in range(num_classes):
            mask = y_true_idx == c
            total_c = int(mask.sum())
            if total_c == 0:
                continue
            correct_c = int((y_pred_idx[mask] == c).sum())
            recalls.append(correct_c / total_c)
        if len(recalls) == 0:
            return 0.0
        return float(np.mean(recalls))

    def compute_balanced_accuracy_from_texts(self, pred_texts, target_texts):
        # Fallback to simple accuracy if classes are not defined
        if not hasattr(self, "classes") or len(self.classes) == 0:
            if len(target_texts) == 0:
                return 0.0
            correct = sum(1 for p, t in zip(pred_texts, target_texts) if p == t)
            return correct / max(1, len(target_texts))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_true_idx, y_pred_idx = [], []
        for p, t in zip(pred_texts, target_texts):
            if t not in class_to_idx:
                continue
            t_idx = class_to_idx[t]
            if p in class_to_idx:
                p_idx = class_to_idx[p]
            else:
                # ensure incorrect prediction, count as FN for true class
                p_idx = (t_idx + 1) % len(self.classes)
            y_true_idx.append(t_idx)
            y_pred_idx.append(p_idx)
        if len(y_true_idx) == 0:
            return 0.0
        return self.compute_balanced_accuracy_from_indices(
            y_true_idx, y_pred_idx, len(self.classes)
        )

    def _load_fast_checkpoint(self, ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"Loading from model member variable in {ckpt_path}")
            model_state_dict = checkpoint["state_dict"]
            model_keys = [k for k in model_state_dict.keys() if k.startswith("model.")]
            if model_keys:
                print(
                    f"Found parameters with 'model.' prefix in model member variable, removing prefix"
                )
                new_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith("model."):
                        new_key = key[6:]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                self.fast.load_state_dict(new_state_dict)
            else:
                self.fast.load_state_dict(model_state_dict)

        except Exception as e:
            print(f"Error loading checkpoint {ckpt_path}: {e}")
            try:
                print(f"Fallback: Loading directly as state_dict from {ckpt_path}")
                self.fast.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=False)
                )
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise e2

    def on_train_epoch_start(self):
        if self.current_epoch <= 5:
            self.whisper.requires_grad_(True)
            self.connect.requires_grad_(True)
            self.fast.requires_grad_(False)
            print(
                f"Stage {self.stage} Stage epoch: {self.current_epoch} Training connect and whisper"
            )
        else:
            self.whisper.requires_grad_(True)
            self.connect.requires_grad_(True)
            self.fast.requires_grad_(True)
            print(f"Stage {self.stage} Stage epoch: {self.current_epoch} Training all")
        # reset training epoch accumulators
        self._train_preds = []
        self._train_targets = []

    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {"params": self.fast.parameters(), "lr": 0.000005},
            {"params": self.whisper.parameters(), "lr": 0.00005},
            {"params": self.connect.parameters(), "lr": 0.00005},
        ]

        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda epoch: self.lr_list[self.global_step - 1]
        )
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def add_sample_boundaries(self, features, B):
        k_total = features.shape[1]
        sample_features = []
        for i in range(k_total):
            sample_feat = features[:, i, :, :]  # (B, N, F)
            pos_embed = self.position_embeddings(
                torch.tensor(i, device=features.device)
            )
            sample_feat = sample_feat + pos_embed.unsqueeze(0).unsqueeze(0)

            sample_with_boundary = torch.cat(
                [
                    self.eeg_start.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1),
                    sample_feat,
                    self.eeg_end.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1),
                ],
                dim=1,
            )
            sample_features.append(sample_with_boundary)

        return torch.cat(sample_features, dim=1)

    def compute_loss(self, batch):
        eeg = batch["eeg"]
        input_tokens = batch["input_tokens"]
        target_tokens = batch["target_tokens"]

        B = eeg.shape[0]
        eeg = einops.rearrange(eeg, "B k C T -> (B k) C T")
        feature = self.fast.forward_get_tokens(eeg)  # (B k, N, F)

        feature = einops.rearrange(feature, "(B k) N F -> B k N F", B=B)
        feature = self.add_sample_boundaries(feature, B)

        feature = self.connect(feature)
        logits = self.whisper.decoder(input_tokens, feature)
        logits = logits.permute(0, 2, 1)

        loss_ce = self.loss_fn(logits, target_tokens)
        return loss_ce

    def training_step(self, batch, batch_id):
        loss = self.compute_loss(batch)
        self.log(
            f"train/loss_{self.stage}",
            loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_id):
        predictions = self.generate_with_input_until_eot(
            self.tokenizer, batch["eeg"], batch["input_tokens"]
        )
        pred_list, target_list = [], []
        for i, prediction in enumerate(predictions):
            pred_list.append(prediction)
            target_list.append(batch["target_text"][i])
        bac = self.compute_balanced_accuracy_from_texts(pred_list, target_list)
        self.log("val_acc", float(bac), on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(
            f"val/acc_{self.stage}",
            float(bac),
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return bac

    def on_validation_epoch_end(self):
        accuracy = self.trainer.callback_metrics.get("val_acc", 0)
        if self.trainer.is_global_zero:
            print(f"Epoch {self.current_epoch}, Validation Accuracy: {accuracy:.4f}")

    def generate_with_input_until_eot(
        self, tokenizer, eeg, input_tokens, max_length=128, temperature=0.7
    ):
        B = eeg.shape[0]
        eeg = einops.rearrange(eeg, "B k C T -> (B k) C T")
        eeg_features = self.fast.forward_get_tokens(eeg)

        eeg_features = einops.rearrange(eeg_features, "(B k) N F -> B k N F", B=B)
        eeg_features = self.add_sample_boundaries(eeg_features, B)

        eeg_features = self.connect(eeg_features)
        current_tokens = input_tokens
        input_lengths = [len(tokens) for tokens in input_tokens]
        logits = self.whisper.decoder(current_tokens, eeg_features)
        batch_size = current_tokens.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=eeg.device)
        end_positions = torch.full((batch_size,), -1, device=eeg.device)

        while (not finished.all()) and (current_tokens.shape[1] < max_length):
            logits = self.whisper.decoder(current_tokens, eeg_features)
            next_token_logits = logits[:, -1, :] / temperature

            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            current_tokens = torch.cat([current_tokens, next_tokens], dim=1)

            # Update finished and end_positions
            is_eot = next_tokens.squeeze(-1) == tokenizer.eot
            newly_finished = ~finished & is_eot
            end_positions = torch.where(
                newly_finished, current_tokens.shape[1] - 1, end_positions
            )
            finished = finished | is_eot

        generated_texts = []
        for i in range(batch_size):
            end_pos = end_positions[i].item()
            if end_pos == -1:
                end_pos = current_tokens.shape[1]

            # Only take tokens after input to end_pos
            output_tokens = current_tokens[i, input_lengths[i] : end_pos].tolist()
            text = tokenizer.decode(output_tokens)
            generated_texts.append(text)
        return generated_texts

    def compute_class_scores(self, eeg, input_tokens, candidate_texts):
        # Return shape: (B, num_classes) of summed log-probs for each candidate text
        B = eeg.shape[0]
        eeg = einops.rearrange(eeg, "B k C T -> (B k) C T")
        eeg_features = self.fast.forward_get_tokens(eeg)
        eeg_features = einops.rearrange(eeg_features, "(B k) N F -> B k N F", B=B)
        eeg_features = self.add_sample_boundaries(eeg_features, B)
        eeg_features = self.connect(eeg_features)
        num_classes = len(candidate_texts)
        scores = torch.zeros(
            (B, num_classes), device=eeg_features.device, dtype=torch.float32
        )
        if num_classes == 0:
            return scores
        with torch.no_grad():
            for i in range(B):
                base_tokens = input_tokens[i : i + 1]
                efeat = eeg_features[i : i + 1]
                for j, text in enumerate(candidate_texts):
                    tok_ids = self.tokenizer.encode(text)
                    if len(tok_ids) == 0:
                        continue
                    current = base_tokens
                    total_logp = 0.0
                    for tid in tok_ids:
                        logits = self.whisper.decoder(current, efeat)
                        next_logits = logits[:, -1, :]
                        log_probs = torch.log_softmax(next_logits, dim=-1)
                        total_logp = total_logp + log_probs[0, tid]
                        next_token = torch.tensor(
                            [[tid]], device=current.device, dtype=current.dtype
                        )
                        current = torch.cat([current, next_token], dim=1)
                    scores[i, j] = total_logp
        return scores

    def get_unique_filepath(self, base_path):
        if not base_path.exists():
            return base_path

        counter = 1
        while True:
            new_path = (
                base_path.parent / f"{base_path.stem}_{counter}{base_path.suffix}"
            )
            if not new_path.exists():
                return new_path
            counter += 1

    def on_test_epoch_start(self):
        # Reset container every test run
        if self.global_rank != 0:
            return
        self.test_step_outputs = []
        if self.trainer.is_global_zero:
            if self.wd is not None:
                wd = self.wd
            ds_name = getattr(self, "test_ds_name", "")
            ds_name = ds_name.replace("/", "_").replace(",", "_") if ds_name else ""
            filename = f"predictions_{ds_name}.txt" if ds_name else "predictions.txt"
            self.pred_file = self.get_unique_filepath(wd / filename)

            with open(self.pred_file, "w") as f:
                f.write("")
        # Set per-dataset classes if possible
        try:
            if getattr(self, "test_ds_name", ""):
                cfg = find_config_by_name(self.test_ds_name)
                ds_classes = cfg.get("classes", [])
                classes_in = list(dict.fromkeys(ds_classes)) if ds_classes else []
                self.classes = [f"<{c}>" for c in classes_in]

        except Exception:
            pass

    def test_step(self, batch, batch_id):
        if self.global_rank != 0:
            return
        predictions = self.generate_with_input_until_eot(
            self.tokenizer, batch["eeg"], batch["input_tokens"]
        )
        pred_list, target_list = [], []
        for i, prediction in enumerate(predictions):
            pred_list.append(prediction)
            target_list.append(batch["target_text"][i])

        input_texts = [
            self.tokenizer.decode(input_tokens)
            for input_tokens in batch["input_tokens"]
        ]

        # Compute class probability scores for PR AUC/ROC AUC
        probs_np = None
        true_idx = None
        if hasattr(self, "classes") and len(self.classes) > 0:
            class_scores = self.compute_class_scores(
                batch["eeg"], batch["input_tokens"], self.classes
            )
            probs = torch.softmax(class_scores, dim=1)
            probs_np = probs.detach().cpu().numpy()
            # Map target text to class index (batch size is 1 for test)
            true_idx = [
                self.classes.index(t) if t in self.classes else -1 for t in target_list
            ]

        if hasattr(self, "pred_file"):
            with open(self.pred_file, "a") as f:
                for inp_text, pred, target in zip(input_texts, pred_list, target_list):
                    f.write(f"input: {inp_text}, pred: {pred}, target: {target}\n")

        output = {"predictions": pred_list, "targets": target_list}
        if probs_np is not None:
            output["probs"] = probs_np
        if true_idx is not None:
            output["true_idx"] = true_idx
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        if self.global_rank != 0:
            return
        outputs = self.test_step_outputs

        all_predictions = []
        all_targets = []
        all_probs = []
        all_true_idx = []
        for output in outputs:
            all_predictions.extend(output["predictions"])
            all_targets.extend(output["targets"])
            if "probs" in output:
                all_probs.append(output["probs"])  # (B, C)
            if "true_idx" in output:
                all_true_idx.extend(output["true_idx"])  # list of ints

        # Balanced accuracy (prefer logits-based prediction if available)
        total_bac = 0.0
        computed_bac = False
        if len(all_probs) > 0 and hasattr(self, "classes") and len(self.classes) > 0:
            try:
                y_scores = np.concatenate(all_probs, axis=0)
                valid_mask = np.array(
                    [idx >= 0 and idx < len(self.classes) for idx in all_true_idx]
                )
                if valid_mask.any():
                    y_idx = np.array(all_true_idx)[valid_mask]
                    y_scores = y_scores[valid_mask]
                    y_pred = np.argmax(y_scores, axis=1)
                    total_bac = self.compute_balanced_accuracy_from_indices(
                        y_idx, y_pred, len(self.classes)
                    )
                    computed_bac = True
            except Exception:
                computed_bac = False
        if not computed_bac:
            total_bac = self.compute_balanced_accuracy_from_texts(
                all_predictions, all_targets
            )
        self.log(
            "test/accuracy",
            float(total_bac),
            on_epoch=True,
            sync_dist=False,
            prog_bar=True,
        )
        self.log(
            "test/balanced_accuracy",
            float(total_bac),
            on_epoch=True,
            sync_dist=False,
            prog_bar=False,
        )

        # Compute ROC/PR AUC for binary & multiclass; and Kappa/Weighted-F1 for multiclass
        if len(all_probs) > 0 and hasattr(self, "classes") and len(self.classes) > 1:
            try:
                y_scores = np.concatenate(all_probs, axis=0)
                valid_mask = np.array(
                    [idx >= 0 and idx < len(self.classes) for idx in all_true_idx]
                )
                if valid_mask.any():
                    y_idx = np.array(all_true_idx)[valid_mask]
                    y_scores = y_scores[valid_mask]
                    num_classes = len(self.classes)
                    y_true_onehot = np.zeros(
                        (y_idx.shape[0], num_classes), dtype=np.int32
                    )
                    y_true_onehot[np.arange(y_idx.shape[0]), y_idx] = 1
                    # Filter classes without both positive and negative samples
                    pos_counts = y_true_onehot.sum(axis=0)
                    neg_counts = y_true_onehot.shape[0] - pos_counts
                    valid_classes = np.where((pos_counts > 0) & (neg_counts > 0))[0]

                    # Binary: record single ROC AUC and PR AUC
                    if num_classes == 2 and valid_classes.size == 2:
                        pos_index = 1
                        y_true_bin = (y_idx == pos_index).astype(np.int32)
                        y_score_pos = y_scores[:, pos_index]
                        try:
                            roc_auc_bin = roc_auc_score(y_true_bin, y_score_pos)
                            self.log(
                                "test/roc_auc",
                                float(roc_auc_bin),
                                on_epoch=True,
                                sync_dist=False,
                                prog_bar=True,
                            )
                        except Exception:
                            pass
                        try:
                            pr_auc_bin = average_precision_score(
                                y_true_bin, y_score_pos
                            )
                            self.log(
                                "test/pr_auc",
                                float(pr_auc_bin),
                                on_epoch=True,
                                sync_dist=False,
                                prog_bar=True,
                            )
                        except Exception:
                            pass

                    # Multiclass & general: macro ROC/PR AUC
                    if valid_classes.size > 0:
                        roc_list = []
                        pr_list = []
                        for c in valid_classes:
                            try:
                                roc_list.append(
                                    roc_auc_score(y_true_onehot[:, c], y_scores[:, c])
                                )
                            except Exception:
                                pass
                            try:
                                pr_list.append(
                                    average_precision_score(
                                        y_true_onehot[:, c], y_scores[:, c]
                                    )
                                )
                            except Exception:
                                pass
                        if len(roc_list) > 0:
                            self.log(
                                "test/roc_auc_macro",
                                float(np.mean(roc_list)),
                                on_epoch=True,
                                sync_dist=False,
                                prog_bar=True,
                            )
                        if len(pr_list) > 0:
                            self.log(
                                "test/pr_auc_macro",
                                float(np.mean(pr_list)),
                                on_epoch=True,
                                sync_dist=False,
                                prog_bar=True,
                            )

                        # Additional multiclass metrics: Cohen's Kappa and weighted F1
                        if num_classes > 2:
                            y_pred = np.argmax(y_scores, axis=1)
                            try:
                                kappa = cohen_kappa_score(y_idx, y_pred)
                                self.log(
                                    "test/kappa",
                                    float(kappa),
                                    on_epoch=True,
                                    sync_dist=False,
                                    prog_bar=True,
                                )
                            except Exception:
                                pass
                            try:
                                wf1 = f1_score(y_idx, y_pred, average="weighted")
                                self.log(
                                    "test/f1_weighted",
                                    float(wf1),
                                    on_epoch=True,
                                    sync_dist=False,
                                    prog_bar=True,
                                )
                            except Exception:
                                pass
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--sot_first", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--supp_loss", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--first_stage_ckpt", type=str, default=None)
    parser.add_argument(
        "--first_stage_test", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument("--second_stage_ckpt", type=str, default=None)
    parser.add_argument("--ds_name", type=str, default="EMO_03_SEED_V,MI_01_KoreaU")
    parser.add_argument(
        "--fold_id",
        type=str,
        default="05",
        help="Not used in fixed split, but kept for compatibility",
    )
    parser.add_argument("--cross_ds4test", type=str, default="")
    parser.add_argument("--enc_version", type=str, default="V0")
    parser.add_argument("--time_len", type=int, default=30)
    parser.add_argument("--whisper", type=str, default="tiny")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finetune", type=lambda x: x.lower() == "true", default=False)
    args = parser.parse_args()

    # Try to import fixed subject split config; fallback to default 7/1.5/1.5 on runtime
    try:
        import dataset_split_config as split_cfg  # May contain user-defined subject splits
    except Exception as _e:
        split_cfg = None

    experiment_id = "EEG2Text_Whisper"
    run_ts = os.environ.get("RUN_TS")
    if run_ts is None:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        os.environ["RUN_TS"] = run_ts  # make it visible to spawned DDP children
    wd = Path(
        f"outputs/multi_source_{args.ds_name}/whisper_{args.whisper}_icl_{run_ts}"
    )
    (wd / "ckpt").mkdir(parents=True, exist_ok=True)
    gpus = [int(gpu) for gpu in args.gpus.split(",")]

    ds_names = args.ds_name.split(",")

    dataloader_num_workers = 0
    prefetch_factor = None
    persistent_workers = False

    main_dataset_config = Split_Config.copy()

    # ========= Lazy Loading =========
    udas = []
    all_classes = []
    for ds_name in ds_names:
        cfg = find_config_by_name(ds_name)
        uda_lazy = LazyH5Dataset(
            ds_name,
            classes=cfg["classes"],
            use_shm=True,
            time_len=args.time_len,
        )
        udas.append(uda_lazy)
        all_classes.append(cfg["classes"])

    # 计算每个数据集的 subject 范围 (全局索引)
    dataset_subject_ranges = []
    start_idx = 0
    for uda in udas:
        dataset_subject_ranges.append((start_idx, start_idx + uda.n_subjects))
        start_idx += uda.n_subjects

    def _safe_get_split(ds_name: str, n_subj: int):
        # 1) Try to read from config
        split = None
        if split_cfg is not None:
            try:
                attr_name = f"{ds_name}_split"
                if hasattr(split_cfg, attr_name):
                    split = getattr(split_cfg, attr_name)
                else:
                    print(
                        f"[split-config] Failed to read {ds_name}. Using default split.",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[split-config] Failed to read {ds_name}: {e}. Using default split.",
                    flush=True,
                )
                split = None
        # 2) If no config or failed, use 7/1.5/1.5
        if split is None:
            train_n = int(np.floor(0.70 * n_subj))
            val_n = int(np.floor(0.15 * n_subj))
            test_n = n_subj - train_n - val_n
            return {
                "train": list(range(0, train_n)),
                "val": list(range(train_n, train_n + val_n)),
                "test": list(range(train_n + val_n, train_n + val_n + test_n)),
            }
        # 3) Normalize indices from config (prevent out-of-bounds) and convert to list
        out = {}
        for k in ["train", "val", "test"]:
            v = split.get(k, [])
            v = list(v)  # compatible with range / list
            v = [i for i in v if 0 <= int(i) < n_subj]
            out[k] = v
        return out

    # Split each dataset individually, then map to global subject index
    all_train_idx, all_val_idx, all_test_idx = [], [], []
    for ds_idx, (ds_name, u) in enumerate(zip(ds_names, udas)):
        local_n = u.n_subjects
        local_split = _safe_get_split(ds_name, local_n)
        start, end = dataset_subject_ranges[ds_idx]
        all_train_idx.extend([start + i for i in local_split["train"]])
        all_val_idx.extend([start + i for i in local_split["val"]])
        all_test_idx.extend([start + i for i in local_split["test"]])

    if not all_val_idx and all_test_idx:
        print("Warning: Validation set is empty. Using test set for validation.")
        all_val_idx = all_test_idx

    if not all_test_idx:
        raise ValueError("Error: Test set is empty after split. Cannot proceed.")

    # train/test subject_map -> (start_trial, end_trial)
    def build_subject_map(
        global_subject_indices, dataset_subject_ranges, uda_list=None
    ):
        if uda_list is None:
            uda_list = udas
        mapping = {}
        cur_idx = 0
        for gsid in global_subject_indices:
            cum, uda_idx = 0, None
            for idx, (start, end) in enumerate(dataset_subject_ranges):
                if start <= gsid < end:
                    uda_idx = idx
                    break
                cum += uda_list[idx].n_subjects
            trial_len = uda_list[uda_idx].n_trials(gsid - cum)
            if trial_len < 15:
                continue
            mapping[gsid] = (cur_idx, cur_idx + trial_len)
            cur_idx += trial_len
        return mapping

    train_subject_map = build_subject_map(all_train_idx, dataset_subject_ranges, udas)
    val_subject_map = build_subject_map(all_val_idx, dataset_subject_ranges, udas)
    test_subject_map = build_subject_map(all_test_idx, dataset_subject_ranges, udas)

    dataset_config = main_dataset_config.copy()
    dataset_config["classes"] = [cls for classes in all_classes for cls in classes]
    dataset_config["fold_ckpt"] = (
        "fixed_split" if not args.finetune else "finetune"
    )  # Changed from fold_id
    dataset_config["n_classes"] = len(set(dataset_config["classes"]))
    dataset_config["head"] = args.enc_version
    dataset_config["seq_len"] = args.time_len * 250

    # ================= Helper Function: Build per-dataset TestLoaders =================
    def build_test_loaders_per_dataset(
        ds_names,
        dataset_subject_ranges,
        all_test_idx,
        udas,
        test_subject_map,
        classes,
        tokenizer,
        k_per_class,
        fixed_support_num,
        sot_first: bool = True,
        support_flag: bool = True,
    ):
        """为每个数据集构建独立的 *Lazy* TestLoader。"""
        test_loaders = []
        for i, ds_name in enumerate(ds_names):
            start, end = dataset_subject_ranges[i]
            ds_test_subjects = [sub for sub in all_test_idx if start <= sub < end]
            if not ds_test_subjects:
                continue  # No test subjects in this dataset

            new_subject_map = {}
            for sub_id in ds_test_subjects:
                s, e = test_subject_map[sub_id]
                new_subject_map[sub_id] = (s, e)

            TestDS = ICLWithSupportDataset_Test_Lazy(
                udas,
                tokenizer,
                classes=classes,
                subject_map=new_subject_map,
                k_per_class=k_per_class,
                max_len=256,
                fixed_support_num=fixed_support_num,
                support=support_flag,
                time_len=args.time_len,
            )

            collator = WhisperICLCollator_Test(tokenizer, sot_first, support_flag)
            loader = torch.utils.data.DataLoader(
                TestDS,
                batch_size=1,
                collate_fn=collator,
                shuffle=False,
                num_workers=dataloader_num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            test_loaders.append((ds_name, loader))

        return test_loaders

    DecOptions = whisper.DecodingOptions(language="en", without_timestamps=True)
    tokenizer = whisper.tokenizer.get_tokenizer(
        True, language="en", task=DecOptions.task
    )

    run_name = f"{experiment_id}_Fold{args.fold_id}_ICL_MultiSource_{args.ds_name}"

    # wandb_logger = WandbLogger(
    #     project="EEG-to-Text",
    #     name=run_name,
    #     log_model=False,
    #     save_dir=str(wd),
    # )

    # use TensorBoard or simple logger
    wandb_logger = TensorBoardLogger(save_dir=str(wd), name="tensorboard_logs")

    def run_tests_and_collect(trainer_obj, model_obj, per_ds_loaders):
        results = {}
        for ds_name, loader in per_ds_loaders:
            model_obj.test_ds_name = ds_name
            res = trainer_obj.test(model_obj, dataloaders=loader, verbose=False)
            print(f"Test Results on {ds_name}: {res}")
            r0 = res[0] if isinstance(res, list) and len(res) > 0 else {}
            metrics = {}
            if "test/accuracy" in r0:
                metrics["accuracy"] = r0["test/accuracy"]
            if "test/roc_auc_macro" in r0:
                metrics["roc_auc_macro"] = r0["test/roc_auc_macro"]
            if "test/pr_auc_macro" in r0:
                metrics["pr_auc_macro"] = r0["test/pr_auc_macro"]
            if "test/roc_auc" in r0:
                metrics["roc_auc"] = r0["test/roc_auc"]
            if "test/pr_auc" in r0:
                metrics["pr_auc"] = r0["test/pr_auc"]
            if "test/kappa" in r0:
                metrics["kappa"] = r0["test/kappa"]
            if "test/f1_weighted" in r0:
                metrics["f1_weighted"] = r0["test/f1_weighted"]
            results[ds_name] = metrics
        return results

    def trainer_init(patience=20, max_epochs=100, gpus=[0]):
        cb = (
            [
                ModelCheckpoint(
                    dirpath=wd / "ckpt",
                    monitor="val_acc",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                    filename="best-{epoch:03d}-{val_acc:.4f}",
                    verbose=True,
                ),
                EarlyStopping(
                    monitor="val_acc", mode="max", patience=patience, verbose=True
                ),
                LearningRateMonitor(logging_interval="step"),
            ]
            if patience > 0
            else [
                ModelCheckpoint(
                    dirpath=wd / "ckpt",
                    monitor="val_acc",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                    filename="best-{epoch:03d}-{val_acc:.4f}",
                    verbose=True,
                ),
                LearningRateMonitor(logging_interval="step"),
            ]
        )
        return Trainer(
            strategy="ddp_find_unused_parameters_true",
            accelerator="gpu",
            devices=gpus,
            max_epochs=max_epochs,
            precision="bf16-mixed",
            enable_progress_bar=False,
            enable_checkpointing=True,
            benchmark=True,
            deterministic=False,
            enable_model_summary=True,
            logger=wandb_logger,
            callbacks=cb,
            check_val_every_n_epoch=1,
        )

    # First Stage
    print("First Stage Training")
    first_stage_k = 1
    first_stage_fixed_support_num = 8

    TrainDS = ICLWithSupportDataset_Lazy(
        udas,
        tokenizer,
        classes=dataset_config["classes"],
        subject_map=train_subject_map,
        k_per_class=first_stage_k,
        max_len=256,
        fixed_support_num=first_stage_fixed_support_num,
        time_len=args.time_len,
    )
    ValDS = ICLWithSupportDataset_Test_Lazy(
        udas,
        tokenizer,
        classes=dataset_config["classes"],
        subject_map=val_subject_map,
        k_per_class=first_stage_k,
        max_len=256,
        fixed_support_num=first_stage_fixed_support_num,
        time_len=args.time_len,
    )

    TrainLoader = torch.utils.data.DataLoader(
        TrainDS,
        batch_size=args.bs,
        collate_fn=WhisperICLCollator(tokenizer, False, args.sot_first, args.supp_loss),
        shuffle=True,
        num_workers=dataloader_num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    ValLoader = torch.utils.data.DataLoader(
        ValDS,
        batch_size=1,
        collate_fn=WhisperICLCollator_Test(tokenizer, args.sot_first),
        shuffle=False,
        num_workers=dataloader_num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    dataset_name_param = ",".join(ds_names)

    # ============================== First Stage Training ==============================
    pl_model = WhisperModelModule(
        len(TrainLoader),
        tokenizer,
        model_name=args.whisper,
        dataset_name=dataset_name_param,
        config=dataset_config,
        wd=wd,
        seed=args.seed,
    )
    # Create Trainer
    train_trainer = trainer_init(patience=10, max_epochs=20, gpus=gpus)
    test_trainer = pl.Trainer(
        strategy="auto",
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        precision="bf16-mixed",
        enable_progress_bar=False,
        enable_checkpointing=False,
        benchmark=True,
        deterministic=False,
        enable_model_summary=False,
        logger=wandb_logger,
    )
    # First Stage Training
    if args.first_stage_ckpt is not None or args.second_stage_ckpt is not None:
        best_model_path = (
            args.first_stage_ckpt
            if args.first_stage_ckpt is not None
            else args.second_stage_ckpt
        )
        print(f"First Stage Model Path: {best_model_path}, skipping training...")
    else:
        train_trainer.fit(
            pl_model, train_dataloaders=TrainLoader, val_dataloaders=ValLoader
        )
        best_model_path = train_trainer.checkpoint_callback.best_model_path
        print(f"Best Model Path: {best_model_path}")

    torch.cuda.empty_cache()
    gc.collect()

    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    safe_load_state_dict(
        pl_model, checkpoint["state_dict"], skip_key_substrings=["fast.last_layer"]
    )
    del checkpoint
    torch.cuda.empty_cache()

    device = torch.device("cuda")
    pl_model = pl_model.to(device)
    pl_model.eval()
    stage1_test_results = {}

    # Test First Stage Model
    if (
        args.first_stage_ckpt is not None and args.first_stage_test
    ) and args.second_stage_ckpt is None:
        per_ds_loaders = build_test_loaders_per_dataset(
            ds_names,
            dataset_subject_ranges,
            all_test_idx,
            udas,
            test_subject_map,
            dataset_config["classes"],
            tokenizer,
            k_per_class=first_stage_k,
            fixed_support_num=first_stage_fixed_support_num,
            sot_first=args.sot_first,
            support_flag=True,
        )
        stage1_test_results = run_tests_and_collect(
            test_trainer, pl_model, per_ds_loaders
        )

    else:
        print("Skip First Stage testing.")

    gc.collect()
    torch.cuda.empty_cache()
    second_stage_k = 1
    second_stage_fixed_support_num = 12
    if second_stage_k == -1:
        print("Second Stage Training Skipped")
        # wandb.finish()
        exit()

    # ============================== Second Stage Training ==============================
    if args.second_stage_ckpt is not None:
        # —— Load Second Stage Model ——
        pl_model.stage = 2
        best_model_path = args.second_stage_ckpt
        print(
            f"Second Stage Model Path: {best_model_path}, skipping second-stage training…"
        )
    else:
        print(f"\nSecond Stage Training")

        train_trainer = trainer_init(patience=10, max_epochs=20, gpus=gpus)
        pl_model.stage = 2
        pl_model.train()

        # -------- Dataset & DataLoader (Stage-2, all Lazy) --------
        if "udas" not in locals():  # single dataset case
            udas = [LazyH5Dataset(ds_names[0], classes=dataset_config["classes"])]

        TrainDS = ICLWithSupportDataset_Lazy(
            udas,
            tokenizer,
            classes=dataset_config["classes"],
            subject_map=train_subject_map,
            max_len=256,
            fixed_support_num=second_stage_fixed_support_num,
            time_len=args.time_len,
        )
        TrainLoader = torch.utils.data.DataLoader(
            TrainDS,
            batch_size=args.bs,
            collate_fn=WhisperICLCollator(
                tokenizer,
                k_is_random=True,
                sot_first=args.sot_first,
                supp_loss=args.supp_loss,
            ),
            shuffle=True,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        ValDS = ICLWithSupportDataset_Test_Lazy(
            udas,
            tokenizer,
            classes=dataset_config["classes"],
            subject_map=val_subject_map,
            k_per_class=second_stage_k,
            max_len=256,
            fixed_support_num=second_stage_fixed_support_num,
            time_len=args.time_len,
        )
        ValLoader = torch.utils.data.DataLoader(
            ValDS,
            batch_size=1,
            collate_fn=WhisperICLCollator_Test(tokenizer, args.sot_first),
            shuffle=False,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        train_trainer.fit(
            pl_model, train_dataloaders=TrainLoader, val_dataloaders=ValLoader
        )
        best_model_path = train_trainer.checkpoint_callback.best_model_path
        print(f"Best Model Path: {best_model_path}")

    torch.cuda.empty_cache()

    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    safe_load_state_dict(
        pl_model, checkpoint["state_dict"], skip_key_substrings=["fast.last_layer"]
    )
    del checkpoint
    torch.cuda.empty_cache()

    device = torch.device("cuda")
    pl_model = pl_model.to(device)
    pl_model.eval()
    stage2_test_results = {}

    print(f"\nTest 1: Using first stage settings:")
    stage2_test_results["first_stage"] = {}
    per_ds_loaders = build_test_loaders_per_dataset(
        ds_names,
        dataset_subject_ranges,
        all_test_idx,
        udas,
        test_subject_map,
        dataset_config["classes"],
        tokenizer,
        k_per_class=0,
        fixed_support_num=first_stage_fixed_support_num,
        sot_first=args.sot_first,
        support_flag=True,
    )
    stage2_test_results["first_stage"] = run_tests_and_collect(
        test_trainer, pl_model, per_ds_loaders
    )

    print(f"\nTest 2: Using second stage settings")
    stage2_test_results["second_stage"] = {}
    per_ds_loaders = build_test_loaders_per_dataset(
        ds_names,
        dataset_subject_ranges,
        all_test_idx,
        udas,
        test_subject_map,
        dataset_config["classes"],
        tokenizer,
        k_per_class=second_stage_k,
        fixed_support_num=second_stage_fixed_support_num,
        sot_first=args.sot_first,
        support_flag=True,
    )
    stage2_test_results["second_stage"] = run_tests_and_collect(
        test_trainer, pl_model, per_ds_loaders
    )

    print(f"\nTest 3: No support")
    stage2_test_results["no_support"] = {}
    per_ds_loaders = build_test_loaders_per_dataset(
        ds_names,
        dataset_subject_ranges,
        all_test_idx,
        udas,
        test_subject_map,
        dataset_config["classes"],
        tokenizer,
        k_per_class=second_stage_k,
        fixed_support_num=0,
        sot_first=args.sot_first,
        support_flag=True,
    )
    stage2_test_results["no_support"] = run_tests_and_collect(
        test_trainer, pl_model, per_ds_loaders
    )

    # ------------------  Cross-dataset Test  ------------------
    cross_ds_names = [n for n in args.cross_ds4test.split(",") if n]
    if len(cross_ds_names) > 0:
        print(f"\nTest 4: Cross-dataset test with first stage settings")
        cross_udas, cross_dataset_subject_ranges, cross_classes = [], [], []
        start_idx = 0
        for ds_name in cross_ds_names:
            try:
                cfg = find_config_by_name(ds_name)
            except Exception as e:
                print(f"Error loading config for {ds_name}: {e}")
                cfg = Split_Config
                cfg["name"] = ds_name
            uda_lazy = LazyH5Dataset(ds_name, classes=cfg["classes"])
            cross_udas.append(uda_lazy)
            cross_classes.extend(cfg["classes"])
            cross_dataset_subject_ranges.append(
                (start_idx, start_idx + uda_lazy.n_subjects)
            )
            start_idx += uda_lazy.n_subjects

        all_cross_test_idx = []
        for name, (s, e), uda in zip(
            cross_ds_names, cross_dataset_subject_ranges, cross_udas
        ):
            local_n = uda.n_subjects
            local_split = _safe_get_split(name, local_n)
            all_cross_test_idx.extend([s + i for i in local_split["test"]])

        cross_test_subject_map = build_subject_map(
            all_cross_test_idx, cross_dataset_subject_ranges, cross_udas
        )

        stage2_test_results["cross_ds_first_stage"] = {}
        per_ds_cross_loaders = build_test_loaders_per_dataset(
            cross_ds_names,
            cross_dataset_subject_ranges,
            all_cross_test_idx,
            cross_udas,
            cross_test_subject_map,
            cross_classes,
            tokenizer,
            k_per_class=second_stage_k,
            fixed_support_num=second_stage_fixed_support_num,
            sot_first=args.sot_first,
            support_flag=True,
        )
        stage2_test_results["cross_ds_first_stage"] = run_tests_and_collect(
            test_trainer, pl_model, per_ds_cross_loaders
        )

        print(f"\nTest 5: Cross-dataset test with second stage settings")

        stage2_test_results["cross_ds_second_stage"] = {}
        per_ds_cross_loaders = build_test_loaders_per_dataset(
            cross_ds_names,
            cross_dataset_subject_ranges,
            all_cross_test_idx,
            cross_udas,
            cross_test_subject_map,
            cross_classes,
            tokenizer,
            k_per_class=second_stage_k,
            fixed_support_num=second_stage_fixed_support_num,
            sot_first=args.sot_first,
            support_flag=True,
        )
        stage2_test_results["cross_ds_second_stage"] = run_tests_and_collect(
            test_trainer, pl_model, per_ds_cross_loaders
        )

        print(f"\nTest 6: Cross-dataset test with no support")

        stage2_test_results["cross_ds_no_support"] = {}
        per_ds_cross_loaders = build_test_loaders_per_dataset(
            cross_ds_names,
            cross_dataset_subject_ranges,
            all_cross_test_idx,
            cross_udas,
            cross_test_subject_map,
            cross_classes,
            tokenizer,
            k_per_class=second_stage_k,
            fixed_support_num=0,
            sot_first=args.sot_first,
            support_flag=False,
        )
        stage2_test_results["cross_ds_no_support"] = run_tests_and_collect(
            test_trainer, pl_model, per_ds_cross_loaders
        )

    else:
        print("No cross datasets specified, skip Test 4.")

    all_test_results = {
        "stage1_test_results": stage1_test_results if stage1_test_results else {},
        "stage2_test_results": stage2_test_results,
    }

    with open(wd / "test_results.pkl", "wb") as f:
        pickle.dump(all_test_results, f)

    print(f"Test results: {all_test_results}")
    # wandb.finish()
