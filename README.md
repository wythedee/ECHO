# The official Repository of "ECHO: Toward Contextual Seq2seq Paradigms in Large EEG Models"

This repository contains the official implementation of ECHO, a framework designed for contextual sequence-to-sequence paradigms in large EEG models. ECHO comprises two main components: an **EEG Encoder** (implemented in the `FAST` directory) and an **EEG-to-Text Decoder** (implemented in the `EEG2Text` directory). The encoder consumes standardized EEG tensors, while the decoder generates textual representations based on the encoded EEG features.

<p align="center">
  <img src="images/main.png" alt="ECHO framework overview" width="900">
  <br>
  <em>Main figure of the ECHO framework. View the high-resolution PDF <a href="images/main.pdf">here</a>.</em>
  
</p>

## Setup
We recommend setting up two separate Conda environments for the encoder and decoder to manage their respective dependencies.

```bash
conda create -n encoder python=3.12
conda create -n decoder python=3.12

conda activate encoder
pip install -r requirements_encoder.txt

conda activate decoder
pip install -r requirements_decoder.txt
```
Configure all paths in the repo using global search function in your IDE. Search for keywords `/path/to/your` to find the position.

## Checkpoint
The released ECHO checkpoint is available on Hugging Face:

```
https://huggingface.co/wythedee/ECHO
```

Download it into the repository root with:

```bash
pip install -U huggingface_hub
huggingface-cli download wythedee/ECHO \
  --repo-type model \
  --include "checkpoints/ECHO.ckpt" \
  --local-dir .
```

## Data Format
This public release uses standardized HDF5 files as the dataset interface. The conversion from source recordings to this format is intentionally not included. To run the framework, prepare one `.h5` file per dataset with the following layout:

```
{DATASET_ROOT}/{TASK_PREFIX}/{DATASET_NAME}.h5
```

For example, `MI_04_BCI_IV_2a` is loaded from:

```
{DATASET_ROOT}/MI/MI_04_BCI_IV_2a.h5
```

Each HDF5 file should contain one group per subject:

```
MI_04_BCI_IV_2a.h5
  subject_001/
    X    float32, shape = (n_trials, n_channels, n_samples)
    Y    int,     shape = (n_trials,)
  subject_002/
    X
    Y
  ...
```

Requirements:
1.  `X` should be trial-first EEG tensors.
2.  `Y` should contain zero-based integer class labels.
3.  The channel dimension should match the channel list used by the config. The released config uses generic placeholders `CH001...CH075`; replace them locally if your HDF5 files use a different channel count or grouping.
4.  `n_samples` can vary across datasets. Training scripts pad or crop trials according to `--time_length`.

Dataset metadata is configured in two places:
1.  `FAST/dataset_metadata.py`: dataset names, classes, and encoder channel metadata.
2.  `EEG2Text/EEG_dataset_config.py`: decoder-side dataset config, sequence length, classes, and channel metadata.

Subject splits are configured in:
1.  `FAST/dataset_split_config.py`
2.  `EEG2Text/dataset_split_config.py`

## Run Pretraining
The pretraining phase involves warming up the EEG Encoder and then contextually pretraining the EEG-to-Text Decoder.

### Dataset Preparation
This public release expects datasets that have already been converted into the standardized HDF5 format described above.

To incorporate additional datasets for training:
1.  Add the `.h5` file under `{DATASET_ROOT}/{TASK_PREFIX}/`.
2.  Add or update the dataset class list in `FAST/dataset_metadata.py`.
3.  Add or update the decoder config in `EEG2Text/EEG_dataset_config.py`.
4.  Configure subject splits in the split config files.

### Quick Run Checklist
1.  Replace `/path/to/your/dataset_root` in `FAST/train_multisource_split.py` with your dataset root.
2.  Replace `/path/to/your/datasets_root` in `EEG2Text/lazy_dataset.py` and `EEG2Text/EEG_dataset_config.py` with the same root.
3.  Edit `FAST/train_multisource_split.sh` and `EEG2Text/multisource_icl.sh` to choose dataset names, GPUs, batch size, and time length.
4.  Make sure every dataset name in `--ds` or `--ds_name` has a matching `.h5` file and metadata entry.

#### Datasets for Pretraining
For a comprehensive list of all datasets utilized in the ECHO pretraining phase, including their specific sources and download instructions, please refer to our paper.

### Pretrain
#### 1. EEG Encoder Warm-up
This step involves pretraining the EEG Encoder, implemented within the `FAST` directory. The `train_multisource_split.py` script is used for this purpose, configured via `train_multisource_split.sh`.

Adjust the settings in `FAST/train_multisource_split.sh` as needed (e.g., specifying datasets, GPU usage, `time_length`).
```bash
cd FAST
bash train_multisource_split.sh
```

#### 2. EEG-to-Text Decoder Contextual Pretraining
After the encoder warm-up, the EEG-to-Text Decoder (located in `EEG2Text`) is contextually pretrained. This phase leverages the pre-trained encoder to learn to generate text from EEG features. The `MultiSource_EEG2Text_Split.py` script handles this pretraining, configured by `multisource_icl.sh`.

Ensure that the `time_len` variable in `ECHO/EEG2Text/multisource_icl.sh` is set to the same value as the `time_length` used during the encoder warm-up.
```bash
cd EEG2Text
bash multisource_icl.sh
```

## Finetune
To finetune the pre-trained ECHO model on specific downstream tasks, adjust the `EEG2Text/finetune.sh` script. Set the `$CKPT` variable to the path of your pre-trained checkpoint and `$DS` to the datasets you wish to finetune on.
```bash
cd EEG2Text
bash finetune.sh
```
