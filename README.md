# The official Repository of "ECHO: Toward Contextual Seq2seq Paradigms in Large EEG Models"

This repository contains the official implementation of ECHO, a framework designed for contextual sequence-to-sequence paradigms in large EEG models. ECHO comprises two main components: an **EEG Encoder** (implemented in the `FAST` directory) and an **EEG-to-Text Decoder** (implemented in the `EEG2Text` directory). The encoder processes raw EEG signals, while the decoder generates textual representations based on the encoded EEG features.

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
## Run Pretraining
The pretraining phase involves warming up the EEG Encoder and then contextually pretraining the EEG-to-Text Decoder.

### Dataset Preparation
To ensure proper integration and functionality with ECHO, follow the recommended dataset directory structure and preparation steps.

#### Dataset Directory Structure
We recommend the following tree map for your `DATASET_SOURCE_ROOT` to ensure seamless integration with ECHO's preprocessing scripts:

```
DATASET_SOURCE_ROOT
  --EMO
    --EMO_01_{dataset_name}...
  --MI
    --MI_{dataset_name} # Note: MI dataset source file directories under 'MI' should not contain numbers (e.g., 'MI_01_KoreaU') if you intend to use the existing preprocessing files without modification.
    --MI_...
  --STR
    --STR_01_{dataset_name}
  --...
```
You can refer to the `FAST/EEG_Dataset` directory for examples of dataset names, which are used in preprocessing files or configured within them. These preprocessing files are responsible for converting raw EEG data into a standardized `.h5` format.

#### Adding New Datasets
To incorporate additional datasets for training, follow these precise steps:
1.  **Name Convention**: Assign a unique name in the format `TASK_0x_DATASET_NAME` (e.g., `MI_01_KoreaU`).
2.  **Preprocessing File**: Create a dedicated preprocessing Python file for your dataset within `FAST/EEG_Dataset/`. This file must define a `META` instance (e.g., `MI_HeBin2021_LR = META(NAME_LR, CH_NAMES, SUBJECTS, ['MI/Left', 'MI/Right'], resample_rate=250, time_length=5)}`). This `META` instance encapsulates essential metadata for your dataset.
3.  **Integrate META Instance**: Update `FAST/EEG_Dataset/__init__.py` to import and include your newly defined `META` instance.
4.  **Configure Subject Split**: Modify `FAST/dataset_split_config.py` (for the encoder) and `EEG2Text/dataset_split_config.py` (for the decoder) to define the subject split (train/validation/test) for your dataset. The configuration variable name should be `{dataset_name(with number)}_split`.
5.  **Configure Decoder Dataset Info**: Update `EEG2Text/EEG_dataset_config.py` to include basic information about your dataset. This file acts as a `META` class for the decoder, so ensure its configuration aligns with the `META` instance defined in the `FAST` directory.

All preprocessing files are expected to output an **HDF5 (.h5) file**. Within this `.h5` file, each subject should serve as a key, with the corresponding EEG data as its value.

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
