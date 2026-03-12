[English](README.md) | 中文

# "ECHO: Toward Contextual Seq2seq Paradigms in Large EEG Models" 官方仓库 [![arXiv](https://img.shields.io/badge/arXiv-2509.22556-b31b1b.svg)](https://arxiv.org/abs/2509.22556)

本仓库提供 ECHO 的官方实现。当前发布版本包含两个阶段：
- `FAST/`：EEG 编码器 warm-up
- `EEG2Text/`：EEG 到文本的上下文预训练

## News
- 2026 年 3 月：ECHO 被 [ICLR 2026](https://openreview.net/forum?id=ClLQ6cLkoR) 接收。🎉

## 本次发布范围
- 将原始数据预处理为标准化 `.h5`
- FAST 编码器预训练
- ECHO / ECHO-long 的 EEG-to-text 预训练

## 最小入口脚本
主流程使用以下 3 个 shell 脚本：
- `FAST/EEG_Dataset/preprocess_selected.sh`
- `FAST/train_multisource_split.sh`
- `EEG2Text/multisource_icl.sh`

## 环境安装
建议使用两个 Conda 环境。

```bash
conda create -n echo-encoder python=3.12
conda create -n echo-decoder python=3.12

conda activate echo-encoder
pip install -r requirements_encoder.txt

conda activate echo-decoder
pip install -r requirements_decoder.txt
```

## 路径
```bash
export FAST_EEG_SOURCE_ROOT=/path/to/raw-datasets
export ECHO_DATASET_ROOT=/path/to/EEG_Standardized_Group
export FAST_EEG_OUTPUT="$ECHO_DATASET_ROOT"
```

- `FAST_EEG_SOURCE_ROOT`：下载后的原始数据目录
- `ECHO_DATASET_ROOT`：训练使用的标准化 `.h5` 目录
- `FAST_EEG_OUTPUT`：预处理输出目录

## 数据准备
预处理脚本位于 `FAST/EEG_Dataset/`。

主预训练 profile：

```bash
cd FAST/EEG_Dataset
FAST_EEG_SOURCE_ROOT=/path/to/raw-datasets \
ECHO_DATASET_ROOT=/path/to/EEG_Standardized_Group \
FAST_EEG_OUTPUT=/path/to/EEG_Standardized_Group \
PREPROCESS_PROFILE=main \
bash preprocess_selected.sh
```

可选睡眠任务（ISRUC）profile：

```bash
cd FAST/EEG_Dataset
FAST_EEG_SOURCE_ROOT=/path/to/raw-datasets \
ECHO_DATASET_ROOT=/path/to/EEG_Standardized_Group \
FAST_EEG_OUTPUT=/path/to/EEG_Standardized_Group \
PREPROCESS_PROFILE=isruc \
bash preprocess_selected.sh
```

你也可以手动传入数据集脚本：

```bash
cd FAST/EEG_Dataset
FAST_EEG_SOURCE_ROOT=/path/to/raw-datasets \
FAST_EEG_OUTPUT=/path/to/EEG_Standardized_Group \
bash preprocess_selected.sh MI_10_HeBin2021.py EMO_03_SEED_V.py
```

`main` 包含：
- MI：`MI_01_KoreaU`、`MI_03_Shin2017A`、`MI_04_BCI_IV_2a`、`MI_05_Weibo2014`、`MI_06_Schirrmeister2017`、`MI_07_Cho2017`、`MI_09_Track4_Upper_limb`、`MI_10_HeBin2021_LR`、`MI_10_HeBin2021_UD`、`MI_11_HeBin2024_LR`、`MI_11_HeBin2024_UD`、`MI_12_PhysioNet`
- EMO：`EMO_02_SEED_IV`、`EMO_03_SEED_V`、`EMO_04_SEED`、`EMO_05_THU-EP`

`isruc` 包含：
- sleep：`SLEEP_05_isruc_S1`、`SLEEP_05_isruc_S3`

## 第 1 步：FAST 编码器 Warm-up
使用相同的标准化数据目录运行 FAST 预训练。

主 profile：

```bash
cd FAST
ECHO_DATASET_ROOT=/path/to/EEG_Standardized_Group \
PRETRAIN_PROFILE=main \
bash train_multisource_split.sh
```

可选 ISRUC profile：

```bash
cd FAST
ECHO_DATASET_ROOT=/path/to/EEG_Standardized_Group \
PRETRAIN_PROFILE=isruc \
bash train_multisource_split.sh
```

## 第 2 步：ECHO EEG-to-Text 预训练
继续使用相同的数据目录，并确保 `TIME_LEN` 与 FAST 阶段一致。

主 profile：

```bash
cd EEG2Text
ECHO_DATASET_ROOT=/path/to/EEG_Standardized_Group \
PRETRAIN_PROFILE=main \
bash multisource_icl.sh
```

可选 ISRUC profile：

```bash
cd EEG2Text
ECHO_DATASET_ROOT=/path/to/EEG_Standardized_Group \
PRETRAIN_PROFILE=isruc \
bash multisource_icl.sh
```

## 本次发布中的主要文件
- `FAST/`：编码器训练与预处理
- `EEG2Text/`：decoder 侧上下文预训练
- `requirements_encoder.txt`：encoder 依赖
- `requirements_decoder.txt`：decoder 依赖

如果你还想进一步裁剪或自定义数据集，可以修改 `FAST/EEG_Dataset/`、`FAST/dataset_split_config.py`、`EEG2Text/EEG_dataset_config.py` 和 `EEG2Text/dataset_split_config.py`。
