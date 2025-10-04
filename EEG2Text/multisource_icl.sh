#!/bin/bash
# Example: Motor Imagery and Emotion Recognition datasets for pretraining
export MI_DS="MI_01_KoreaU,MI_03_Shin2017A,MI_04_BCI_IV_2a,MI_05_Weibo2014,MI_06_Schirrmeister2017,MI_07_Cho2017,MI_09_Track4_Upper_limb,MI_10_HeBin2021_LR,MI_10_HeBin2021_UD,MI_11_HeBin2024_LR,MI_11_HeBin2024_UD,MI_12_PhysioNet"
export EMO_DS="EMO_02_SEED_IV,EMO_03_SEED_V,EMO_04_SEED,EMO_05_THU-EP"
export DS="$MI_DS,$EMO_DS"
echo "Whisper start training..."

python -u MultiSource_EEG2Text_Split.py \
	--ds_name $DS \
	--gpus "0,1,2,3" \
	--time_len 10 \
	--enc_version "V3" \
	--whisper "tiny" \
	--bs 64 \
	--cross_ds4test "MI_08_Track1_Few_shot"
