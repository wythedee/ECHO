export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS="EMO_03_SEED_V,EMO_06_FACED,SLEEP_05_isruc_S1,CS_04_BCIC_Track3,MDD_01_Mumtaz,MI_13_SHU,EP_01_CHBMIT"
# export DS="MI_04_BCI_IV_2a,MI_06_Schirrmeister2017,MI_10_HeBin2021_LR,MI_10_HeBin2021_UD,EMO_02_SEED_IV,EMO_03_SEED_V,EMO_04_SEED"
export DS="SLEEP_05_isruc_S1,CS_04_BCIC_Track3,MDD_01_Mumtaz,EP_01_CHBMIT,MI_04_BCI_IV_2a,MI_06_Schirrmeister2017,MI_10_HeBin2021_LR,MI_10_HeBin2021_UD,MI_13_SHU,EMO_02_SEED_IV,EMO_03_SEED_V,EMO_04_SEED,EMO_06_FACED"
python -u train_multisource_split.py \
    --ds $DS \
    --time_length 30 \
    --gpus 0,1,2,3 \
    --head V3 \
    --dim1 256 \
    --dim2 256 \
    --win 100 \
    --step 90 \
    --lay 4 \
    --lr 5e-05 \
    --bs 64 \
    --workers 64 \
    --persistent_workers \
    --encoder_dir "/path/to/your/ECHO/eeg_encoders/$DS" \
