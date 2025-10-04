export CKPT="/path/to/your/ckpt.ckpt"
export DS="EMO_03_SEED_V"
python -u MultiSource_EEG2Text_Split.py \
	--ds_name $DS \
	--gpus "0,1" \
	--time_len 10 \
	--enc_version "V3" \
	--whisper "tiny" \
	--bs 48 \
	--first_stage_ckpt $CKPT \
	--finetune True
	
export DS="MI_12_PhysioNet"
python -u MultiSource_EEG2Text_Split.py \
	--ds_name $DS \
	--gpus "0,1" \
	--time_len 10 \
	--enc_version "V3" \
	--whisper "tiny" \
	--bs 48 \
	--first_stage_ckpt $CKPT \
	--finetune True

