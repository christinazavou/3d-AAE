
train-ae-content-style:
	export MAIN_FILE=train_autoencoder_content_style.py \
		&& export CONFIG_FILE=hyperparams_ae_content_style_gypsum.json \
		&& sbatch train.sh
