
train-ae-content-style:
	export MAIN_FILE=train_autoencoder_content_style.py \
		&& export CONFIG_FILE=hyperparams_ae_content_style_gypsum.json \
		&& sbatch train.sh


train-ae-annfass:
	export MAIN_FILE=train_autoencoder.py \
		&& export CONFIG_FILE=annfass/ae/gypsum/hyperparams.json \
		&& sbatch train.sh
