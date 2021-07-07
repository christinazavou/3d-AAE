PARTITION ?= titanx-long

train-ae-content-style:
	export MAIN_FILE=train_autoencoder_content_style.py \
		&& export CONFIG_FILE=annfass/ae_content_style/gypsum/hyperparams.json \
		&& sbatch train.sh

train-ae-content-style-extraloss:
	export MAIN_FILE=train_autoencoder_content_style_extraloss.py \
		&& export CONFIG_FILE=annfass/ae_content_style_extraloss/gypsum/hyperparams.json \
		&& sbatch train.sh


train-ae-annfass:
	export MAIN_FILE=train_autoencoder.py \
		&& export CONFIG_FILE=annfass/ae/gypsum/hyperparams.json \
		&& sbatch train.sh


train-ae-buildnet:
	export MAIN_FILE=train_autoencoder.py \
		&& export CONFIG_FILE=buildnet/ae/gypsum/hyperparams.json \
		&& sbatch --job-name=3dae_bc --partition=$(PARTITION) train.sh

train-ae-content-style-extraloss-buildnet:
	export MAIN_FILE=train_autoencoder_content_style_extraloss.py \
		&& export CONFIG_FILE=buildnet/ae_content_style_extraloss/gypsum/hyperparams.json \
		&& sbatch train.sh

export-ae-on-annfass:
	cd evaluation \
 		&& /home/graphicslab/miniconda3/envs/py3-mink/bin/python export_encodings.py \
			--config ../settings/buildnet/ae/export/hyperparams.json \

