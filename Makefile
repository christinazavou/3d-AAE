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

THIS_DIR := /media/graphicslab/BigData/zavou/ANNFASS_CODE/3d-AAE
export-ae-on-annfass:
	export PYTHONPATH=$$PYTHONPATH:$(THIS_DIR):$(THIS_DIR)/evaluation \
		&& cd evaluation \
 		&& /home/graphicslab/miniconda3/envs/py3-mink/bin/python export_encodings.py \
			--config ../settings/buildnet/ae/export/hyperparams_annfass.json
export-ae-on-buildnettest:
	export PYTHONPATH=$$PYTHONPATH:$(THIS_DIR):$(THIS_DIR)/evaluation \
		&& cd evaluation \
 		&& /home/graphicslab/miniconda3/envs/py3-mink/bin/python export_encodings.py \
			--config ../settings/buildnet/ae/export/hyperparams_buildnet.json

train-aae-buildnet:
	export MAIN_FILE=train_aae.py \
		&& export CONFIG=buildnet/aae/turing/hyperparams.json \
		&& GPU=0 \
		&& sbatch --job-name=3daaeNa -w gpu-0-1 --mem-per-cpu=90000 train_turing.sh
train-aae-buildnet1:
	export MAIN_FILE=train_aae.py \
		&& export CONFIG=buildnet/aae/turing/hyperparams1.json \
		&& GPU=0 \
		&& sbatch --job-name=3daaeNa1 -w gpu-0-1 --mem-per-cpu=90000 train_turing.sh
train-aae-buildnet6:
	export MAIN_FILE=train_aae.py \
		&& export CONFIG=buildnet/aae/turing/hyperparams6.json \
		&& GPU=0 \
		&& sbatch --job-name=3daaeNa6 -w gpu-0-1 --mem-per-cpu=90000 --cpus-per-task=6 train_turing.sh
