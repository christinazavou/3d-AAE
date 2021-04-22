#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=titanx-long    # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=10000   # Memory in MB per cpu allocated

TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export PYTHONUNBUFFERED="True"

PY_EXE=/home/maverkiou/miniconda2/envs/style_detect_env/bin/python
SOURCE_DIR=/home/maverkiou/zavou/3d-AAE

log_file=$TIME.log

VERSION=$(git rev-parse HEAD)

echo Logging output to "$log_file"
echo "Version: ${VERSION}" > "$log_file"
echo -e "GPU(s): $CUDA_VISIBLE_DEVICES" >> $log_file
export PYTHONPATH=$PYTHONPATH:${SOURCE_DIR}:${SOURCE_DIR}/experiments
cd ${SOURCE_DIR}/experiments

args="--config ${SOURCE_DIR}/settings/hyperparams_annfass_gypsum.json"
echo "${PY_EXE} train_aae.py $args" >> "$log_file"
${PY_EXE} train_aae.py $args 2>&1 | tee -a "$log_file"
