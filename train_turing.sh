#!/bin/bash
#SBATCH -J smi
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=150000

SOURCE_DIR=${SOURCE_DIR:-/home/czavou01/3d-AAE}
PY_EXE=${PY_EXE:-/home/czavou01/miniconda3/envs/decorgan/bin/python}
CONFIG=${CONFIG:-/home/czavou01/3d-AAE/settings/buildnet/aae/turing/hyperparams.json}
GPU=${GPU:-0}
MAIN_FILE=${MAIN_FILE:-train_aae}

export CUDA_VISIBLE_DEVICES=${GPU}

echo "start ${PY_EXE} ${SOURCE_DIR}/experiments/${MAIN_FILE}.py --config ${CONFIG} --gpu ${GPU}"
${PY_EXE} ${SOURCE_DIR}/experiments/${MAIN_FILE}.py --config ${CONFIG} --gpu ${GPU}
