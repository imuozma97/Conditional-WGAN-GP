#!/bin/bash 
echo ">>>>> ACTIVATE VIRTUALENV TensorFlow 2.3.1"

ENVIRONMENT="cuda-env"
eval "$(conda shell.bash hook)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENVIRONMENT

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export TF_XLA_FLAGS=--tf_xla_auto_jit=0
cd "$(dirname "$0")"
~/miniconda3/envs/cuda-env/bin/python3 -u -m Main.Entrenamiento0.py
