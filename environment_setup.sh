#!/usr/bin/env bash
# ============================================================
# WARNING: This script is inherited from upstream NVlabs/Sana
# and is NOT compatible with DGX Spark (ARM64 + CUDA 13.0).
# 
# For DGX Spark, please use:  bash setup_sana_env.sh
# ============================================================
echo "[WARNING] This script targets x86_64 + CUDA 12.x + conda environments."
echo "[WARNING] DGX Spark (ARM64 + CUDA 13.0) users should run 'bash setup_sana_env.sh' instead."
read -p "Continue anyway? (y/N): " ans
[[ "$ans" != "y" && "$ans" != "Y" ]] && exit 1

set -e

# Check if we should skip environment setup entirely
if [ "${SKIP_ENV_SETUP}" = "true" ]; then
    echo "SKIP_ENV_SETUP is set to true. Skipping all environment setup steps."
    echo "Using default conda environment. Make sure it has all required packages installed."
    exit 0
fi

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit=12.8 -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# init a raw torch to avoid installation errors.
# pip install torch

# update pip to latest version for pyproject.toml setup.
pip install -U pip

# for fast attn
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128

# install sana
pip install -e .

pip install flash-attn==2.8.2 --no-build-isolation

# install torchprofile
# pip install git+https://github.com/zhijian-liu/torchprofile
