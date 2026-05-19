#!/usr/bin/env bash
# Modified for DGX Spark ARM64 by igashira0324 on 2026-05-19
# - Added venv initialization, manual build steps for mmcv, and compilation flags/warnings for xformers
set -e

echo "=== Starting SANA-WM Environment Setup for DGX SPARK ==="

# Initialize venv
ENV_NAME="sana_env"

if [ -d "$ENV_NAME" ]; then
    echo "Python environment $ENV_NAME already exists. Skipping creation."
else
    echo "Creating Python environment $ENV_NAME with python3..."
    python3 -m venv $ENV_NAME
fi

echo "Activating $ENV_NAME..."
source $ENV_NAME/bin/activate

# Install pip and basic build tools
pip install -U pip setuptools wheel ninja

# Install PyTorch
# In the DGX SPARK ARM64 environment, a compatible PyTorch (2.12.0+cu130) is natively resolved from PyPI.
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install sana from local directory
echo "Installing SANA..."
pip install -e .

# Install xformers and flash-attn
# NOTE: xformers 0.0.32.post2 does not have a pre-built wheel for ARM64.
# It will be compiled from source. This compilation can take 15 to 30 minutes!
# We set MAX_JOBS to speed up the compilation using multiple cores.
export MAX_JOBS=16
echo "Installing xformers..."
pip install xformers==0.0.32.post2

echo "Attempting to install flash-attn..."
pip install flash-attn==2.8.2 --no-build-isolation || echo "WARNING: flash-attn installation failed. SANA will automatically fallback to xformers memory_efficient_attention or PyTorch scaled_dot_product_attention (SDPA)."

echo "=== Setup Complete ==="
