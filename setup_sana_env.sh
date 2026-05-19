#!/usr/bin/env bash
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

# Install PyTorch (DGX SPARK usually requires ARM64 wheels, standard cu124 or cu128 if available, or just default which pip resolves)
# We use the default PyTorch pip installation for ARM64 if it has it, otherwise rely on system's ability to pull the right wheel.
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || pip install torch torchvision torchaudio

# Install sana from local directory
echo "Installing SANA..."
pip install -e .

# Install xformers and flash-attn (may require compilation on ARM64)
export MAX_JOBS=16 # Speed up compilation if needed
echo "Installing xformers and flash-attn..."
# Try pre-built first, fallback to source build if needed.
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128 || pip install xformers
pip install flash-attn==2.8.2 --no-build-isolation || echo "WARNING: flash-attn installation failed. This might need manual building or waiting for ARM64 wheels."

echo "=== Setup Complete ==="
