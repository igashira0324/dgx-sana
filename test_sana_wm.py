#!/usr/bin/env python3
import sys

print("=== SANA-WM DGX SPARK Validation Script ===")

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch is not installed or configured correctly.")
    sys.exit(1)

try:
    import sana
    print(f"SANA successfully imported. Version: {sana.__version__ if hasattr(sana, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"Failed to import SANA: {e}")
    sys.exit(1)

# Try to import typical diffusion modules or video modules to ensure they are available
try:
    import diffusion
    print("Diffusion module successfully imported.")
except ImportError as e:
    print(f"Failed to import diffusion module: {e}")

print("=== Validation Complete ===")
print("Please run inference_video_scripts/inference_sana_video.sh for actual generation.")
