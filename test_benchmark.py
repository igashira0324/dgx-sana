# Modified for DGX Spark ARM64 by igashira0324 on 2026-05-19
# - Added benchmark logging for latency, peak VRAM/Unified Memory, and result saving

import os
import time
import torch
from torchvision.utils import save_image
from app.sana_pipeline import SanaPipeline

def run_benchmark():
    print("=== SANA-WM Image Generation Benchmark on DGX SPARK ===")
    
    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()
    
    print("Loading SanaPipeline (1.6B)...")
    t0 = time.time()
    
    # Initialize pipeline
    pipeline = SanaPipeline("configs/sana_config/1024ms/Sana_1600M_img1024.yaml")
    
    # Load pretrained weights
    pipeline.from_pretrained("hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth")
    
    loading_time = time.time() - t0
    print(f"Pipeline loaded successfully in {loading_time:.2f} seconds.")
    
    prompt = "A cyberpunk cat with a neon sign that says 'Sana'."
    print(f"Generating image for prompt: '{prompt}'...")
    
    # Start inference timer
    t1 = time.time()
    
    # Execute inference
    generator = torch.Generator(device="cuda").manual_seed(42)
    output = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=20,
        generator=generator
    )
    
    generation_time = time.time() - t1
    print(f"Generation complete!")
    print(f"Time taken for image generation: {generation_time:.2f} seconds.")
    
    # Get peak memory allocated
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak VRAM (or Unified Memory) allocated: {peak_vram:.2f} GB")
    
    # Save the output image
    os.makedirs("output_benchmark", exist_ok=True)
    save_path = "output_benchmark/cyberpunk_cat.jpg"
    save_image(output, save_path, normalize=True, value_range=(-1, 1))
    print(f"Generated image saved at: {save_path}")
    print("=======================================================")

if __name__ == "__main__":
    run_benchmark()
