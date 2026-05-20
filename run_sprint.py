# Modified for DGX Spark ARM64 by igashira0324 on 2026-05-19
# - Added warm-up execution and median latency benchmark for SANA-Sprint (1-step)

import torch
import time
from diffusers import SanaSprintPipeline

# 1. パイプラインのロード（bf16でメモリ節約）
pipe = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# bf16でVAE・テキストエンコーダも統一
pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

# 2. プロンプト
prompt = "a futuristic Tokyo street at night, neon lights, cyberpunk style, ultra detailed"

# 3. ウォームアップ（初回ロード）
print("Warmup (Initial run)...")
_ = pipe(prompt=prompt, num_inference_steps=2, height=1024, width=1024)

# 4. 本計測（3回計測の中央値で評価）
times = []
for i in range(3):
    start = time.time()
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=2,     # SANA-Sprint 2-step (SCM requires exactly 2 steps)
        guidance_scale=4.5,
        generator=torch.Generator(device="cuda").manual_seed(42 + i),
    ).images[0]
    elapsed = time.time() - start
    times.append(elapsed)
    image.save(f"output_sprint_{i}.png")
    print(f"  Run {i+1}: {elapsed:.2f} s")

times.sort()
print(f"\n✅ Median (2-step): {times[1]:.2f} s (DGX Spark)")
print(f"   Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

