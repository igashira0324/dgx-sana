# Modified for DGX Spark ARM64 by igashira0324 on 2026-05-19
# - Added warm-up execution and median latency benchmark for SANA 1.6B

import torch
import time
from diffusers import SanaPipeline

# 1. パイプラインロード
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# bf16でVAE・テキストエンコーダも統一
pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

# 2. プロンプト
prompt = 'a cyberpunk cat with a neon sign that says "SANA on DGX Spark"'

# 3. ウォームアップ（初回はJITコンマイル等の初期化処理が入るため）
print("Warmup (Initial run)...")
_ = pipe(prompt=prompt, num_inference_steps=20, height=1024, width=1024)

# 4. 本計測（3回計測の中央値で評価）
times = []
for i in range(3):
    start = time.time()
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(42 + i),
    ).images[0]
    elapsed = time.time() - start
    times.append(elapsed)
    image.save(f"output_sana_{i}.png")
    print(f"  Run {i+1}: {elapsed:.2f} s")

times.sort()
print(f"\n✅ Median: {times[1]:.2f} s (DGX Spark)")
print(f"   Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
