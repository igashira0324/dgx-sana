# Modified for DGX Spark ARM64 by igashira0324 on 2026-05-19
# - Added quick SANA-Sprint image generation script for real-time latency evaluation

import torch
from diffusers import SanaSprintPipeline

# 1. パイプラインのロード（bf16でメモリ節約）
pipe = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# 2. プロンプト
prompt = "a futuristic Tokyo street at night, neon lights, cyberpunk style, ultra detailed"

# 3. 画像生成（1ステップ！）
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=2,     # 1〜4が推奨。多くても4
    guidance_scale=4.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

# 4. 保存
image.save("output_sprint.png")
print("✅ Saved: output_sprint.png")
