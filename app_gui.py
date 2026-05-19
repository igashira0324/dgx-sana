# Modified for DGX Spark ARM64 by igashira0324 on 2026-05-19
# - Added Gradio-based Web UI for interactive SANA image generation

import gradio as gr
import torch
import time
from diffusers import SanaPipeline

print("Loading SANA Pipeline...")
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)
print("SANA Pipeline loaded successfully!")

def generate_image(prompt, height, width, steps, guidance_scale, seed):
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    
    start_time = time.time()
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images[0]
    elapsed = time.time() - start_time
    
    return image, f"生成完了！ 所要時間: {elapsed:.2f} 秒"

# Gradio インターフェースの構築
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 SANA-1.6B Web UI on DGX Spark (ARM64)
        NVIDIA Blackwell GB10 を搭載した DGX Spark 向けに最適化された超高速画像生成 GUI です。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="プロンプト (英語推奨)",
                placeholder="a cyberpunk cat with a neon sign that says 'SANA'...",
                lines=3
            )
            
            with gr.Row():
                height = gr.Dropdown(
                    choices=[512, 768, 1024],
                    value=1024,
                    label="高さ (Height)"
                )
                width = gr.Dropdown(
                    choices=[512, 768, 1024],
                    value=1024,
                    label="幅 (Width)"
                )
                
            steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=20,
                step=1,
                label="推論ステップ数 (Steps)"
            )
            
            guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=4.5,
                step=0.5,
                label="Guidance Scale"
            )
            
            seed = gr.Number(
                value=42,
                label="シード値 (Seed)",
                precision=0
            )
            
            btn = gr.Button("🎨 画像を生成する", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="生成された画像")
            output_status = gr.Textbox(label="ステータス", interactive=False)
            
    btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, steps, guidance_scale, seed],
        outputs=[output_image, output_status]
    )

if __name__ == "__main__":
    # 社内プロキシ (i-FILTER等) の誤作動・ブロックを回避するため、127.0.0.1 で安全に起動
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
