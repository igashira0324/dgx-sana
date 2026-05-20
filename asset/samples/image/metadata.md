# SANA / SANA-Sprint Image Sample Metadata

This file documents the generation parameters and prompts for the image samples generated on the **NVIDIA DGX Spark** platform.

---

## 💻 System Configuration

* **Hardware**: NVIDIA DGX Spark (Grace ARM64 + Blackwell GB10 128GB Unified Memory)
* **Software**: PyTorch 2.12.0+cu130 (ARM64)
* **Attention Backend**: xformers (automatic fallback due to lack of ARM64 `flash-attn` compiler support)

---

## 🖼️ Sample Details

### 1. SANA 1.6B Standard (20-step)

These images are located in `asset/samples/image/` as `output_sana_0.png`, `output_sana_1.png`, and `output_sana_2.png`.

* **Prompt**:
  > a cyberpunk cat with a neon sign that says "SANA on DGX Spark"
* **Settings**:
  * Resolution: 1024×1024
  * Scheduler: `flow_dpm-solver`
  * Inference Steps: 20 steps
  * Guidance Scale: 4.5
  * Seeds:
    * `output_sana_0.png`: Seed 42
    * `output_sana_1.png`: Seed 43
    * `output_sana_2.png`: Seed 44
  * Performance:
    * Median Generation Time: **6.46 seconds**
    * Peak VRAM: **12.18 GB**

### 2. SANA-Sprint (2-step SCM)

These images are located in `asset/samples/image/` as `output_sprint_0.png`, `output_sprint_1.png`, and `output_sprint_2.png`.

* **Prompt**:
  > a futuristic Tokyo street at night, neon lights, cyberpunk style, ultra detailed
* **Settings**:
  * Resolution: 1024×1024
  * Inference Steps: 2 steps (SCM requires exactly 2 steps)
  * Guidance Scale: 4.5
  * Seeds:
    * `output_sprint_0.png`: Seed 42
    * `output_sprint_1.png`: Seed 43
    * `output_sprint_2.png`: Seed 44
  * Performance:
    * Median Generation Time: **1.22 seconds**
    * Peak VRAM: **10.94 GB**
