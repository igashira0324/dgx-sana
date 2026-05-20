# SANA-Video 2B Sample Metadata

This file documents the generation parameters and performance metrics for the video samples generated on the **NVIDIA DGX Spark** platform using **SANA-Video 2B**.

---

## 💻 System Configuration

* **Hardware**: NVIDIA DGX Spark (Grace ARM64 + Blackwell GB10 128GB Unified Memory)
* **Software**: PyTorch 2.12.0+cu130 (ARM64)
* **Attention Backend**: xformers (automatic fallback due to lack of ARM64 `flash-attn` compiler support)

---

## 📊 Shared Generation Settings

* **Model**: SANA-Video 2B (480p)
* **Resolution**: 832×480 (480p, latent size: 21t, 60h, 104w)
* **Duration**: 5 seconds (81 frames, 16 fps)
* **Scheduler**: `flow_dpm-solver`
* **Inference Steps**: 50 steps
* **Classifier-Free Guidance (CFG)**: 6.0
* **Flow Shift**: 7.0
* **Motion Score**: 10
* **Precision**: `bfloat16` (`bf16` mixed precision via `accelerate`)
* **Generation Performance**:
  * **Time Taken**: **12.4 minutes (736 seconds)** per video
  * **Memory Footprint**: High memory thrashing/swapping on LPDDR5X Unified Memory (causing system UI to freeze/lock up temporarily during execution, while successfully avoiding OOM crashes)

---

## 🎥 Sample Details

### 1. `sample_anime.mp4` (Anime Woman)

* **Source File**: `003. Japanese animated film style, a young woman standif405c5f83a.mp4`
* **Prompt**:
  > Japanese animated film style, a young woman standing on a ship's deck, looking back at the camera with a serene expression. The background shows the ocean and sky stretching out behind her. Medium shot focusing on her profile.
* **Negative Prompt**:
  > A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience.
* **Seed**: 0

### 2. `sample_car.mp4` (Anime White Car)

* **Source File**: `008. In the daytime, an anime-style white car drives to9ad0c5748f.mp4`
* **Prompt**:
  > In the daytime, an anime-style white car drives towards the camera, splashing water from a pond as it passes by, medium shot.
* **Negative Prompt**:
  > A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience.
* **Seed**: 0
