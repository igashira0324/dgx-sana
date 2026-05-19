# SANA-WM on DGX Spark — 詳細セットアップガイド

このドキュメントでは、NVIDIA DGX Spark (ARM64 + GB10) 上で NVlabs/Sana (SANA-WM) を動作させるためのセットアップ手順を、ハマりどころと回避策を含めて時系列で記録しています。

---

## 前提条件

- NVIDIA DGX Spark (GB10 SuperChip, ARM64)
- Ubuntu 24.04.4 LTS
- NVIDIA Driver 580.142 / CUDA 13.0 がプリインストール済み
- Python 3.12.3 がシステムで利用可能
- インターネット接続（PyPI / GitHub / HuggingFace へのアクセス）

> **注意**: 社内プロキシ環境の場合、`git push/pull` 時に `env -u http_proxy -u https_proxy` が必要な場合があります。

---

## Step 1: リポジトリのクローン

```bash
cd /home/nttdmse/aipf/worldmodel
git clone https://github.com/NVlabs/Sana.git sana
cd sana
```

## Step 2: Python 仮想環境の作成

DGX Spark にはデフォルトで conda が入っていないため、`python3 -m venv` を使用します。

```bash
python3 -m venv sana_env
source sana_env/bin/activate
pip install -U pip setuptools wheel ninja
```

## Step 3: PyTorch のインストール

ARM64 向けの PyTorch wheel は `cu124` インデックスから取得します。
DGX Spark は CUDA 13.0 ですが、PyTorch の cu124 wheel で動作確認済みです。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

実際にインストールされるバージョン:
- torch: 2.12.0+cu130 (pip が自動的に互換バージョンを解決)
- torchvision: 対応バージョン
- torchaudio: 対応バージョン

## Step 4: mmcv の手動インストール

`mmcv==1.7.2` は `pyproject.toml` から除外されています（理由: Python 3.12 で `pkg_resources` 問題が発生するため）。
以下の手順で個別にインストールします。

```bash
# setuptools を 70 未満にダウングレード (pkg_resources 維持のため)
pip install "setuptools<70.0.0"

# Cython のインストール (mmcv のビルドに必要)
pip install cython

# mmcv のビルド (build isolation を無効化して既存の setuptools を使用)
pip install mmcv==1.7.2 --no-build-isolation
```

## Step 5: SANA 本体のインストール

```bash
pip install -e .
```

このステップでは大量の依存パッケージがインストールされます。
特に `xformers==0.0.32.post2` は ARM64 用のプリビルド wheel が存在しないため、
**ソースからのビルドが自動的に実行されます**（15〜30分程度）。

### xformers ビルド時の注意

- ビルドには `ninja` が必要（Step 2 でインストール済み）
- CUDA 13.0 + Compute Capability 12.1 (Blackwell) のカーネルがコンパイルされる
- メモリ使用量が大きいため、他の重い処理は避けること

## Step 6: インストール検証

```bash
python3 test_sana_wm.py
```

期待される出力:
```
=== SANA-WM DGX SPARK Validation Script ===
PyTorch Version: 2.12.0+cu130
CUDA Available: True
CUDA Device Name: NVIDIA GB10
SANA successfully imported. Version: Unknown
Diffusion module successfully imported.
=== Validation Complete ===
```

以下の Warning は正常動作に影響しません:
- `mmcv: UserWarning: On January 1, 2023, MMCV will release v2.0.0...` → mmcv v1 の非推奨警告
- `FutureWarning: torch.cuda.amp.autocast(args...)` → PyTorch API 変更の予告
- `Cannot import apex RMSNorm, switch to vanilla implementation` → apex 未インストール時のフォールバック

---

## インストールできなかったパッケージ

### flash-attn

ARM64 (aarch64) 向けのビルド済み wheel が PyPI に存在しません。
ソースビルドも CUDA カーネルが ARM64 で未対応のため失敗します。

```bash
# 以下は失敗する:
# pip install flash-attn==2.8.2 --no-build-isolation
```

**影響**: SANA は `xformers` または PyTorch ネイティブの attention にフォールバックするため、推論自体は可能です。ただし、特にSANA-WMの長時間動画生成（60秒720p）では性能差が出る可能性があります。

**今後の対応策**:
1. flash-attn の ARM64 対応リリースを待つ
2. [FlashInfer](https://github.com/flashinfer-ai/flashinfer) などの代替ライブラリを検証
3. NVIDIA NGC コンテナ版の flash-attn が ARM64 対応した場合に切り替え

---

## DGX Spark 固有の注意点

### ユニファイドメモリの挙動

DGX Spark は CPU/GPU 間で 128 GB のメモリを共有するユニファイドメモリ設計です。
`nvidia-smi` の Memory-Usage 欄は `Not Supported` と表示されますが、
PyTorch からは 121.7 GB として認識されます。

```
$ nvidia-smi
|   0  NVIDIA GB10   On  |  0000000F:01:00.0  On |  N/A |
| N/A   59C    P0   21W / N/A  | Not Supported   |  28%  Default |
```

メモリ容量的には SANA-WM のフル版（推定 40〜80 GB）でも余裕がありますが、
GPU のメモリ帯域がデータセンター GPU (H100 等) より低いため、生成時間は大幅に長くなる可能性が高いです。

### bf16 サポート

GB10 は bf16 (bfloat16) をサポートしており、mixed precision での推論が可能です。

```python
>>> torch.cuda.is_bf16_supported()
True
```

### CUDA Compute Capability

GB10 の Compute Capability は **12.1** (Blackwell アーキテクチャ) です。
PyTorch 2.12 の CUDA Arch List に `sm_120` が含まれているため、ネイティブサポートされています。

---

## Git リモート構成

```
origin    → https://github.com/igashira0324/dgx-sana.git   (このリポジトリ)
upstream  → https://github.com/NVlabs/Sana.git              (公式リポジトリ)
```

公式の更新を取り込む場合:
```bash
git fetch upstream
git merge upstream/main
```

---

## 参考リンク

- [NVlabs/Sana (公式リポジトリ)](https://github.com/NVlabs/Sana)
- [SANA-WM プロジェクトページ](https://nvlabs.github.io/Sana/WM/)
- [DGX Spark 公式ページ](https://www.nvidia.com/dgx-spark/)
- [PyTorch ARM64 Wheels](https://download.pytorch.org/whl/cu124)
