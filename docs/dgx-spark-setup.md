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
cd $HOME/aipf/worldmodel
git clone https://github.com/igashira0324/dgx-sana.git sana
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

DGX Spark (ARM64, CUDA 13.0) 向けの PyTorch は、追加の wheel インデックス（cu124等）を指定しなくても、**通常の PyPI リポジトリから直接インストール可能です**。

```bash
pip install torch torchvision torchaudio
```

**インストールの整合性と整合元について**:
- **PyTorch 2.12.0+cu130** が標準の PyPI リポジトリからネイティブの ARM64 (aarch64) ビルドとして解決されます。
- CUDA 13.0 および Blackwell アーキテクチャ (sm_120) 向けの最適化が標準で組み込まれているため、追加の nightly インデックスや NGC コンテナ等から wheel を持ってくる必要はありません。

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

## Step 5: SANA 本体のインストールと xformers ビルド

```bash
pip install -e .
```

このステップでは大量の依存パッケージがインストールされます。

> [!WARNING]
> **xformers のコンパイル時間に関する注意**
> `xformers==0.0.32.post2` は ARM64 向けにビルド済みの wheel が存在しないため、**完全にソースからコンパイルされます**。
> - コンパイルには **約15〜30分** かかり、進捗表示が一時的に止まります（フリーズではありません）。
> - コンパイルの並列度を上げ、フリーズやメモリ不足 (OOM) を防ぐため、`setup_sana_env.sh` 内では `export MAX_JOBS=16` を設定しています。マシンスペックに合わせてコア数を調整してください。
> - 事前に `ninja` が正しくインストールされていることを確認してください（Step 2 で導入済み）。

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

## インストールできなかったパッケージと影響

### flash-attn

ARM64 (aarch64) 向けのビルド済み wheel が PyPI に存在しません。
ソースビルドも CUDA カーネルが ARM64 で未対応のため失敗します。

```bash
# 以下は失敗する:
# pip install flash-attn==2.8.2 --no-build-isolation
```

**フォールバックの具体的な挙動と影響**:
- **インポートと基本動作**: `flash-attn` が未インストールでも `test_sana_wm.py` のインポートおよび動作検証は正常に通ります。
- **アテンションのフォールバック**: SANA は自動的に `xformers` の `memory_efficient_attention` もしくは PyTorch ネイティブの `scaled_dot_product_attention` (SDPA / PyTorch 2.0+ の機能) へフォールバックします。
- **パフォーマンスと制限**: SDPA および xformers フォールバックにより画像生成や短尺の動画生成は動作可能ですが、SANA-WM の特徴である **60秒クラスの長尺 720p 動画生成** を実行する際には、メモリ効率や計算レイテンシの面で差が出る（またはシーケンス長が長くなった際に OOM が発生しやすくなる）可能性があります。

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
