# dgx-sana — SANA-WM on DGX Spark (ARM64)

> **NVlabs/Sana を DGX Spark (GB10 · ARM64 · Unified Memory 128 GB) で動かすための検証用フォークです。**
>
> 公式リポジトリ: [NVlabs/Sana](https://github.com/NVlabs/Sana)
> プロジェクトページ: [SANA-WM](https://nvlabs.github.io/Sana/WM/)

---

## このリポジトリは何か

[SANA-WM](https://nvlabs.github.io/Sana/WM/) は NVIDIA Research が公開した **2.6B パラメータのワールドモデル** です。
1枚の画像とカメラ軌跡（6-DoF）を入力として、720p・最大1分間の動画を生成できます。

本リポジトリは、**DGX Spark (ARM64 + Blackwell GB10)** という稀少な環境で SANA を動作させるために行った
環境構築手順、依存パッケージの修正内容、ハマりどころと回避策を記録・共有するものです。

公式 README は → [NVlabs/Sana README](https://github.com/NVlabs/Sana/blob/main/README.md)

---

## 検証環境

| 項目 | 値 |
|---|---|
| **ハードウェア** | NVIDIA DGX Spark (GB10 SuperChip) |
| **CPU** | Grace ARM64 (aarch64) |
| **GPU** | NVIDIA GB10, Compute Capability 12.1 |
| **メモリ** | 128 GB LPDDR5X Unified Memory (GPU認識: 121.7 GB) |
| **OS** | Ubuntu 24.04.4 LTS (Noble Numbat) |
| **NVIDIA Driver** | 580.142 |
| **CUDA Toolkit** | 13.0 (V13.0.88) |
| **cuDNN** | 9.2.0 |
| **Python** | 3.12.3 |
| **PyTorch** | 2.12.0+cu130 (ARM64 wheel) |
| **bf16 サポート** | ✅ 対応 |
| **CUDA Arch List** | sm_80, sm_90, sm_100, sm_110, sm_120 |

---

## 公式リポジトリからの変更点

### `pyproject.toml` の修正

ARM64 環境では一部の依存パッケージで PyPI にビルド済み wheel が存在しないため、以下の修正を行いました。

```diff
 # mmcv 1.7.2 は setup.py が pkg_resources を要求し Python 3.12 でビルド失敗
-    "mmcv==1.7.2",
+    # mmcv は pyproject.toml から除外し、手動インストール (後述)

 # ARM64 用の PyTorch wheel はバージョンが公式指定と一致しないため緩和
-    "torchaudio==2.8.0",
-    "torchvision==0.23.0",
+    "torchaudio",
+    "torchvision",

 # triton 3.4.0 は ARM64 wheel が存在しない (3.7.0 以降で対応)
-    "triton==3.4.0",
+    "triton>=3.4.0",
```

### 追加ファイル

| ファイル | 説明 |
|---|---|
| `setup_sana_env.sh` | DGX Spark 向けの環境構築スクリプト (venv + pip) |
| `test_sana_wm.py` | インストール検証スクリプト (PyTorch / CUDA / SANA import テスト) |
| `docs/dgx-spark-setup.md` | 詳細なセットアップ手順とトラブルシューティング |

---

## クイックスタート

```bash
# 1. リポジトリをクローン
git clone https://github.com/igashira0324/dgx-sana.git
cd dgx-sana

# 2. 環境構築 (venv + 依存パッケージのインストール)
bash setup_sana_env.sh

# 3. インストール検証
source sana_env/bin/activate
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

詳細な手順・トラブルシューティングは → [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md)

---

## インストール済みパッケージの状態

| パッケージ | バージョン | 状態 |
|---|---|---|
| PyTorch | 2.12.0+cu130 | ✅ ARM64 wheel |
| xformers | 0.0.32.post2 | ✅ ソースビルド成功 |
| triton | 3.7.0 | ✅ ARM64 wheel |
| diffusers | 0.38.0 | ✅ |
| transformers | 4.57.0 | ✅ |
| accelerate | 1.0.1 | ✅ |
| mmcv | 1.7.2 | ✅ 手動ビルド (`--no-build-isolation`) |
| flash-attn | — | ❌ 未インストール (ARM64 wheel 未提供) |

---

## ハマりどころと回避策

### 1. `mmcv==1.7.2` のビルド失敗

**症状**: `pip install -e .` 実行時に `ModuleNotFoundError: No module named 'pkg_resources'` で停止。

**原因**: mmcv 1.7.2 の `setup.py` が `pkg_resources` を使用しているが、Python 3.12 + setuptools ≥ 70 では `pkg_resources` が削除済み。pip のビルド分離環境で新しい setuptools が入ると発生する。

**回避策**:
```bash
# pyproject.toml から mmcv を除外し、以下で個別インストール
pip install "setuptools<70.0.0"
pip install cython
pip install mmcv==1.7.2 --no-build-isolation
```

### 2. `triton==3.4.0` が ARM64 で見つからない

**症状**: `ERROR: No matching distribution found for triton==3.4.0`

**原因**: PyPI に triton 3.4.0 の `manylinux_aarch64` wheel が存在しない。ARM64 対応は 3.5.0 以降。

**回避策**: `pyproject.toml` のバージョン制約を `triton>=3.4.0` に緩和。実際には 3.7.0 がインストールされる。

### 3. `torchaudio` / `torchvision` のバージョン不一致

**症状**: PyTorch ARM64 wheel のバージョン体系が公式指定と異なりインストール失敗。

**回避策**: バージョン固定を外し、pip に PyTorch と互換のバージョンを自動解決させる。

### 4. `flash-attn` の ARM64 ビルド

**現状**: ARM64 (aarch64) 向けのビルド済み wheel が PyPI に存在せず、ソースビルドも CUDA カーネルのコンパイルが ARM64 で未対応のため失敗する。

**影響**: flash-attn が無い場合、SANA は xformers または PyTorch ネイティブの attention にフォールバックする。推論は可能だが、パフォーマンスが低下する可能性がある。

**今後**: flash-attn の ARM64 対応 wheel のリリースを待つか、[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 等の代替を検討。

### 5. GitHub Actions workflow の push 拒否

**症状**: `git push` 時に `refusing to allow a Personal Access Token to create or update workflow` エラー。

**回避策**: upstream 由来の `.github/workflows/` を削除してコミット。本フォークでは CI 不要のため問題なし。

---

## ベンチマーク (TODO)

> ⚠️ 以下は今後計測予定の項目です。計測結果が出次第、更新します。

| タスク | 解像度 | 長さ | 生成時間 | メモリ使用量 | 備考 |
|---|---|---|---|---|---|
| SANA 画像生成 | 1024×1024 | — | TBD | TBD | |
| SANA-Sprint 1step | 1024×1024 | — | TBD | TBD | |
| SANA-Video | 720p | 5秒 | TBD | TBD | |
| SANA-WM | 720p | 60秒 | TBD | TBD | flash-attn 不在の影響 |

### 参考: 公称値 (公式発表)

| 環境 | タスク | 生成時間 |
|---|---|---|
| H100 (80 GB) | SANA-WM 720p 60秒 | 公称スペック内 |
| RTX 5090 (32 GB) + NVFP4 | SANA-WM 720p 60秒 | ~34秒 |

---

## ディレクトリ構成 (主要ファイル)

```
dgx-sana/
├── README.md                ← 本ファイル (DGX Spark 導入記録)
├── LICENSE                  ← Apache License 2.0 (NVlabs/Sana 由来)
├── pyproject.toml           ← ARM64 向けに依存関係を修正済み
├── setup_sana_env.sh        ← DGX Spark 向け環境構築スクリプト
├── test_sana_wm.py          ← インストール検証スクリプト
├── docs/
│   └── dgx-spark-setup.md   ← 詳細セットアップ手順
├── sana/                    ← SANA ソースコード (公式と同一)
├── diffusion/               ← 拡散モデル関連 (公式と同一)
├── inference_video_scripts/  ← 推論スクリプト (公式と同一)
└── configs/                 ← モデル設定ファイル (公式と同一)
```

---

## ライセンスと帰属

本リポジトリは [NVlabs/Sana](https://github.com/NVlabs/Sana) のフォークです。

- **ライセンス**: [Apache License 2.0](LICENSE) (NVlabs/Sana 由来)
- **変更点**: ARM64 (DGX Spark) 環境向けの依存関係修正、セットアップスクリプト追加、README の差し替え
- **モデル重み**: HuggingFace 上の公式モデル重みには別途ライセンス条件が適用される場合があります。利用前に各モデルカードを確認してください。

### 引用

公式論文の引用は [NVlabs/Sana](https://github.com/NVlabs/Sana#citation) を参照してください。
