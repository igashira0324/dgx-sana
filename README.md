# dgx-sana — SANA / SANA-Video on DGX Spark (ARM64)

> **NVlabs/Sana を DGX Spark (GB10 · ARM64 · Unified Memory 128 GB) で動作させるための検証用フォークです。**
>
> 公式リポジトリ: [NVlabs/Sana](https://github.com/NVlabs/Sana)
> プロジェクトページ: [SANA-WM](https://nvlabs.github.io/Sana/WM/)

---

## このリポジトリは何か

本リポジトリは、**DGX Spark (ARM64 + Blackwell GB10)** 上での NVlabs/Sana ファミリーの動作検証用フォークです。
画像生成（SANA 1.6B / SANA-Sprint）は実用レベル、動画生成（SANA-Video）は実機ベンチマークと制約を記録、SANA-WM は公式リリース待ち、という3層ステータスで検証を進めています。

稀少なハードウェア環境における環境構築手順・依存パッケージの修正・実機ベンチマーク結果・ハマりどころと回避策を詳細に記録・共有するリファレンスドキュメントとして運用しています。

> ⚠️ **重要**: 論文やプロジェクトページの公称値（H100 + 最適化環境）と、実機フォールバック実装（ARM64 + flash-attn なし）の結果には大幅な乖離があります。その技術的背景と実機挙動（OSフリーズ現象など）をコードレベルで追求した結果も併記しています。

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

## 実行モード一覧

本フォークでは `run.sh` から 5つの動作モードを選択できます。

```bash
bash run.sh [sana|sprint|gui|video|benchmark]
```

| モード | コマンド | 説明 |
|---|---|---|
| `sana` | `bash run.sh sana` | SANA 1.6B 標準推論（高品質・20 steps） |
| `sprint` | `bash run.sh sprint` | SANA-Sprint（1〜2 step 高速推論） |
| `gui` | `bash run.sh gui` | Gradio Web UI（ブラウザで操作） |
| `video` | `bash run.sh video` | SANA-Video 2B (480p / 81フレーム / 5秒) |
| `benchmark` | `bash run.sh benchmark` | インストール検証・ベンチマーク |

---

## クイックスタート

> ⚠️ **環境構築スクリプトに関する注意**:
> リポジトリ内にある `environment_setup.sh` は公式上流（upstream）に由来する x86_64 + CUDA 12.x 用の古いスクリプトです。
> **DGX Spark（ARM64 + CUDA 13.0）環境では、必ず新しく追加された `setup_sana_env.sh` を使用してください。**

```bash
# 1. リポジトリをクローン
git clone https://github.com/igashira0324/dgx-sana.git
cd dgx-sana

# 2. 環境構築 (venv + 依存パッケージのインストール)
bash setup_sana_env.sh

# 3. インストール検証
source sana_env/bin/activate
python3 test_sana_wm.py

# 4. 画像生成（推奨スタート）
bash run.sh sana

# 5. Web UI 起動
bash run.sh gui
# → ブラウザで http://127.0.0.1:7860 にアクセス
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
| `run.sh` | 1コマンドで各モードを実行するディスパッチャ |
| `app_gui.py` | Gradio Web UI（127.0.0.1 バインドでプロキシ問題を回避） |
| `run_sana.py` | 高品質 SANA 1.6B 推論スクリプト（ウォームアップ・中央値計測付き） |
| `run_sprint.py` | SANA-Sprint（1〜2 step）超高速推論スクリプト |
| `test_sana_wm.py` | インストール検証スクリプト (PyTorch / CUDA / SANA import テスト) |
| `docs/dgx-spark-setup.md` | 詳細なセットアップ手順とトラブルシューティング |
| `asset/samples/video/` | DGX Spark 上で実際に生成された動画サンプル（SANA-Video 2B） |

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

## ベンチマーク実測値 (DGX Spark 実測)

> ⚠️ **重要**: 下記の実測値は、**flash-attn なし・PyTorch フォールバック実装**で計測した値です。
> 公式の論文値・H100 公称値とは大幅に乖離しています。乖離の原因については後述の「[実装成熟度と実機乖離について](#実装成熟度と実機乖離について)」を参照してください。

### 計測メタ情報 (詳細条件)
* **計測実施日**: 2026年5月20日
* **検証コミットID**: `6c1c162` (SANA-Sprint計測追加コミット)
* **OS / GPUドライバ**: Ubuntu 24.04.4 LTS / NVIDIA Driver 580.142 / CUDA 13.0
* **PyTorch バージョン**: 2.12.0+cu130 (ARM64用ビルド)
* **モデル精度**: `bfloat16` (`bf16` / 混合精度)
* **TF32設定**: `torch.backends.cuda.matmul.allow_tf32 = True` (PyTorchデフォルト)
* **torch.compile**: 未適用 (Eager / 未コンパイル)
* **Attention バックエンド**: 
  * 画像生成: `xformers` memory-efficient attention (flash-attn未検出による自動フォールバック)
  * 動画生成: 線形アテンション (`sana_blocks.py`) の `float64` キャスト処理を伴う PyTorch Eager 実行

### DGX Spark 実測値

| タスク | 解像度 | 生成時間 (中央値) | 再現コマンド | 備考 / 使用VRAM |
|---|---|---|---|---|
| **SANA 画像生成 (1.6B)** | 1024×1024 | **6.46 秒** | `bash run.sh sana` | 20 steps / `flow_dpm-solver` / 3回計測中央値。初回ロード約15秒。VRAM: **12.18 GB** |
| **SANA-Sprint (2-step)** | 1024×1024 | **1.22 秒** | `bash run.sh sprint` | **[リアルタイム用途に最適]** 2 steps (SCM) / 3回計測中央値。VRAM: **10.94 GB** |
| **SANA-Video (2B)** | 480p (5秒 / 81f) | **約 12.4 分 (736秒)** | `bash run.sh video` | 50 steps / 10件連続生成時の中央値。**OS応答不能（スラッシング）発生**。VRAM: **Unified Memory上限まで消費** |
| SANA-WM | 720p (60秒) | **未実施** | — | チェックポイント未公開 (2026-05 現在)。実行リスクも高いため未計測 |

### 参考: 公式環境の公称値

| 環境 | タスク | 生成時間 | 備考 |
|---|---|---|---|
| **H100 (80 GB)** | SANA 1.6B 画像生成 (1024px) | **1.2 秒** | `bf16` / flash-attn あり (公式公称値) |
| **H100 (80 GB)** | SANA-Video 2B (480p / 5秒) | **36 秒** | `bf16` / 専用最適化カーネル (公式公称値) |
| RTX 5090 + NVFP4 | SANA-WM 720p 60秒 | ~34 秒 | 最適化カーネル + 量子化 (公式発表) |

> 📌 **乖離の規模**: 画像生成は H100 比 **約 5.4 倍**（6.46秒 vs 1.2秒）、動画生成は H100 比 **約 20 倍**（736秒 vs 36秒）。
> この乖離が単なるハードウェアの差ではなく、実装レベルの問題に起因することを次節で説明します。

---

## 実装成熟度と実機乖離について

### なぜ SANA-Video は LTX-Video や WAN より遅く、OS をフリーズさせたのか

これは **「論文の理論値」と「現在の PyTorch 実装の成熟度」のギャップ**が引き起こした問題です。
DGX Spark 上での実機計測を通じて、以下の 3 つの根本原因を特定しました。

#### 原因 1: `float64` 強制キャスト（最大のボトルネック）

SANA-Video の線形アテンション実装（`LiteLAReLURope`、`diffusion/model/nets/sana_blocks.py`）は、
位置エンコーディング（RoPE）適用の際に毎ステップ `float64`（倍精度）へのキャストを行っています。

```python
# sana_blocks.py より抜粋 — 毎ステップこれが走る
def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
    x_rotated = torch.view_as_complex(
        hidden_states.permute(0, 1, 3, 2).to(torch.float64)  # ← bf16 → float64 へ強制変換
        .unflatten(3, (-1, 2))
    )
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
    return x_out.type_as(hidden_states)
```

GB10 を含む現代の NVIDIA GPU は、AI 推論で常用される `bf16` / `fp16` には大量の演算ユニットを割り当てていますが、`float64`（FP64）の演算能力は `fp32` の 1/32〜1/64 程度に意図的に制限されています。推論パス全体を `bf16` で統一した状態で、このキャストが発生し続けることが深刻なスローダウンの主因です。

#### 原因 2: PyTorch SDPA の高速パスに乗れない（素の Eager 実行）

LTX-Video や WAN-2.1 は標準的な Softmax Attention を使用しているため、PyTorch 組み込みの `F.scaled_dot_product_attention`（SDPA）によって、C++/CUDA で最適化された FlashAttention 等のカーネルが自動的に選択されます。

SANA-Video の線形アテンション（ReLU カーネルベース）は、数式的に Softmax と異なるため SDPA の対象外です。結果として `torch.matmul` の連鎖が PyTorch Eager モード（素の Python）で実行され、最適化カーネルの恩恵を一切受けられません。

#### 原因 3: Unified Memory 特有の「OS 全体フリーズ」

上記 (1)(2) の最適化不足により、各ステップで大量の中間テンソルが動的に確保されます。通常の Discrete GPU（独立 VRAM）であれば `CUDA Out of Memory` エラーでプロセスが即座に終了します。

DGX Spark の Unified Memory（CPU・GPU 共有の 128 GB LPDDR5X）は OOM で死なず、代わりにホストメモリ全体を使い切ろうとします。これがカーネルレベルのメモリ圧迫とスラッシング（CPU-GPU 間の激しいデータ転送）を引き起こし、プロセス単体ではなく **OS 全体が応答不能（完全フリーズ）** になる現象が発生しました。

> 📋 **実測まとめ**: 10件の 5秒動画生成を完走（OOM クラッシュなし）したが、その 124 分間 OS は応答不能だった。これは Unified Memory アーキテクチャの「失敗モード」を示す貴重なデータポイントです。

### モデル別 実機適合性の整理

| モデル | 理論上の設計 | 実装成熟度 | SDPA 適合 | DGX Spark 実績 |
|---|---|---|---|---|
| [FLUX.1][flux] / [SD3][sd3] | 標準的・重い | 高い | ✅ | 重いが安定 |
| [LTX-Video][ltx] | 標準的 | 高い | ✅ | 速く安定 |
| [WAN-2.1][wan] | 標準的 | 高い | ✅ | 速く安定 |
| **SANA 画像 (1.6B)** | 革新的・軽量設計 | 中〜高 | △ (独自カーネル前提) | ✅ 実用ラインに到達 |
| **SANA-Video (2B)** | 革新的・長尺特化 | **低い（研究初期）** | ❌ (FP64混入 / Eager実行) | ⚠️ 完走するが OS フリーズ |
| SANA-WM | 革新的・6-DoF制御 | 未公開 | 未確認 | 🔒 チェックポイント未公開 |

[flux]: https://github.com/black-forest-labs/flux
[sd3]: https://huggingface.co/stabilityai/stable-diffusion-3-medium
[ltx]: https://github.com/Lightricks/LTX-Video
[wan]: https://github.com/Wan-Video/Wan2.1

### 業務利用の指針

* **今すぐ動画生成を業務に使いたい**: → **LTX-Video / WAN-2.1** を採用してください。DGX Spark 上でも快適に動作します。
* **静止画生成（高速・高品質）**: → **SANA 1.6B / SANA-Sprint** は実用ラインに到達しており有効です。
* **将来のワールドモデル（SANA-WM）に備えたい**: → 画像系（SANA 1.6B）で運用ノウハウを溜めつつ、SANA-WM チェックポイントの公開を待つ戦略が合理的です。

> 💡 **SANA-Video / SANA-WM の真価が発揮されるのは、最適化カーネル（Triton/CUDA 実装・FP64 排除）が揃った後です。**
> SANA 1.0 → SANA-Sprint の進化が示すように、画像系では既に最適化の第二フェーズに入っています。動画系も 6〜12 ヶ月以内に追随することが期待されます。

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

### 4. `flash-attn` の ARM64 ビルド不可

**現状**: ARM64 (aarch64) 向けのビルド済み wheel が PyPI に存在せず、ソースビルドも CUDA カーネルのコンパイルが ARM64 で未対応のため失敗する。

**影響**: flash-attn が無い場合、SANA は xformers または PyTorch SDPA にフォールバックする。
* 画像生成（SANA 1.6B）: フォールバックでも実用速度に到達
* 動画生成（SANA-Video）: フォールバック時は FP64 混入 + Eager 実行が顕在化し深刻な性能低下（OS フリーズ）

**今後の対応**: [FlashInfer](https://github.com/flashinfer-ai/flashinfer) の ARM64 対応確認、または SANA-Video 実装側の FP64 排除（Triton カーネル化）を待つ。

### 5. GitHub Actions workflow の push 拒否

**症状**: `git push` 時に `refusing to allow a Personal Access Token to create or update workflow` エラー。

**回避策**: upstream 由来の `.github/workflows/` を削除してコミット。本フォークでは CI 不要のため問題なし。

### 6. `xformers` の ARM64 向けコンパイル所要時間（15〜30分）

**症状**: `pip install xformers==0.0.32.post2` の実行時に進行表示が完全に停止し、フリーズしたように見える。

**原因**: ARM64 向けにビルド済みの xformers wheel が存在しないため、ソースコードから C++ / CUDA コンパイルが走る。

**回避策**: フリーズではないため **そのまま15〜30分待機**してください。`setup_sana_env.sh` 内では `export MAX_JOBS=16` を設定し、`ninja` の事前インストールでコンパイルを高速化しています。

### 7. Hugging Face の Gated Model による推論の「停滞」

**症状**: 初回推論実行時に、進捗表示がないままプログラムが完全に停止（停滞）する。

**原因**: テキストエンコーダ `gemma-2-2b-it` は Hugging Face の **Gated Model** であり、利用規約への合意が必要。アクセス許可がない状態で API が叩かれると、無限に入力待ちになりフリーズする。

**回避策**: SANAのソースコード（`diffusion/model/builder.py`）では、`gemma-2-2b-it` のマッピング先がゲートフリーのミラー **`Efficient-Large-Model/gemma-2-2b-it`** に修正されています。ただし初回起動時に総計約 10 GB のアセットをダウンロードするため、事前キャッシュを推奨します:
```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Efficient-Large-Model/gemma-2-2b-it')"
```

### 8. SANA-Video 実行中の OS 完全フリーズ（Unified Memory スラッシング）

**症状**: `bash run.sh video` 実行後、数分で PC の操作が完全に不能になる。

**原因**: 前節「[実装成熟度と実機乖離について](#実装成熟度と実機乖離について)」を参照。FP64 混入と Eager 実行が組み合わさり、Unified Memory 全体（128 GB）を使い切るスラッシングが OS レベルで発生する。

**回避策（暫定）**:
* **業務用途では LTX-Video / WAN-2.1 への切り替えを推奨**
* SANA-Video を試す場合は、**サンプル数を 1〜2 件に絞り**、**Unified Memory 使用量を `nvidia-smi` で監視**しながら実行する
* OS フリーズを検知したら強制再起動（電源ボタン長押し）で回復可能

## Secret / 安全な運用管理ガイドライン

本リポジトリは検証用途での円滑な運用と、GitHub上のセキュリティリスク（トークン流出防止など）を両立するために、以下のルールに沿って運用されています。

### 1. Hugging Face アクセストークン (`HF_TOKEN`) の扱い
SANAが使用するテキストエンコーダや一部のモデルは Hugging Face 認証を必要とする場合があります。
* **直接ハードコードしない**: スクリプトの引数やコード内部に `token="hf_xxx"` のようにトークン文字列を直書きしないでください。
* **推奨される管理方法**:
  1. `huggingface-cli login` コマンドを実行して `~/.cache/huggingface/token` に認証トークンを安全に永続キャッシュさせる。
  2. あるいは、環境変数 `HF_TOKEN` または `.env` ファイルを利用する。
* **環境ファイルの保護**: `.env` を使用する場合、`.gitignore` に明記されていることを確認し、絶対にリポジトリにコミットされないようにしてください（本フォークではすでに対策済みです）。

### 2. プロキシ環境下でのGit運用と認証
社内等のプロキシ環境下でプロキシエラーにより Git のプッシュ・プルが失敗する場合は、以下のように一時的にプロキシ設定をアンセットして実行することを推奨します。
```bash
env -u http_proxy -u https_proxy git push origin main
```
* **認証情報の平文保存回避**: コマンドライン履歴（`.bash_history` など）に Personal Access Token (PAT) を含む URL を残さないよう、SSHキーペアによる認証か、資格情報ヘルパー (`git-credential-store`等) を使用してください。

---

## ディレクトリ構成 (主要ファイル)

```
dgx-sana/
├── README.md                ← 本ファイル (DGX Spark 導入記録)
├── LICENSE                  ← Apache License 2.0 (NVlabs/Sana 由来)
├── pyproject.toml           ← ARM64 向けに依存関係を修正済み
├── setup_sana_env.sh        ← DGX Spark 向け環境構築スクリプト
├── run.sh                   ← 統合実行スクリプト (sana/sprint/gui/video/benchmark)
├── app_gui.py               ← Gradio Web UI
├── run_sana.py              ← SANA 1.6B 推論スクリプト（ベンチマーク付き）
├── run_sprint.py            ← SANA-Sprint 推論スクリプト（超高速 1step）
├── test_sana_wm.py          ← インストール検証スクリプト
├── docs/
│   └── dgx-spark-setup.md  ← 詳細セットアップ手順
├── asset/
│   └── samples/video/      ← DGX Spark 実機生成サンプル動画
│       ├── sample_anime.mp4 ← アニメ風女性（SANA-Video 2B 480p 5秒）
│       └── sample_car.mp4  ← アニメ風自動車（SANA-Video 2B 480p 5秒）
├── inference_video_scripts/ ← 動画推論スクリプト (公式と同一)
└── configs/                 ← モデル設定ファイル (公式と同一)
```

---

## SANA-WM チェックポイントの公開状況

> 🔒 **2026年5月現在**: `Efficient-Large-Model/SANA-WM_bidirectional` は HuggingFace 上で **認証が必要な状態（Gated / 未公開）** です。
> `inference_sana_wm.py` も NVlabs/Sana のメインリポジトリには含まれていません。

追跡方法:
* [NVlabs/Sana Releases](https://github.com/NVlabs/Sana/releases)
* [Efficient-Large-Model HuggingFace Collection](https://huggingface.co/Efficient-Large-Model)

---

## ライセンスと帰属

本リポジトリは [NVlabs/Sana](https://github.com/NVlabs/Sana) のフォークです。

- **ライセンス**: [Apache License 2.0](LICENSE) (NVlabs/Sana 由来)
- **変更点**: ARM64 (DGX Spark) 環境向けの依存関係修正、実行スクリプト群の追加、実機ベンチマーク記録を含む README の差し替え
- **モデル重み**: HuggingFace 上の公式モデル重みには別途ライセンス条件が適用される場合があります。利用前に各モデルカードを確認してください。

### 引用

公式論文の引用は [NVlabs/Sana](https://github.com/NVlabs/Sana#citation) を参照してください。
