#!/usr/bin/env bash
# Modified for DGX Spark ARM64 by igashira0324 on 2026-05-19
# - 1-step pipeline runner execution script with auto venv activation and arg dispatcher

set -e

# スクリプトがあるディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境の自動有効化
if [ -f "sana_env/bin/activate" ]; then
    echo "Activating virtual environment (sana_env)..."
    source sana_env/bin/activate
else
    echo "ERROR: Virtual environment 'sana_env' not found. Please run 'bash setup_sana_env.sh' first."
    exit 1
fi

# モード引数の判定 (デフォルトは sana)
MODE=${1:-sana}

if [ "$MODE" = "sprint" ]; then
    echo "=== Running SANA-Sprint (Fast 1-Step Generation) ==="
    python3 run_sprint.py
elif [ "$MODE" = "sana" ]; then
    echo "=== Running SANA 1.5 1.6B (High Quality 20-Step Generation) ==="
    python3 run_sana.py
elif [ "$MODE" = "gui" ]; then
    echo "=== Launching SANA 1.5 1.6B Web UI ==="
    python3 app_gui.py
elif [ "$MODE" = "benchmark" ]; then
    echo "=== Running SANA-WM Benchmark ==="
    python3 test_benchmark.py
else
    echo "ERROR: Invalid mode '$MODE'."
    echo "Usage: bash run.sh [sana|sprint|gui|benchmark]"
    exit 1
fi
