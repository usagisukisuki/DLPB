#!/usr/bin/env bash
# Stanford Cars: 全手法の比較実験
# 推定時間: ~20s/ep × 200ep × 10 models ≈ 11 時間 (RTX 4090)
#
# データ配置が必要 (公式 URL 停止中のため手動ダウンロード):
#   data/stanford_cars/cars_train/
#   data/stanford_cars/cars_test/
#   data/stanford_cars/devkit/

set -e
GPU=0
DATASET=cars
EPOCHS=200
BATCH=256
WORKERS=8

PYTHON="uv run python"

MODELS=(resnet18 no_pe ape alibi_2d rpb cpb rope_2d kerple_log_2d dlpb dlpb_O2 dlpb_O3 dlpb_rope_2d dlpb_O2_rope_2d dlpb_O3_rope_2d)

if [ ! -d "./data/stanford_cars" ]; then
    echo "ERROR: data/stanford_cars/ が見つかりません。"
    echo "       Kaggle 等からダウンロードし配置してから再実行してください。"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    RESULT="./results/$DATASET/$MODEL/result.json"
    if [ -f "$RESULT" ]; then
        ACC=$(uv run python -c "import json; print(json.load(open('$RESULT'))['best_val_acc'])" 2>/dev/null || echo "?")
        echo "=== SKIPPING $MODEL (already done: $ACC%) ==="
        continue
    fi
    echo ""
    echo "========================================"
    echo "  Starting: $DATASET / $MODEL"
    echo "========================================"
    $PYTHON train.py \
        --dataset     "$DATASET" \
        --pe_type     "$MODEL" \
        --gpu         $GPU \
        --epochs      $EPOCHS \
        --batch_size  $BATCH \
        --num_workers $WORKERS
done

echo ""
echo "=== All $DATASET experiments complete ==="
uv run python src/summarize_results.py
