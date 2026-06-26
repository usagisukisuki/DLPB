#!/usr/bin/env bash
# ImageNet-100: 全手法の比較実験
# 推定時間: ~60s/ep × 200ep × 10 models ≈ 33 時間 (RTX 4090)
#
# データ配置が必要:
#   data/imagenet100/train/  (ImageFolder 形式)
#   data/imagenet100/val/    (ImageFolder 形式)

set -e
GPU=0
DATASET=imagenet100
EPOCHS=200
BATCH=256
WORKERS=8

PYTHON="uv run python"

MODELS=(resnet18 no_pe ape alibi_2d rpb cpb rope_2d kerple_log_2d dlpb dlpb_O2 dlpb_O3 dlpb_rope_2d dlpb_O2_rope_2d dlpb_O3_rope_2d)

if [ ! -d "./data/imagenet100/train" ] || [ ! -d "./data/imagenet100/val" ]; then
    echo "ERROR: data/imagenet100/{train,val}/ が見つかりません。"
    echo "       ImageFolder 形式でデータを配置してから再実行してください。"
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
