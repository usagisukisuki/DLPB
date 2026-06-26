#!/usr/bin/env bash
# Flowers-102: 全手法の比較実験
# 推定時間: ~15s/ep × 200ep × 10 models ≈ 8 時間 (RTX 4090)
# データは初回実行時に自動ダウンロード

set -e
GPU=0
DATASET=flowers102
EPOCHS=200
BATCH=256
WORKERS=8

PYTHON="uv run python"

MODELS=(resnet18 no_pe ape alibi_2d rpb cpb rope_2d kerple_log_2d dlpb dlpb_O2 dlpb_O3 dlpb_rope_2d dlpb_O2_rope_2d dlpb_O3_rope_2d)

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
