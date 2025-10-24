#!/usr/bin/env bash
set -euo pipefail

# Comprehensive training and evaluation script for all detection models
# Trains each model, then evaluates on test split
# Usage: ./run.sh [-d dataset] [-e epochs] [-b batch_size] [-s eval_split] [-c cuda_device]
#!/usr/bin/env bash


# ==============================================================
# CATTLE DETECTION: TRAIN + EVALUATE ALL MODELS (CUSTOM DEVICE)
# ==============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="cattle"
EPOCHS="100"
BATCH_SIZE="4"
EVAL_SPLIT="test"
CUDA_DEVICE="cuda:2"  # <--- Default device; change here if desired

# Parse command-line arguments
while getopts "d:e:b:s:c:h" opt; do
    case $opt in
        d) DATASET="$OPTARG" ;;
        e) EPOCHS="$OPTARG" ;;
        b) BATCH_SIZE="$OPTARG" ;;
        s) EVAL_SPLIT="$OPTARG" ;;
        c) CUDA_DEVICE="$OPTARG" ;;  # Allows override via -c cuda:1 etc.
        h)
            echo "Usage: $0 [-d dataset] [-e epochs] [-b batch_size] [-s eval_split] [-c cuda_device]"
            echo "Example: $0 -d cattle -e 100 -b 8 -c cuda:2"
            exit 0
            ;;
        *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# List of models to run
MODELS=(
    "faster_rcnn"
    "yolov8_resnet"
#    "fusion_model"
)

echo "=========================================="
echo "CATTLE DETECTION: FULL MODEL SUITE RUN"
echo "=========================================="
echo "Dataset:      $DATASET"
echo "Epochs:       $EPOCHS"
echo "Batch Size:   $BATCH_SIZE"
echo "Eval Split:   $EVAL_SPLIT"
echo "CUDA Device:  $CUDA_DEVICE"
echo "Models:       ${MODELS[*]}"
echo "=========================================="

# Function: Train one model
train_model() {
    local model="$1"
    echo ""
    echo "----------------------------------------"
    echo "TRAINING: $model on $DATASET ($CUDA_DEVICE)"
    echo "----------------------------------------"

    python "$PROJECT_ROOT/train.py" train \
        -m "$model" \
        -d "$DATASET" \
        -e "$EPOCHS" \
        -b "$BATCH_SIZE" \
        --device "$CUDA_DEVICE"

    echo "✓ Training complete for $model"
}

# Function: Evaluate one model
eval_model() {
    local model="$1"
    local checkpoint_path="$PROJECT_ROOT/outputs/$DATASET/$model/checkpoints/best.pth"

    if [[ ! -f "$checkpoint_path" ]]; then
        echo "⚠️  Checkpoint not found: $checkpoint_path"
        echo "   Skipping evaluation for $model"
        return
    fi

    echo ""
    echo "----------------------------------------"
    echo "EVALUATING: $model on $DATASET ($EVAL_SPLIT split)"
    echo "----------------------------------------"

    python "$PROJECT_ROOT/train.py" eval \
        -m "$model" \
        -d "$DATASET" \
        -p "$checkpoint_path" \
        --split "$EVAL_SPLIT" \
        --device "$CUDA_DEVICE"

    echo "✓ Evaluation complete for $model"
}

# Main Execution
echo ""
echo "PHASE 1: TRAINING ALL MODELS"
echo "============================="
for model in "${MODELS[@]}"; do
    train_model "$model"
done

echo ""
echo "=========================================="
echo "PHASE 2: EVALUATING ALL MODELS"
echo "=========================================="
for model in "${MODELS[@]}"; do
    eval_model "$model"
done

echo ""
echo "=========================================="
echo "FULL SUITE COMPLETE!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "- outputs/$DATASET/<model>/logs/train.log"
echo "- outputs/$DATASET/<model>/metrics/"
echo "- outputs/$DATASET/<model>/evaluations/"
echo "- outputs/$DATASET/<model>/visualizations/"
echo ""
echo "To compare metrics across models:"
echo "  python -c \"import json; [print(f'{m}: {json.load(open(f'outputs/$DATASET/{m}/metrics/test_metrics.json'))['latest']['map_50']:.4f}') for m in ['faster_rcnn', 'yolov8_resnet']]\""
echo "=========================================="
