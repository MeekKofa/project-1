#!/bin/bash
# Universal Training & Visualization Scripts
# Short, intuitive commands for all operations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Print helpers
info() { echo -e "${BLUE}ℹ${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

# ============================================================================
# DATASET OPERATIONS
# ============================================================================

# Analyze all datasets (DEEP ANALYSIS)
analyze() {
    info "Running DEEP dataset analysis..."
    python analyze_datasets_deep.py "$@"
}

# Analyze all datasets (QUICK)
analyze_quick() {
    info "Running quick dataset analysis..."
    python analyze_datasets.py "$@"
}

# Analyze specific dataset
analyze_ds() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh analyze_ds <dataset_name>"
        exit 1
    fi
    info "Analyzing dataset: $1"
    python analyze_datasets.py --dataset "$1"
}

# Generate config for dataset
gen_config() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh gen_config <dataset_name>"
        exit 1
    fi
    info "Generating config for: $1"
    python analyze_datasets.py --generate-config "$1"
}

# ============================================================================
# TRAINING OPERATIONS
# ============================================================================

# Quick test (2 epochs)
test() {
    info "Running quick test (2 epochs)..."
    python src/scripts/train_universal.py \
        --config config.yaml \
        --epochs 2 \
        --batch-size 2 \
        "$@"
}

# Train with default config
train() {
    info "Training with default config..."
    python src/scripts/train_universal.py \
        --config config.yaml \
        "$@"
}

# Train specific dataset
train_ds() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh train_ds <dataset_name>"
        exit 1
    fi
    info "Training on dataset: $1"
    
    # Update config with dataset
    python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['dataset']['name'] = '$1'
config['dataset']['root'] = 'dataset/$1'
with open('config_temp.yaml', 'w') as f:
    yaml.dump(config, f)
"
    
    python src/scripts/train_universal.py \
        --config config_temp.yaml \
        "${@:2}"
    
    rm -f config_temp.yaml
}

# Resume training from checkpoint
resume() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh resume <checkpoint_path>"
        exit 1
    fi
    info "Resuming from: $1"
    python src/scripts/train_universal.py \
        --config config.yaml \
        --resume "$1" \
        "${@:2}"
}

# ============================================================================
# EVALUATION OPERATIONS
# ============================================================================

# Evaluate model
eval() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh eval <model_path>"
        exit 1
    fi
    info "Evaluating model: $1"
    python src/scripts/evaluate.py \
        --model "$1" \
        --config config.yaml \
        "${@:2}"
}

# ============================================================================
# PREPROCESSING OPERATIONS
# ============================================================================

# Preprocess dataset
preprocess() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh preprocess <dataset_name>"
        exit 1
    fi
    info "Preprocessing dataset: $1"
    python src/processing/preprocessing.py \
        --dataset "$1" \
        "${@:2}"
}

# ============================================================================
# VISUALIZATION OPERATIONS
# ============================================================================

# Visualize predictions
vis_pred() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh vis_pred <model_path>"
        exit 1
    fi
    info "Visualizing predictions from: $1"
    python src/training/visualize_predictions.py \
        --model "$1" \
        --config config.yaml \
        "${@:2}"
}

# Visualize dataset
vis_ds() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh vis_ds <dataset_name>"
        exit 1
    fi
    info "Visualizing dataset: $1"
    python src/utils/visualization.py \
        --dataset "$1" \
        --num-samples 16 \
        "${@:2}"
}

# Visualize training logs
vis_logs() {
    if [ -z "$1" ]; then
        error "Usage: ./run.sh vis_logs <experiment_dir>"
        exit 1
    fi
    info "Visualizing training logs: $1"
    python src/utils/plot_training.py \
        --experiment "$1" \
        "${@:2}"
}

# ============================================================================
# UTILITY OPERATIONS
# ============================================================================

# Show help
show_help() {
    cat << EOF
${GREEN}Universal Training & Visualization Commands${NC}

${YELLOW}Dataset Operations:${NC}
  analyze              Deep analysis (brightness, contrast, labels, quality)
  analyze_quick        Quick analysis (structure only)
  analyze_ds <name>    Analyze specific dataset
  gen_config <name>    Generate config for dataset

${YELLOW}Training Operations:${NC}
  test                 Quick test (2 epochs)
  train                Train with default config
  train_ds <name>      Train on specific dataset
  resume <ckpt>        Resume from checkpoint

${YELLOW}Evaluation:${NC}
  eval <model>         Evaluate model

${YELLOW}Preprocessing:${NC}
  preprocess <name>    Preprocess dataset

${YELLOW}Visualization:${NC}
  vis_pred <model>     Visualize predictions
  vis_ds <name>        Visualize dataset
  vis_logs <exp>       Visualize training logs

${YELLOW}Examples:${NC}
  ./run.sh analyze
  ./run.sh train_ds cattlebody
  ./run.sh test --device cuda:1
  ./run.sh vis_ds cattleface
  ./run.sh eval outputs/exp_001/best_model.pt

${YELLOW}Quick Start:${NC}
  1. ./run.sh analyze              # Analyze datasets
  2. ./run.sh test                 # Quick test
  3. ./run.sh train                # Full training
  4. ./run.sh vis_logs outputs/exp # Visualize results

EOF
}

# ============================================================================
# MAIN
# ============================================================================

# Check if command provided
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Execute command
CMD=$1
shift

case "$CMD" in
    # Dataset
    analyze|a)          analyze "$@" ;;
    analyze_quick|aq)   analyze_quick "$@" ;;
    analyze_ds|ads)     analyze_ds "$@" ;;
    gen_config|gc)      gen_config "$@" ;;
    
    # Training
    test|t)             test "$@" ;;
    train|tr)           train "$@" ;;
    train_ds|tds)       train_ds "$@" ;;
    resume|r)           resume "$@" ;;
    
    # Evaluation
    eval|e)             eval "$@" ;;
    
    # Preprocessing
    preprocess|prep|p)  preprocess "$@" ;;
    
    # Visualization
    vis_pred|vp)        vis_pred "$@" ;;
    vis_ds|vd)          vis_ds "$@" ;;
    vis_logs|vl)        vis_logs "$@" ;;
    
    # Help
    help|h|-h|--help)   show_help ;;
    
    *)
        error "Unknown command: $CMD"
        echo ""
        show_help
        exit 1
        ;;
esac
