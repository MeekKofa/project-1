#!/bin/bash

# Quick Start Script for Universal Training System
# Usage: ./quick_start.sh [command]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Python is available
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    print_success "Python found: $(python --version)"
}

# Show usage
show_usage() {
    cat << EOF
${GREEN}Universal Training System - Quick Start${NC}

${YELLOW}Usage:${NC}
    ./quick_start.sh [command]

${YELLOW}Commands:${NC}
    help           Show this help message
    test           Run a quick test (5 epochs)
    train          Run standard training (100 epochs)
    train-fast     Run fast training (20 epochs, smaller images)
    train-hp       Run high-performance training (300 epochs)
    custom         Interactive custom training configuration

${YELLOW}Examples:${NC}
    ./quick_start.sh test          # Quick 5-epoch test
    ./quick_start.sh train         # Standard training
    ./quick_start.sh train-hp      # High-performance training
    ./quick_start.sh custom        # Interactive configuration

${YELLOW}Manual Training:${NC}
    python src/scripts/train_universal.py --config configs/yolov8_cattlebody.yaml

${YELLOW}Documentation:${NC}
    - TRAINING_GUIDE.md    Complete usage guide
    - COMPARISON.md        Old vs New comparison
    - REBUILD_SUMMARY.md   Architecture overview

EOF
}

# Quick test
run_test() {
    print_header "Running Quick Test (5 epochs)"
    
    python src/scripts/train_universal.py \
        --model yolov8 \
        --dataset-root dataset/cattlebody \
        --num-classes 2 \
        --epochs 5 \
        --batch-size 4 \
        --learning-rate 0.001 \
        --optimizer adamw \
        --scheduler cosine \
        --augment \
        --experiment-name "quick_test_$(date +%Y%m%d_%H%M%S)" \
        --device cuda
    
    print_success "Test completed!"
}

# Standard training
run_train() {
    print_header "Running Standard Training (100 epochs)"
    
    python src/scripts/train_universal.py \
        --config configs/yolov8_cattlebody.yaml \
        --experiment-name "standard_train_$(date +%Y%m%d_%H%M%S)"
    
    print_success "Training completed!"
}

# Fast training
run_train_fast() {
    print_header "Running Fast Training (20 epochs, smaller images)"
    
    python src/scripts/train_universal.py \
        --config configs/quick_test.yaml \
        --epochs 20 \
        --image-size 416 \
        --batch-size 16 \
        --experiment-name "fast_train_$(date +%Y%m%d_%H%M%S)"
    
    print_success "Fast training completed!"
}

# High-performance training
run_train_hp() {
    print_header "Running High-Performance Training (300 epochs)"
    
    print_warning "This will take a long time!"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python src/scripts/train_universal.py \
            --config configs/high_performance.yaml \
            --experiment-name "hp_train_$(date +%Y%m%d_%H%M%S)"
        
        print_success "High-performance training completed!"
    else
        print_warning "Training cancelled"
    fi
}

# Interactive custom training
run_custom() {
    print_header "Custom Training Configuration"
    
    # Model
    echo -e "${YELLOW}Select model:${NC}"
    echo "  1) yolov8"
    echo "  2) faster_rcnn (coming soon)"
    read -p "Choice [1]: " model_choice
    model_choice=${model_choice:-1}
    
    case $model_choice in
        1) MODEL="yolov8" ;;
        2) MODEL="faster_rcnn" ;;
        *) MODEL="yolov8" ;;
    esac
    
    # Dataset
    read -p "Dataset root path [dataset/cattlebody]: " dataset_root
    dataset_root=${dataset_root:-dataset/cattlebody}
    
    read -p "Number of classes [2]: " num_classes
    num_classes=${num_classes:-2}
    
    # Training parameters
    read -p "Epochs [100]: " epochs
    epochs=${epochs:-100}
    
    read -p "Batch size [8]: " batch_size
    batch_size=${batch_size:-8}
    
    read -p "Learning rate [0.001]: " lr
    lr=${lr:-0.001}
    
    # Optimizer
    echo -e "\n${YELLOW}Select optimizer:${NC}"
    echo "  1) adamw"
    echo "  2) adam"
    echo "  3) sgd"
    echo "  4) rmsprop"
    read -p "Choice [1]: " opt_choice
    opt_choice=${opt_choice:-1}
    
    case $opt_choice in
        1) OPTIMIZER="adamw" ;;
        2) OPTIMIZER="adam" ;;
        3) OPTIMIZER="sgd" ;;
        4) OPTIMIZER="rmsprop" ;;
        *) OPTIMIZER="adamw" ;;
    esac
    
    # Scheduler
    echo -e "\n${YELLOW}Select scheduler:${NC}"
    echo "  1) cosine"
    echo "  2) step"
    echo "  3) plateau"
    echo "  4) onecycle"
    echo "  5) none"
    read -p "Choice [1]: " sch_choice
    sch_choice=${sch_choice:-1}
    
    case $sch_choice in
        1) SCHEDULER="cosine" ;;
        2) SCHEDULER="step" ;;
        3) SCHEDULER="plateau" ;;
        4) SCHEDULER="onecycle" ;;
        5) SCHEDULER="none" ;;
        *) SCHEDULER="cosine" ;;
    esac
    
    # Augmentation
    read -p "Enable augmentation? [Y/n]: " augment
    augment=${augment:-Y}
    
    if [[ $augment =~ ^[Yy]$ ]]; then
        AUGMENT="--augment"
    else
        AUGMENT="--no-augment"
    fi
    
    # Device
    read -p "Device [cuda]: " device
    device=${device:-cuda}
    
    # Experiment name
    read -p "Experiment name [custom_train]: " exp_name
    exp_name=${exp_name:-custom_train}
    exp_name="${exp_name}_$(date +%Y%m%d_%H%M%S)"
    
    # Summary
    print_header "Training Configuration Summary"
    echo -e "Model:         ${GREEN}$MODEL${NC}"
    echo -e "Dataset:       ${GREEN}$dataset_root${NC}"
    echo -e "Classes:       ${GREEN}$num_classes${NC}"
    echo -e "Epochs:        ${GREEN}$epochs${NC}"
    echo -e "Batch size:    ${GREEN}$batch_size${NC}"
    echo -e "Learning rate: ${GREEN}$lr${NC}"
    echo -e "Optimizer:     ${GREEN}$OPTIMIZER${NC}"
    echo -e "Scheduler:     ${GREEN}$SCHEDULER${NC}"
    echo -e "Augmentation:  ${GREEN}$([[ $AUGMENT == "--augment" ]] && echo "Yes" || echo "No")${NC}"
    echo -e "Device:        ${GREEN}$device${NC}"
    echo -e "Experiment:    ${GREEN}$exp_name${NC}"
    
    echo
    read -p "Start training? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_header "Starting Training"
        
        python src/scripts/train_universal.py \
            --model "$MODEL" \
            --dataset-root "$dataset_root" \
            --num-classes "$num_classes" \
            --epochs "$epochs" \
            --batch-size "$batch_size" \
            --learning-rate "$lr" \
            --optimizer "$OPTIMIZER" \
            --scheduler "$SCHEDULER" \
            $AUGMENT \
            --device "$device" \
            --experiment-name "$exp_name"
        
        print_success "Training completed!"
    else
        print_warning "Training cancelled"
    fi
}

# Main script
main() {
    # Check Python
    check_python
    
    # Parse command
    COMMAND=${1:-help}
    
    case $COMMAND in
        help|--help|-h)
            show_usage
            ;;
        test)
            run_test
            ;;
        train)
            run_train
            ;;
        train-fast)
            run_train_fast
            ;;
        train-hp)
            run_train_hp
            ;;
        custom)
            run_custom
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
