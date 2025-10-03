# Universal Training System - Complete Guide

## 🎯 Overview

This is a **zero-hardcoding** training system with comprehensive configuration support. You can control **EVERY** aspect of training through:

- Command-line arguments
- YAML/JSON configuration files
- Programmatic configuration

## 🚀 Quick Start

### 1. Basic Training (Command Line)

```bash
# Minimal command - uses defaults
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattlebody \
    --num-classes 2

# Customized training
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattlebody \
    --num-classes 2 \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 0.001 \
    --device cuda:0
```

### 2. Training with Config File

```bash
# Use predefined configuration
python src/scripts/train_universal.py --config configs/yolov8_cattlebody.yaml

# Override specific parameters
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --learning-rate 0.002 \
    --epochs 200
```

### 3. Advanced Training

```bash
# Custom optimizer and scheduler
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattlebody \
    --num-classes 2 \
    --optimizer adamw \
    --learning-rate 1e-3 \
    --scheduler cosine \
    --scheduler-params '{"T_max": 100, "eta_min": 1e-6}' \
    --augment \
    --mixed-precision \
    --early-stopping
```

## 📋 Complete Argument Reference

### Model Configuration

```bash
--model, -m              # Model architecture (yolov8, faster_rcnn, etc.)
--num-classes, -nc       # Number of object classes
--pretrained             # Use pretrained backbone
--freeze-backbone        # Freeze backbone weights
```

### Dataset Configuration

```bash
--dataset-root, -dr      # Root directory (auto-configures paths)
--train-images           # Training images directory
--train-labels           # Training labels directory
--val-images             # Validation images directory
--val-labels             # Validation labels directory
--image-size, -is        # Input image size (640, 512, etc.)
--label-format, -lf      # Label format (yolo, coco, voc)
```

### Training Hyperparameters

```bash
--epochs, -e             # Number of training epochs
--batch-size, -b         # Training batch size
--learning-rate, -lr     # Initial learning rate
--weight-decay, -wd      # Weight decay (L2 regularization)
--momentum               # Momentum for SGD optimizer
```

### Optimizer Configuration

```bash
--optimizer, -opt        # Optimizer type (sgd, adam, adamw, rmsprop)
--optimizer-params       # Additional parameters as JSON string

# Examples:
--optimizer adamw --optimizer-params '{"betas": [0.9, 0.999], "eps": 1e-8}'
--optimizer sgd --optimizer-params '{"nesterov": true}'
```

### Scheduler Configuration

```bash
--scheduler, -sch        # Scheduler type (step, cosine, plateau, onecycle, none)
--scheduler-params       # Scheduler parameters as JSON string

# Examples:
--scheduler step --scheduler-params '{"step_size": 30, "gamma": 0.1}'
--scheduler cosine --scheduler-params '{"T_max": 100, "eta_min": 1e-6}'
--scheduler plateau --scheduler-params '{"factor": 0.1, "patience": 10}'
--scheduler onecycle --scheduler-params '{"pct_start": 0.3}'
```

### Loss Configuration

```bash
--box-weight             # Weight for box regression loss (default: 7.5)
--cls-weight             # Weight for classification loss (default: 0.5)
--obj-weight             # Weight for objectness loss (default: 1.0)
```

### Data Augmentation

```bash
--augment, -aug          # Enable data augmentation
--no-augment             # Disable data augmentation
--augment-params         # Augmentation parameters as JSON

# Example:
--augment --augment-params '{
    "horizontal_flip": 0.5,
    "rotation": 10,
    "brightness": 0.2,
    "contrast": 0.2
}'
```

### Regularization

```bash
--dropout                # Dropout probability (0.0 - 1.0)
--label-smoothing        # Label smoothing factor
--mixup-alpha            # Mixup alpha parameter
--gradient-clip          # Gradient clipping threshold
```

### Training Strategy

```bash
--warmup-epochs          # Number of warmup epochs
--early-stopping         # Enable early stopping
--early-stopping-patience  # Early stopping patience (epochs)
--mixed-precision        # Enable mixed precision training
--no-mixed-precision     # Disable mixed precision
```

### Validation

```bash
--val-interval           # Validation interval (epochs)
--val-metric             # Metric for best model (mAP, loss, f1)
```

### Checkpointing

```bash
--save-interval          # Checkpoint save interval (epochs)
--save-best-only         # Only save best checkpoint
--checkpoint-dir         # Directory for checkpoints
--resume                 # Resume from checkpoint
```

### Output Configuration

```bash
--output-dir, -o         # Output directory for results
--experiment-name, -n    # Name of experiment
--log-interval           # Logging interval (iterations)
--save-predictions       # Save validation predictions
```

### Device Configuration

```bash
--device, -d             # Device (cuda, cpu, cuda:0, etc.)
--num-workers, -nw       # Number of dataloader workers
--no-pin-memory          # Disable pin memory
```

### Debugging

```bash
--debug                  # Enable debug mode
--profile                # Enable profiling
--seed                   # Random seed for reproducibility
```

## 🔥 Real-World Examples

### Example 1: Basic Training (Like Your Current Command)

```bash
# Your original command:
# python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# New universal command (equivalent):
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattle \
    --num-classes 2 \
    --epochs 2 \
    --batch-size 2 \
    --device cuda:1
```

### Example 2: Change Optimizer on the Fly

```bash
# Try different optimizers easily
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --optimizer sgd \
    --momentum 0.95

python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --optimizer rmsprop \
    --learning-rate 0.001
```

### Example 3: Experiment with Different Schedulers

```bash
# Step LR scheduler
python src/scripts/train_universal.py \
    --dataset-root dataset/cattlebody \
    --model yolov8 \
    --num-classes 2 \
    --scheduler step \
    --scheduler-params '{"step_size": 30, "gamma": 0.1}'

# Cosine annealing
python src/scripts/train_universal.py \
    --dataset-root dataset/cattlebody \
    --model yolov8 \
    --num-classes 2 \
    --scheduler cosine \
    --scheduler-params '{"T_max": 100, "eta_min": 1e-7}'

# Reduce on plateau
python src/scripts/train_universal.py \
    --dataset-root dataset/cattlebody \
    --model yolov8 \
    --num-classes 2 \
    --scheduler plateau \
    --scheduler-params '{"factor": 0.5, "patience": 10}'
```

### Example 4: High-Performance Training

```bash
# Production-ready configuration
python src/scripts/train_universal.py \
    --config configs/high_performance.yaml \
    --device cuda:0 \
    --num-workers 8
```

### Example 5: Quick Testing

```bash
# Fast iteration for debugging
python src/scripts/train_universal.py \
    --config configs/quick_test.yaml \
    --epochs 5 \
    --debug
```

### Example 6: Custom Loss Weights

```bash
# Emphasize box accuracy
python src/scripts/train_universal.py \
    --dataset-root dataset/cattlebody \
    --model yolov8 \
    --num-classes 2 \
    --box-weight 10.0 \
    --cls-weight 0.5 \
    --obj-weight 1.0
```

### Example 7: Different Augmentation Strategies

```bash
# Minimal augmentation
python src/scripts/train_universal.py \
    --dataset-root dataset/cattlebody \
    --model yolov8 \
    --num-classes 2 \
    --augment \
    --augment-params '{"horizontal_flip": 0.5, "rotation": 5}'

# Aggressive augmentation
python src/scripts/train_universal.py \
    --dataset-root dataset/cattlebody \
    --model yolov8 \
    --num-classes 2 \
    --augment \
    --augment-params '{
        "horizontal_flip": 0.5,
        "vertical_flip": 0.3,
        "rotation": 15,
        "brightness": 0.3,
        "contrast": 0.3
    }'
```

### Example 8: Multi-GPU Training

```bash
# Train on specific GPU
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --device cuda:0

# Or different GPU
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --device cuda:1
```

## 📁 Configuration File Examples

### Basic Config (configs/base.yaml)

```yaml
model: yolov8
num_classes: 2
dataset_root: dataset/cattlebody
epochs: 100
batch_size: 8
learning_rate: 0.001
optimizer: adamw
scheduler: cosine
augment: true
device: cuda
```

### Override from Command Line

```bash
# Use base config but change specific parameters
python src/scripts/train_universal.py \
    --config configs/base.yaml \
    --optimizer sgd \
    --learning-rate 0.01 \
    --epochs 200
```

## 🎯 Recommended Workflows

### Workflow 1: Development → Testing → Production

```bash
# 1. Quick development test (2 epochs)
python src/scripts/train_universal.py \
    --config configs/quick_test.yaml \
    --epochs 2

# 2. Medium test (20 epochs)
python src/scripts/train_universal.py \
    --config configs/quick_test.yaml \
    --epochs 20

# 3. Full production training
python src/scripts/train_universal.py \
    --config configs/high_performance.yaml
```

### Workflow 2: Hyperparameter Search

```bash
# Try different learning rates
for lr in 0.0001 0.0005 0.001 0.005; do
    python src/scripts/train_universal.py \
        --config configs/base.yaml \
        --learning-rate $lr \
        --experiment-name "yolov8_lr_${lr}"
done

# Try different optimizers
for opt in adam adamw sgd; do
    python src/scripts/train_universal.py \
        --config configs/base.yaml \
        --optimizer $opt \
        --experiment-name "yolov8_opt_${opt}"
done
```

### Workflow 3: Dataset Comparison

```bash
# Train on different datasets
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattlebody \
    --num-classes 2 \
    --experiment-name yolov8_body

python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattleface \
    --num-classes 2 \
    --experiment-name yolov8_face
```

## 🔧 Troubleshooting

### Issue: Out of Memory

```bash
# Solution: Reduce batch size
python src/scripts/train_universal.py \
    --config configs/base.yaml \
    --batch-size 4  # or 2

# Or disable mixed precision
python src/scripts/train_universal.py \
    --config configs/base.yaml \
    --no-mixed-precision
```

### Issue: Training Too Slow

```bash
# Solution: Increase workers, reduce image size
python src/scripts/train_universal.py \
    --config configs/base.yaml \
    --num-workers 8 \
    --image-size 416  # instead of 640
```

### Issue: Model Not Learning

```bash
# Solution: Adjust learning rate, add warmup
python src/scripts/train_universal.py \
    --config configs/base.yaml \
    --learning-rate 0.01 \
    --warmup-epochs 10
```

## 🎓 Best Practices

1. **Start with a config file**: Easier to track experiments
2. **Use experiment names**: Organize your results
3. **Enable early stopping**: Save time on bad runs
4. **Save configurations**: Auto-saved to output_dir/config.yaml
5. **Monitor logs**: Check output_dir/logs/training.log
6. **Version your configs**: Commit them to git

## 📊 Output Structure

```
outputs/
└── {experiment_name}/
    ├── config.yaml              # Saved configuration
    ├── checkpoints/             # Model checkpoints
    │   ├── best_model.pth
    │   ├── checkpoint_epoch_5.pth
    │   └── checkpoint_epoch_10.pth
    ├── logs/
    │   └── training.log         # Detailed training logs
    ├── metrics/
    │   ├── training_history.json
    │   └── epoch_metrics.csv
    ├── predictions/             # Validation predictions
    └── visualizations/          # Training curves, plots
```

## 🚀 Next Steps

1. Start with `configs/quick_test.yaml` for rapid iteration
2. Move to `configs/yolov8_cattlebody.yaml` for standard training
3. Use `configs/high_performance.yaml` for best results
4. Create your own config files for specific experiments
5. Use command-line overrides for quick parameter testing

---

**Remember**: Zero hardcoding means MAXIMUM flexibility! 🎯
