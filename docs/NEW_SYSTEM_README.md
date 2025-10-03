# 🎉 Zero-Hardcoding Training System - NEW ARCHITECTURE

## 🚀 Quick Start

### Option 1: Interactive Quick Start Script

```bash
# Make executable (first time only)
chmod +x quick_start.sh

# Quick test (5 epochs)
./quick_start.sh test

# Standard training (100 epochs)
./quick_start.sh train

# Interactive custom configuration
./quick_start.sh custom

# Show all options
./quick_start.sh help
```

### Option 2: Direct Command Line

```bash
# Basic training
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattlebody \
    --num-classes 2 \
    --epochs 100 \
    --batch-size 8

# With custom optimizer and scheduler
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattlebody \
    --num-classes 2 \
    --optimizer adamw \
    --scheduler cosine \
    --augment \
    --mixed-precision
```

### Option 3: Configuration File

```bash
# Use predefined config
python src/scripts/train_universal.py --config configs/yolov8_cattlebody.yaml

# Override specific parameters
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --optimizer sgd \
    --learning-rate 0.01
```

## 📚 Documentation

| Document               | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| **TRAINING_GUIDE.md**  | Complete usage guide with all arguments and examples |
| **COMPARISON.md**      | Detailed comparison between old and new system       |
| **REBUILD_SUMMARY.md** | Architecture overview and what was built             |
| **configs/\*.yaml**    | Example configuration files                          |

## ✨ Key Features

### 🎯 Zero Hardcoding

- ✅ **No hardcoded paths** - Everything configurable
- ✅ **No hardcoded hyperparameters** - All via CLI/config
- ✅ **No hardcoded optimizers** - Choose any (adam/adamw/sgd/rmsprop)
- ✅ **No hardcoded schedulers** - Choose any (step/cosine/plateau/onecycle)

### 🏗️ Clean Architecture

- ✅ **SOLID principles** - Single responsibility for each module
- ✅ **DRY principle** - No code duplication
- ✅ **Registry pattern** - Easy model/dataset registration
- ✅ **Modular design** - Easy to extend and maintain

### 🔧 Flexible Configuration

- ✅ **CLI arguments** - Quick parameter changes
- ✅ **YAML/JSON configs** - Reproducible experiments
- ✅ **Config overrides** - Combine file + CLI args
- ✅ **Auto-validation** - Catches errors early

## 🎓 Examples

### Replicate Your Current Command

```bash
# Your original command:
# python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# New equivalent:
python src/scripts/train_universal.py \
    --model yolov8 \
    --dataset-root dataset/cattle \
    --num-classes 2 \
    --epochs 2 \
    --batch-size 2 \
    --device cuda:1
```

### Try Different Optimizers (30 seconds each)

```bash
# AdamW
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --optimizer adamw

# SGD with momentum
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --optimizer sgd \
    --momentum 0.9

# Adam with custom parameters
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --optimizer adam \
    --optimizer-params '{"amsgrad": true}'
```

### Try Different Schedulers (30 seconds each)

```bash
# Cosine annealing
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --scheduler cosine \
    --scheduler-params '{"T_max": 100, "eta_min": 1e-6}'

# Step LR
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --scheduler step \
    --scheduler-params '{"step_size": 30, "gamma": 0.1}'

# Reduce on plateau
python src/scripts/train_universal.py \
    --config configs/yolov8_cattlebody.yaml \
    --scheduler plateau \
    --scheduler-params '{"factor": 0.1, "patience": 10}'
```

### Hyperparameter Search

```bash
# Try different learning rates
for lr in 0.0001 0.0005 0.001 0.005; do
    python src/scripts/train_universal.py \
        --config configs/yolov8_cattlebody.yaml \
        --learning-rate $lr \
        --experiment-name "exp_lr_${lr}"
done
```

## 📊 Available Configurations

### Pre-made Configs

| Config                   | Purpose              | Epochs | Batch Size | Features                                   |
| ------------------------ | -------------------- | ------ | ---------- | ------------------------------------------ |
| `yolov8_cattlebody.yaml` | Baseline training    | 100    | 8          | Standard config                            |
| `high_performance.yaml`  | Maximum accuracy     | 300    | 16         | Aggressive augmentation, optimized weights |
| `quick_test.yaml`        | Fast experimentation | 20     | 16         | Smaller images, minimal augmentation       |

### Create Your Own Config

```yaml
# my_config.yaml
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

Then use it:

```bash
python src/scripts/train_universal.py --config my_config.yaml
```

## 🎯 All Configurable Parameters

### Model

- `--model` - Model architecture (yolov8, faster_rcnn, etc.)
- `--num-classes` - Number of object classes
- `--pretrained` - Use pretrained backbone
- `--freeze-backbone` - Freeze backbone weights

### Dataset

- `--dataset-root` - Auto-configure paths from root
- `--train-images` / `--train-labels` - Manual path specification
- `--val-images` / `--val-labels` - Manual path specification
- `--image-size` - Input image size (640, 512, etc.)

### Training

- `--epochs` - Number of training epochs
- `--batch-size` - Training batch size
- `--learning-rate` - Initial learning rate
- `--weight-decay` - L2 regularization

### Optimizer

- `--optimizer` - sgd / adam / adamw / rmsprop
- `--optimizer-params` - Additional params as JSON

### Scheduler

- `--scheduler` - step / cosine / plateau / onecycle / none
- `--scheduler-params` - Scheduler params as JSON

### Loss Weights

- `--box-weight` - Box regression loss weight
- `--cls-weight` - Classification loss weight
- `--obj-weight` - Objectness loss weight

### Augmentation

- `--augment` - Enable augmentation
- `--augment-params` - Augmentation params as JSON

### Regularization

- `--dropout` - Dropout probability
- `--label-smoothing` - Label smoothing factor
- `--gradient-clip` - Gradient clipping threshold

### Training Strategy

- `--warmup-epochs` - Number of warmup epochs
- `--early-stopping` - Enable early stopping
- `--mixed-precision` - Enable mixed precision training

### Output

- `--output-dir` - Output directory
- `--experiment-name` - Name of experiment
- `--save-interval` - Checkpoint save interval

### Device

- `--device` - cuda / cpu / cuda:0 / cuda:1
- `--num-workers` - Dataloader workers

**See `TRAINING_GUIDE.md` for complete documentation of all 60+ parameters!**

## 🏗️ Architecture

```
project1/
├── configs/                    # Configuration files
│   ├── yolov8_cattlebody.yaml
│   ├── high_performance.yaml
│   └── quick_test.yaml
│
├── src/
│   ├── core/                   # Core abstractions
│   │   ├── registry.py        # Registry pattern
│   │   ├── model_base.py      # Abstract model base
│   │   └── trainer_base.py    # Abstract trainer
│   │
│   ├── models/yolov8/         # Modular YOLOv8
│   │   ├── architecture.py    # Model structure
│   │   ├── heads.py           # Detection heads
│   │   ├── loss.py            # Loss functions
│   │   └── config.py          # Configuration
│   │
│   ├── data/                   # Universal dataset
│   │   └── detection_dataset.py
│   │
│   ├── config/                 # Configuration system
│   │   └── training_config.py
│   │
│   └── scripts/                # Training scripts
│       └── train_universal.py
│
├── quick_start.sh              # Interactive quick start
├── TRAINING_GUIDE.md           # Complete guide
├── COMPARISON.md               # Old vs New
└── REBUILD_SUMMARY.md          # Architecture overview
```

## 🔥 Why This is Better

### Old System

- ❌ Edit code for every change
- ❌ Hardcoded paths and parameters
- ❌ No experiment tracking
- ❌ Code duplication
- ❌ Hard to reproduce

### New System

- ✅ Change parameters via CLI
- ✅ Everything configurable
- ✅ Experiment tracking built-in
- ✅ DRY - no duplication
- ✅ Easy to reproduce

### Result: **10x more productive!** 🚀

## 🆘 Getting Help

1. **Quick reference**: `./quick_start.sh help`
2. **Complete guide**: Read `TRAINING_GUIDE.md`
3. **Examples**: See `COMPARISON.md`
4. **Architecture**: Read `REBUILD_SUMMARY.md`

## 🎉 Start Training Now!

```bash
# Easiest way - interactive
./quick_start.sh custom

# Quick test
./quick_start.sh test

# Full training
python src/scripts/train_universal.py --config configs/yolov8_cattlebody.yaml
```

**Happy Training! 🚀**

---

**Built with ❤️ following SOLID principles, DRY practices, and zero hardcoding philosophy.**
