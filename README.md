# 🐄 Cattle Detection System

End-to-end PyTorch pipeline for cattle detection across multiple datasets and model architectures. The project ships with reproducible training and evaluation utilities, dataset sanity checks, and visualization tooling so you can iterate quickly and keep artifacts organized.

---

## Key Features
- **Model zoo**: Faster R-CNN and YOLOv8 (ResNet/CSP backbones) behind a single CLI.
- **Config-first workflow**: YAML + CLI merge with sensible defaults and dataset auto-detection.
- **Unified artifacts**: Metrics, predictions, plots, and visualizations written to a predictable directory layout.
- **Standalone evaluation**: Re-run metrics and export predictions from any checkpoint with one command.
- **Diagnostics**: System/config verifiers and dataset scanners catch issues before long training runs.

---

## Requirements
- Python 3.8+
- CUDA-capable GPU recommended (CPU and Apple MPS supported via flags)
- `pip install -r requirements.txt`

```bash
# From the project root
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment (venv, Conda, Poetry, etc.) to isolate dependencies.

---

## Dataset Layout
The repository expects raw data under `dataset/<name>/` and (optionally) processed data under `processed_data/<name>/`. Each dataset should provide `train/`, `val/`, and `test/` splits in YOLO format.

```
dataset/
    cattle/
        train/ images + labels
        val/
        test/
    cattlebody/
    cattleface/
processed_data/
    cattle/
    cattlebody/
    cattleface/
```

Use `python test_dataset_detection.py` if you are unsure whether the expected metadata is present—this script prints what the loaders see without importing heavy torch dependencies.

---

## Quick Start Workflow

### 1. Train a model
```bash
python train.py train -m yolov8_resnet -d cattle -e 20 -b 4
```
- `-m / --model`: one of `faster_rcnn`, `yolov8_resnet`, `yolov8_csp`
- `-d / --dataset`: `cattle`, `cattlebody`, `cattleface`
- `-e / --epochs`, `-b / --batch-size`: override defaults from the config
- Add `--mixed-precision` for FP16 on CUDA, `--device cpu` to force CPU, or `--resume <checkpoint>` to continue a run

### 2. Evaluate / test a checkpoint
```bash
python train.py eval \
    -m yolov8_resnet \
    -d cattle \
    -p outputs/cattle/yolov8_resnet/checkpoints/best.pth \
    --split test
```
- Works with `best.pth`, `latest.pth`, or any `epoch_XXX.pth`
- `--split` supports `train`, `val`, or `test`
- Optional flags: `--batch-size`, `--run-name`, `--save-predictions` / `--no-save-predictions`, `--device`

All evaluation artifacts land in `outputs/<dataset>/<model>/evaluations/<run_name>/`, keeping them separate from training-time metrics.

### 3. Inspect metrics & visualizations
- Metrics CSV/JSON: `outputs/<dataset>/<model>/metrics/`
- Evaluation exports: `outputs/<dataset>/<model>/evaluations/<run>/metrics/`
- Detection overlays: `outputs/<dataset>/<model>/visualizations/<split>/`
- Plots: `outputs/<dataset>/<model>/metrics/plots/`

Our `BaseTrainer` maintains a clean history (`metrics_summary.csv`, `{split}_metrics.csv/json`) while pruning old per-epoch files.

---

## Command Reference

### `train.py`
- `python train.py train ...` – launch training with merged config/CLI settings.
- `python train.py eval ...` – run standalone evaluation on any checkpoint (see above).
- `python train.py preprocess ...` – placeholder for future preprocessing integration (current preprocessing is handled automatically by loaders).

### Diagnostics & utilities
- `python verify_system.py` – environment sanity checks (CUDA availability, package versions, etc.).
- `python verify_training_config.py` – validate merged config to catch missing keys or conflicting overrides.
- `python test_dataset_detection.py` – lightweight dataset metadata inspection.
- `python check_system.py` – quick hardware summary.
- `python scripts/analyze_datasets_deep.py --dataset cattle` – optional dataset statistics and figures.
- `python scripts/workflow_manager.py --dataset cattle --stage all` – orchestrated end-to-end pipeline (analyze → preprocess → train → evaluate).

### Visualization helpers
Visualization images are generated automatically during validation/testing (configurable via `visualization` settings). To inspect the latest assets quickly:
```bash
ls outputs/cattle/yolov8_resnet/visualizations/val
```
Adjust `visualization.max_epochs_to_keep` in the config if you want to retain more or fewer epochs.

---

## Workflow helper script
To streamline recurring commands, use the Bash wrapper we provide.

```bash
chmod +x scripts/workflow_commands.sh
./scripts/workflow_commands.sh --help
```

Available subcommands include:

| Command | Description |
| --- | --- |
| `train [args...]` | Proxy to `python train.py train ...` |
| `eval [args...]` | Proxy to `python train.py eval ...` |
| `preprocess [args...]` | Runs `scripts/workflow_manager.py --stage preprocess ...` |
| `workflow [args...]` | Direct pass-through to `scripts/workflow_manager.py` |
| `verify-system` | Runs `verify_system.py` |
| `verify-config` | Runs `verify_training_config.py` |
| `dataset-detect` | Runs `test_dataset_detection.py` |
| `check-system` | Runs `check_system.py` |
| `analyze-dataset [args...]` | Runs the deep dataset analysis script |
| `list-metrics --dataset D --model M` | Lists available metric files |
| `list-visuals --dataset D --model M` | Lists generated visualization images |

Example: re-run the test split on the best checkpoint via the wrapper:
```bash
./scripts/workflow_commands.sh eval \
    -m yolov8_resnet \
    -d cattle \
    -p outputs/cattle/yolov8_resnet/checkpoints/best.pth \
    --split test
```

---

## Outputs & Artifacts
```
outputs/<dataset>/<model>/
├── checkpoints/          # best.pth, latest.pth, epoch_XXX.pth
├── logs/                 # train.log, evaluation logs
├── metrics/              # train_metrics.csv, {split}_metrics.csv/json, metrics_summary.csv, plots/
├── predictions/          # {split}_predictions.json (latest evaluation)
├── visualizations/       # {split}/ detection overlays
└── evaluations/<run>/    # Standalone eval artifacts (metrics, predictions, logs, config)
```

During evaluation we disable checkpoint saves, so the new artifacts remain isolated under `evaluations/<run_name>/` while still writing split-aware metrics and predictions.

---

## Troubleshooting Checklist
- `python verify_system.py` – confirm environment before long trainings.
- `python verify_training_config.py` – review merged config values.
- `python test_dataset_detection.py` – ensure class counts and label files are detected correctly.
- `./scripts/workflow_commands.sh list-metrics --dataset cattle --model yolov8_resnet` – verify metrics exported as expected.
- Inspect `outputs/<dataset>/<model>/logs/train.log` for per-epoch summaries.

If you encounter issues with the visualization quota, adjust `visualization.max_epochs_to_keep` or disable the feature via `visualization.enabled` in `config.yaml`.

---

## Project Structure (abridged)
```
project1/
├── train.py                      # Entry point for training/evaluation
├── verify_system.py              # Environment diagnostics
├── verify_training_config.py     # Config validator
├── test_dataset_detection.py     # Dataset metadata probe
├── src/
│   ├── cli/args.py               # CLI definitions
│   ├── config/                   # Defaults, manager, config.yaml
│   ├── evaluation/               # Standalone evaluation orchestrator
│   ├── loaders/                  # Dataset registry + transforms
│   ├── models/                   # Detection model registry & definitions
│   ├── training/                 # Trainers, loops, checkpoints
│   └── utils/                    # Metric plotting, helpers
├── scripts/
│   ├── workflow_commands.sh      # Command wrapper (this doc)
│   ├── workflow_manager.py       # Multi-stage workflow
│   └── analyze_datasets_deep.py  # Dataset analysis utilities
├── outputs/                      # Training & evaluation artifacts
└── docs/                         # Additional documentation (evaluation, architecture, etc.)
```

---

Need something that is not documented here? Check `docs/` for deeper dives (`docs/evaluation.md` covers evaluation artifacts in detail) or open an issue describing the workflow you would like to automate.

---

## Legacy README (original quickstart)

# 🐄 Cattle Detection System# 🐄 Cattle Detection System````markdown

Professional ML engineering system for cattle detection using PyTorch.# Cattle Detection & Recognition System

## ✨ Features> **Production-ready object detection pipeline for cattle datasets with dynamic configuration and robust preprocessing**

- **Multiple Models**: Faster R-CNN, YOLOv8 (ResNet/CSP backbones)A comprehensive machine learning pipeline for cattle detection and recognition using various deep learning architectures including Faster R-CNN, YOLOv8, and Ultralytics YOLO.

- **Registry Pattern**: Add models/datasets with 1 line of code

- **Auto-Detection**: Automatically detects classes from dataset files## 🚀 Quick Start - Complete Workflow

- **Clean Architecture**: Modular, extensible, maintainable

- **Config Priority**: CLI > YAML > Defaults### Prerequisites

- **No Hardcoding**: Everything is configurable

````bash

## 🚀 Quick Start# Install dependencies

pip install -r requirements.txt

### Install Dependencies```

```bash

pip install torch torchvision pyyaml pillow numpy### Step 1: Analyze Your Dataset (Optional but Recommended)

````

````bash

### Verify System# Deep analysis to understand your data

```bashpython scripts/analyze_datasets_deep.py

python verify_system.py

```# Results will be in: dataset_analysis_results/

````

### Train a Model

````bash### Step 2: Preprocess the Dataset ⚠️ **REQUIRED**

# Quick test (2 epochs, ~2-3 minutes)

python train.py train -m yolov8_resnet -d cattle -e 2 -b 4```bash

# First time preprocessing - No split needed (dataset already has train/val/test folders!)

# Full trainingpython main.py preprocess --dataset cattlebody

python train.py train -m yolov8_resnet -d cattle -e 100 -b 8

```# Or use workflow manager

python scripts/workflow_manager.py --dataset cattlebody --stage preprocess

## 📖 Documentation

# 🔄 REPROCESS if already processed (use --force or -f flag)

- **README_QUICKSTART.md** - Complete reference with all optionspython main.py preprocess --dataset cattlebody --force

- **USAGE_GUIDE.md** - Detailed usage examples and commandspython main.py preprocess --dataset cattlebody -f  # Short version



## 📝 Commands# Optional: Custom split ratio with force reprocess

python main.py preprocess --dataset cattlebody -s 0.8 -f

```bash

# Train# ✅ This creates: processed_data/cattlebody/

python train.py train -m <model> -d <dataset> [options]```



# Evaluate### Step 3: Train the Model

python train.py eval -m <model> -d <dataset> -p <checkpoint>

```bash

# Preprocess# Train with default settings (using src/config/config.yaml)

python train.py preprocess -d <dataset>python main.py train --model faster_rcnn --dataset cattlebody --epochs 50 --batch-size 4

````

# Or with YOLOv8

**Available Models**: `faster_rcnn`, `yolov8_resnet`, `yolov8_csp` python main.py train --model yolov8 --dataset cattlebody --epochs 100 --batch-size 8

**Available Datasets**: `cattle`, `cattlebody`, `cattleface`

# With GPU selection

## 🏗️ Architecturepython main.py train --model faster_rcnn --dataset cattlebody --device cuda:0

```

```

src/### Step 4: Evaluate the Model

├── cli/ # Command-line interface

├── config/ # Configuration (YAML + CLI)```bash

├── models/ # Faster R-CNN, YOLOv8# Evaluate trained model

├── loaders/ # Data loaders + transformspython main.py evaluate --model faster_rcnn --dataset cattlebody --batch-size 4

└── training/ # Training infrastructure

`````# With custom confidence threshold

python main.py evaluate --model faster_rcnn --dataset cattlebody --score-threshold 0.7

## 🎯 Examples```



```bash### 🎯 Complete Pipeline (All-in-One)

# Basic training

python train.py train -m yolov8_resnet -d cattle -e 100 -b 8```bash

# Run complete workflow: analyze → preprocess → train → evaluate

# Resume trainingpython scripts/workflow_manager.py --dataset cattlebody --stage all

python train.py train -m yolov8_resnet -d cattle \```

  --resume outputs/cattle/yolov8_resnet/checkpoints/latest.pth

## 🏗️ Project Structure

# Custom learning rate

python train.py train -m yolov8_resnet -d cattle -e 150 -lr 0.0001````

project1/

# Mixed precision├── main.py                 # 🚀 Main entry point - START HERE

python train.py train -m yolov8_resnet -d cattle --mixed-precision├── requirements.txt        # 📦 Python dependencies

├── README.md              # 📖 This file

# CPU training├──

python train.py train -m yolov8_resnet -d cattle --device cpu├── dataset/               # 📊 Raw datasets

```│   ├── cattlebody/        # Raw cattle body dataset

│   └── cattleface/        # Raw cattle face dataset

## 📊 Output Structure├──

├── processed_data/        # 🔄 Preprocessed datasets (created by workflow)

```│   ├── cattlebody/        # Preprocessed cattlebody

outputs/{dataset}/{model}/│   ├── cattle/            # Preprocessed cattle

├── checkpoints/     # best.pth, latest.pth, epoch_*.pth│   └── cattleface/        # Preprocessed cattleface

├── logs/            # train.log├──

├── metrics/         # CSV metrics├── scripts/               # All executable workflow scripts

└── visualizations/  # Plots and images│   ├── workflow_manager.py        # Main workflow orchestrator

```│   ├── preprocess_dataset.py      # Data preprocessing

│   ├── analyze_datasets_deep.py   # Deep dataset analysis

## 🐛 Troubleshooting│   └── analyze_datasets.py        # Basic analysis

├──

**CUDA Out of Memory**: Reduce batch size `-b 4`  ├── src/                   # 💻 Source code

**Dataset Not Found**: Check `dataset/cattle/data.yaml` exists  │   ├── config/            # Configuration management

**Import Errors**: Make sure you're in project root directory│   │   ├── config.yaml   # ⭐ Main unified configuration

│   │   ├── dynamic_config_loader.py  # Runtime detection

See **README_QUICKSTART.md** for detailed troubleshooting.│   │   ├── settings.py    # System settings

│   │   ├── paths.py       # Path configurations

---│   │   └── hyperparameters.py # Training hyperparameters

│   ├── models/            # 🧠 Model architectures

**Ready to train?** 🚀│   │   ├── faster_rcnn.py

```bash│   │   ├── yolov8.py

python train.py train -m yolov8_resnet -d cattle -e 100 -b 8

```├── dataset/               # 📊 Dataset files


## 📁 Project Structure│   ├── cattleface/        # Cattle face dataset

│   │   ├── CowfaceImage/  # Original images

```│   │   └── Annotation/    # Annotation files

project1/│   └── cattlebody/        # Cattle body dataset

├── config.yaml                    # Main training configuration (dynamic!)├── processed_data/        # 🔄 Processed dataset

├── dataset_profiles.yaml          # Dataset-specific settings│   ├── cattleface/        # Face detection splits

├── main.py                        # Legacy main entry point│   │   ├── train/

├── requirements.txt               # Python dependencies│   │   ├── val/

││   │   └── test/

├── scripts/                       # All executable scripts│   └── cattlebody/        # Body detection splits

│   ├── workflow_manager.py        # Main workflow orchestrator│       ├── train/

│   ├── preprocess_dataset.py      # Data preprocessing│       ├── val/

│   ├── analyze_datasets_deep.py   # Dataset analysis│       └── test/

│   └── analyze_datasets.py        # Basic analysis├──

│├── src/                   # 💻 Source code

├── src/                           # Source code│   ├── config/            # Configuration modules

│   ├── config/                    # Configuration management│   │   ├── __init__.py

│   │   └── dynamic_config_loader.py  # Runtime property detection│   │   ├── settings.py    # Main configuration

│   ├── data/                      # Data loading│   │   ├── paths.py       # Path configurations

│   ├── models/                    # Model implementations│   │   └── hyperparameters.py # Training hyperparameters

│   ├── training/                  # Training logic│   ├── models/            # 🧠 Model architectures

│   ├── evaluation/                # Evaluation metrics│   │   ├── faster_rcnn.py

│   └── utils/                     # Utility functions│   │   ├── yolov8.py

││   │   └── fusion_model.py

├── dataset/                       # Raw datasets│   ├── training/          # 🎯 Training scripts

│   ├── cattle/│   │   ├── train_faster_rcnn.py

│   ├── cattlebody/│   │   ├── train_yolov8.py

│   └── cattleface/│   │   ├── train_ultralytics.py

││   │   └── utils.py

├── processed_data/                # Preprocessed datasets│   ├── evaluation/        # 📈 Evaluation scripts

├── dataset_analysis_results/      # Analysis outputs│   │   ├── evaluate_model.py

├── outputs/                       # Training outputs│   │   └── metrics.py

││   ├── processing/        # Data processing

├── docs/                          # Documentation│   │   ├── preprocessing.py

│   ├── QUICK_REFERENCE.md         # Command cheat sheet│   │   └── dataset.py

│   ├── FINAL_SUMMARY.md           # System overview│   ├── utils/             # 🛠️ Utility functions

│   └── CONFIG_SYSTEM_README.md    # Configuration guide│   │   ├── data_validation.py

││   │   ├── logging_utils.py

└── archive/                       # Old/deprecated files│   │   ├── model_validation.py

```│   │   └── memory.py

│   └── scripts/           # 📜 Additional scripts

## ⚙️ Configuration│       ├── train_all.py

│       ├── evaluate_all.py

### Dynamic Configuration (No Hardcoding!)│       └── inference.py

├──

The system automatically detects:└── outputs/               # 📤 All outputs organized by dataset/model

- ✅ Number of classes    ├── {dataset}/         # Dataset-specific outputs

- ✅ Class names    │   └── {model}/       # Model-specific outputs

- ✅ Image/label counts    │       ├── models/    # Trained model weights

- ✅ Dataset format (YOLO/COCO/VOC)    │       ├── logs/      # Training and execution logs

- ✅ Optimal loss function    │       ├── metrics/   # Evaluation metrics (JSON + TXT)

    │       ├── images/    # Generated images/visualizations

Simply set the dataset name in `config.yaml`:    │       ├── results/   # Training results

    │       └── checkpoints/ # Model checkpoints

```yaml    └── legacy/            # Legacy output structure

dataset:        ├── models/

  name: cattlebody      # Change this to switch datasets        ├── logs/

  split: raw           # raw or processed        ├── images/

  # Everything else is auto-detected!        └── results/

`````

### Training Presets## 🚀 Quick Start

Quick preset switching for different training modes:### 1. Install Dependencies

`yaml`bash

active_preset: standard # Options: quick_test, standard, high_performancepip install -r requirements.txt

`````



| Preset | Epochs | Resolution | Use Case |### 2. Interactive Menu

|--------|--------|------------|----------|

| quick_test | 20 | 416x416 | Fast iteration |```bash

| standard | 100 | 640x640 | Balanced training |python main.py

| high_performance | 300 | 640x640 | Maximum accuracy |```



## 🔧 Key FeaturesThis launches an interactive menu for all operations.



- **Dynamic Detection**: No hardcoded dataset properties### 3. Robust Dataset Configuration (NEW!)

- **Analysis-Driven**: Uses dataset insights for optimal settings

- **Robust Preprocessing**: Quality filtering, letterboxing, format normalizationThe system now supports both traditional dataset names and direct dataset paths for maximum flexibility:

- **Auto-Configured Loss**: Handles class imbalance automatically

- **Unified Workflow**: Single CLI for all operations```bash

- **Comprehensive Logging**: Track everything from analysis to training# Traditional method (backward compatible)

python main.py train -m faster_rcnn -d cattlebody

## 📊 Dataset Status

# NEW: Robust method using direct paths (works anywhere!)

| Dataset | Classes | Images | Status | Resolution |python main.py train -m faster_rcnn --dataset-path /path/to/any/dataset

|---------|---------|--------|--------|------------|

| cattlebody | 1 | 4,852 | ⚠️ Needs preprocessing | 640x640 |# With comprehensive validation (recommended)

| cattle | 2 (imbalanced) | 11,369 | ✅ Ready | 1280x1280 |python main.py train -m faster_rcnn --dataset-path dataset/cattle --validate-dataset

| cattleface | 0 | 6,528 | ❌ No labels | N/A |```



## 🎯 Common Commands### 4. CUDA Error Prevention



```bashThe system automatically prevents common CUDA device-side assert errors:

# Check dataset health

python scripts/workflow_manager.py --dataset cattlebody --stage check```bash

# Debug your dataset before training (highly recommended)

# Run deep analysispython main.py debug --dataset-path dataset/cattle --validate-dataset

python scripts/workflow_manager.py --dataset cattlebody --stage analyze

# The system will automatically detect and fix:

# Preprocess dataset (fixes issues)# - Class count mismatches

python scripts/workflow_manager.py --dataset cattlebody --stage preprocess# - Label range issues

# - Invalid bounding boxes

# Train with standard preset# - Dataset compatibility problems

python scripts/workflow_manager.py --dataset cattlebody --stage train```



# Run full pipeline## � Command Reference

python scripts/workflow_manager.py --dataset cattlebody --stage all

```The system provides several commands with both short and long aliases for convenience. **NEW**: All training commands now support robust dataset configuration!



## 📚 Documentation### � **NEW: Robust Dataset Configuration**



See the `docs/` folder for detailed guides:The system now supports two modes for maximum flexibility and portability:



- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command cheat sheet#### **Mode 1: Robust Path-Based (Recommended)**

- **[FINAL_SUMMARY.md](docs/FINAL_SUMMARY.md)** - Complete system overview

- **[CONFIG_SYSTEM_README.md](docs/CONFIG_SYSTEM_README.md)** - Configuration guide```bash

- **[WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md)** - Step-by-step workflows# Works anywhere - just specify the dataset path directly

python main.py train -m faster_rcnn --dataset-path /absolute/path/to/dataset

## 🛠️ Installationpython main.py train -m faster_rcnn --dataset-path ./relative/path/to/dataset

python main.py train -m faster_rcnn --dataset-path dataset/cattle

```bash

# Install dependencies# With validation and overrides

pip install -r requirements.txtpython main.py train -m faster_rcnn --dataset-path dataset/cattle --validate-dataset --num-classes 3

```

# Verify installation

python scripts/workflow_manager.py --help#### **Mode 2: Traditional Name-Based (Backward Compatible)**

```

```bash

## 🔍 Troubleshooting# Uses predefined dataset names (still works)

python main.py train -m faster_rcnn -d cattlebody

| Issue | Solution |python main.py train -m faster_rcnn -d cattleface

|-------|----------|python main.py train -m faster_rcnn -d cattle

| Image/label mismatch | Run preprocessing: `python scripts/workflow_manager.py --dataset cattlebody --stage preprocess` |```

| Class imbalance | Set `loss.type: auto` in config.yaml (default) |

| Small objects | Increase resolution to 1280 in config.yaml |### 🛡️ **CUDA Error Prevention System**

| Too slow | Use `active_preset: quick_test` |

**Before training, always run diagnostics to prevent CUDA device-side assert errors:**

## 🤝 Contributing

```bash

When adding features:# Comprehensive dataset diagnostic (HIGHLY RECOMMENDED)

- Keep config.yaml for hyperparameters onlypython main.py debug --dataset-path dataset/cattle --validate-dataset --sample-size 10

- Use dynamic_config_loader.py for dataset facts

- Add quality checks to preprocessing# Quick diagnostic check

- Document everythingpython main.py debug --dataset-path dataset/cattle



## 📝 License# Debug with traditional dataset names

python main.py debug -d cattle --validate-dataset

[Your License]```



## 🙏 Acknowledgments**The diagnostic system checks for:**



Built with modern ML engineering practices for production-ready cattle detection.- ✅ CUDA compatibility and GPU status

- ✅ Dataset structure and file integrity

---- ✅ Label range validation (prevents assert errors)

- ✅ Class count analysis and recommendations

**Ready for world-class cattle detection! 🐄🚀**- ✅ Bounding box format validation

- ✅ Model compatibility testing

### �📋 All Available Commands

| Command          | Robust Support | Description                 | Key Features                                          |
| ---------------- | -------------- | --------------------------- | ----------------------------------------------------- |
| `train`          | ✅ **NEW**     | Train a model               | Dataset paths, validation, auto-class detection       |
| `debug`          | ✅ **NEW**     | CUDA error diagnostics      | Comprehensive validation, CUDA error prevention       |
| `train-advanced` | ✅ **NEW**     | Advanced training           | Path support + optimized profiles, augmentation       |
| `evaluate`       | ✅ **NEW**     | Evaluate a trained model    | Path support + comprehensive metrics                  |
| `preprocess`     | ⚠️ Planned     | Preprocess datasets         | Split data into train/val/test sets                   |
| `optimize`       | ✅ **NEW**     | Hyperparameter optimization | Path support + automated parameter tuning             |
| `cleanup`        | ➖             | Clean old metric files      | Remove individual epoch files, keep consolidated data |
| `info`           | ➖             | Show system information     | List models, datasets, and project structure          |

### 🏷️ Argument Aliases (Short Forms)

For faster command entry, you can use short aliases:

| Long Form            | Short | Description                      | Example Values                               |
| -------------------- | ----- | -------------------------------- | -------------------------------------------- |
| `--model`            | `-m`  | Model architecture               | `faster_rcnn`, `yolov8`, `ultralytics`       |
| `--dataset`          | `-d`  | Dataset name (traditional)       | `cattlebody`, `cattleface`, `cattle`         |
| `--dataset-path`     | N/A   | **NEW**: Direct dataset path     | `dataset/cattle`, `/path/to/data`            |
| `--validate-dataset` | N/A   | **NEW**: Pre-training validation | Boolean flag (no value needed)               |
| `--num-classes`      | N/A   | **NEW**: Override class count    | `2`, `3`, `4` (for testing)                  |
| `--sample-size`      | N/A   | **NEW**: Debug sample count      | `5`, `10`, `20` (for debug command)          |
| `--epochs`           | `-e`  | Number of training epochs        | `50`, `100`, `200`                           |
| `--batch-size`       | `-b`  | Training batch size              | `2`, `4`, `8`, `16`                          |
| `--learning-rate`    | `-lr` | Learning rate                    | `0.001`, `0.002`, `0.01`                     |
| `--score-threshold`  | `-t`  | Confidence threshold             | `0.5`, `0.7`, `0.8`                          |
| `--output-dir`       | `-o`  | Output directory                 | `./custom_output/`                           |
| `--model-path`       | `-p`  | Specific model path              | `outputs/model.pth`                          |
| `--split-ratio`      | `-s`  | Train/val split ratio            | `0.7`, `0.8`, `0.9`                          |
| `--force`            | `-f`  | Force operation                  | Used with cleanup/preprocess                 |
| `--profile`          | `-pr` | Training profile                 | `default`, `high_precision`, `fast_training` |
| `--trials`           | `-tr` | Optimization trials              | `10`, `20`, `50`                             |
| `--max-epochs`       | `-me` | Maximum epochs                   | `100`, `200`, `300`                          |
| `--augmentation`     | `-a`  | Enable data augmentation         | Boolean flag                                 |
| `--early-stopping`   | `-es` | Enable early stopping            | Boolean flag                                 |
| `--force`            | `-f`  | Force operation                  | Used with cleanup/preprocess                 |
| `--profile`          | `-pr` | Training profile                 | `default`, `high_precision`, `fast_training` |
| `--trials`           | `-tr` | Optimization trials              | `10`, `20`, `50`                             |
| `--max-epochs`       | `-me` | Maximum epochs                   | `100`, `200`, `300`                          |
| `--augmentation`     | `-a`  | Enable data augmentation         | Boolean flag                                 |
| `--early-stopping`   | `-es` | Enable early stopping            | Boolean flag                                 |

### 📝 Command Examples

#### 🚀 **NEW: Robust Dataset Training (Recommended)**

**Always start with diagnostics to prevent CUDA errors:**

```bash
# 1. FIRST: Run comprehensive diagnostics (HIGHLY RECOMMENDED)
python main.py debug --dataset-path dataset/cattle --validate-dataset --sample-size 5

# 2. THEN: Train with robust path-based configuration
python main.py train --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -e 100 -b 4 -lr 0.002

# 3. Advanced robust training with all features
python main.py train-advanced --dataset-path dataset/cattle -m faster_rcnn --validate-dataset --num-classes 3 -pr high_precision -a -es -me 300
```

#### 🔄 **Traditional Training (Backward Compatible)**

```bash
# Basic training (your standard command format)
python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# Customized training
python main.py train -m faster_rcnn -d cattlebody -e 100 -b 4 -lr 0.002 --device cuda

# Advanced training
python main.py train-advanced -m faster_rcnn -d cattlebody -pr high_precision -a -es -me 300
```

#### 1. **Debug Command** - `python main.py debug` ⭐ **NEW**

```bash
# Comprehensive diagnostic (MUST RUN BEFORE TRAINING)
python main.py debug --dataset-path dataset/cattle --validate-dataset --sample-size 10

# Quick diagnostic check
python main.py debug --dataset-path dataset/cattle

# Traditional dataset diagnostic
python main.py debug -d cattle --validate-dataset

# Full diagnostic with overrides
python main.py debug --dataset-path dataset/cattle --validate-dataset --num-classes 3 --sample-size 5
```

#### 2. **Train Command** - `python main.py train`

```bash
# Your standard training command
python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# Robust training with path
python main.py train --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -e 50 -b 4

# Different models
python main.py train -m yolov8 -d cattle -e 50 -b 8 --device cuda:0
python main.py train -m ultralytics -d cattlebody -e 100 -b 16 --device cuda
```

#### 3. **Train-Advanced Command** - `python main.py train-advanced`

```bash
# High precision training (recommended for best results)
python main.py train-advanced --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -pr high_precision -a -es -me 300

# Quick training for experiments
python main.py train-advanced -m faster_rcnn -d cattlebody -pr fast_training -me 50
```

#### 4. **Evaluate Command** - `python main.py evaluate`

````bash
# Robust evaluation with dataset paths
python main.py evaluate --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -b 4 -t 0.7

# Traditional evaluation
python main.py evaluate -m faster_rcnn -d cattlebody -b 4 -t 0.7
```#### 3. **Train-Advanced Command** - `python main.py train-advanced`

Advanced training with optimization features and robust dataset support:

```bash
# High precision training (recommended for best results)
python main.py train-advanced \
    --dataset-path dataset/cattle \   # NEW: Robust path support
    -m faster_rcnn \                  # Use Faster R-CNN model
    --validate-dataset \              # NEW: Pre-training validation
    -pr high_precision \              # Use high precision profile
    -a \                              # Enable data augmentation
    -es \                             # Enable early stopping
    -me 300                           # Maximum 300 epochs

# Quick training for experiments (backward compatible)
python main.py train-advanced -m faster_rcnn -d cattlebody -pr fast_training -me 50
````

#### 4. **Evaluate Command** - `python main.py evaluate`

Comprehensive model evaluation with robust dataset support:

```bash
# NEW: Robust evaluation with dataset paths
python main.py evaluate \
    --dataset-path dataset/cattle \   # Direct dataset path
    -m faster_rcnn \                  # Evaluate Faster R-CNN model
    --validate-dataset \              # Validate before evaluation
    -b 4 \                           # Batch size 4 for evaluation
    -t 0.7                           # Confidence threshold 0.7

# Traditional evaluation (still works)
python main.py evaluate -m faster_rcnn -d cattlebody -b 4 -t 0.7
```

#### 5. **Preprocess Command** - `python main.py preprocess`

```bash
# Basic preprocessing (80% train, 20% val)
python main.py preprocess -d cattlebody

# Custom split ratio with force reprocess
python main.py preprocess -d cattlebody -s 0.7 -f
```

#### 6. **Optimize Command** - `python main.py optimize`

```bash
# Basic hyperparameter optimization
python main.py optimize -m faster_rcnn -d cattlebody

# Advanced optimization with robust paths
python main.py optimize --dataset-path dataset/cattle -m faster_rcnn -pr high_precision -tr 20 -me 100
```

### 🎯 Recommended Workflows

#### 🛡️ **CUDA Error-Free Training (HIGHLY RECOMMENDED)**

```bash
# 1. ALWAYS start with comprehensive diagnostics
python main.py debug --dataset-path dataset/cattle --validate-dataset --sample-size 5

# 2. Train with robust configuration
python main.py train-advanced --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -pr high_precision -a -es -me 300

# 3. Evaluate with the same robust configuration
python main.py evaluate --dataset-path dataset/cattle -m faster_rcnn --validate-dataset
```

#### 🚀 **Machine-Portable Training (Works Anywhere)**

```bash
# Copy your code to any machine and run:
python main.py debug --dataset-path /path/to/your/dataset --validate-dataset
python main.py train --dataset-path /path/to/your/dataset -m faster_rcnn --validate-dataset
python main.py evaluate --dataset-path /path/to/your/dataset -m faster_rcnn
```

#### 🎯 **Quick Development Workflow**

```bash
# Your standard quick training
python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# Quick evaluation
python main.py evaluate -m faster_rcnn -d cattle -t 0.5
```

#### **For Quick Experiments**

```bash
# Quick training test
python main.py train -m faster_rcnn -d cattlebody -e 10 -b 2

# Quick evaluation
python main.py evaluate -m faster_rcnn -d cattlebody
```

#### **For Hyperparameter Tuning**

```bash
# Run optimization to find best parameters
python main.py optimize -m faster_rcnn -d cattlebody -tr 10

# Train with optimized parameters (check logs for best params)
python main.py train-advanced -m faster_rcnn -d cattlebody -pr high_precision
```

## �📊 Data Processing

### Preprocess Datasets

```bash
# Preprocess cattle face dataset
python main.py preprocess -d cattleface -s 0.8

# Preprocess cattle body dataset
python main.py preprocess -d cattlebody -s 0.8

# Force reprocessing
python main.py preprocess -d cattleface -s 0.7 -f
```

### Available Datasets

- **cattleface**: Cattle face detection dataset
- **cattlebody**: Cattle body detection dataset

## 🎯 Model Training

### Available Models

| Model         | Description                          | Best For                | Status |
| ------------- | ------------------------------------ | ----------------------- | ------ |
| `faster_rcnn` | Faster R-CNN with ResNet-50 backbone | High accuracy detection | ✅ Stable |
| `ultralytics` | Ultralytics YOLO implementation      | Real-time detection ⭐ **RECOMMENDED** | ✅ Stable |
| `yolov8`      | Custom YOLOv8 (modular)             | Advanced customization  | ⚠️ Fixed (or use ultralytics) |

### Training Examples

```bash
# Your standard training command
python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# Different models
python main.py train -m ultralytics -d cattle -e 50 -b 8 --device cuda:0  # ⭐ RECOMMENDED for YOLO
python main.py train -m yolov8 -d cattle -e 50 -b 8 --device cuda:0       # Custom YOLOv8

# Robust training with validation
python main.py train --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -e 50 -b 4

# Advanced training with all features
python main.py train-advanced --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -pr high_precision -a -es -me 300
```

### Training Output Structure

```
outputs/
└── {dataset}/              # e.g., cattlebody/
    └── {model}/            # e.g., faster_rcnn/
        ├── models/
        │   ├── faster_rcnn.pth     # Main model
        │   └── best_model.pth      # Best checkpoint
        ├── logs/
        │   └── faster_rcnn_cattlebody.log
        ├── metrics/
        │   ├── metrics_epoch_1.json    # Detailed metrics
        │   ├── metrics_epoch_1.txt     # Human-readable summary
        │   └── final_metrics.json
        ├── images/         # Training visualizations
        ├── results/        # Training curves, plots
        └── checkpoints/    # Model checkpoints
```

## 📈 Model Evaluation

### Evaluation Examples

```bash
# Basic evaluation
python main.py evaluate -m faster_rcnn -d cattlebody

# Custom evaluation with specific parameters
python main.py evaluate -m faster_rcnn -d cattlebody -b 4 -t 0.7

# Robust evaluation with validation
python main.py evaluate --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -b 4 -t 0.7

# Standalone evaluation script
python src/evaluation/evaluate_model.py -m outputs/cattlebody/faster_rcnn/models/faster_rcnn.pth -d cattlebody -o outputs/cattlebody/faster_rcnn/evaluation/
```

### Evaluation Metrics

The system provides comprehensive metrics:

- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.75**: Mean Average Precision at IoU 0.75
- **mAP@0.5:0.95**: COCO-style mAP across IoU thresholds
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Example Evaluation Output

```
============================================================
DETECTION METRICS SUMMARY
============================================================
📊 Dataset Statistics:
   • Images evaluated: 714
   • Total predictions: 714
   • Total ground truths: 714
   • Score threshold: 0.50

🎯 Average Precision (AP) Metrics:
   • mAP@0.5      : 0.7209
   • mAP@0.75     : 0.5142
   • mAP@0.5:0.95 : 0.4331

📈 Classification Metrics @ IoU 0.5:
   • Precision    : 0.8519
   • Recall       : 0.7493
   • F1-Score     : 0.7973

🔍 Per-Class Metrics @ IoU 0.5:
   • Cattle  : AP=0.7209, P=0.8519, R=0.7493, F1=0.7973
============================================================
```

## 🚀 Performance Optimization

### Performance Analysis Tool

The system includes a powerful performance optimization script that analyzes your current results and provides actionable recommendations:

```bash
# Analyze current performance and get recommendations
python src/scripts/optimize_performance.py --current-map 0.6744 --target-map 0.8

# Get hyperparameter suggestions
python src/scripts/optimize_performance.py --current-map 0.6744 --suggest-hyperparams --model faster_rcnn

# Analyze results from output directory
python src/scripts/optimize_performance.py --analyze-results outputs/cattlebody/faster_rcnn/metrics/
```

### Performance Improvement Strategies

Based on your current performance, the system provides specific recommendations:

| Performance Level         | Current mAP | Recommended Actions                                                               | Expected Improvement |
| ------------------------- | ----------- | --------------------------------------------------------------------------------- | -------------------- |
| 🔴 Poor (0-30%)           | < 0.30      | • Review data quality<br>• Increase training epochs<br>• Check model architecture | +15-20% mAP          |
| 🟠 Below Average (30-50%) | 0.30-0.50   | • Hyperparameter tuning<br>• Data augmentation<br>• Longer training               | +10-15% mAP          |
| 🟡 Average (50-70%)       | 0.50-0.70   | • Advanced training profiles<br>• Optimization techniques<br>• Fine-tuning        | +5-10% mAP           |
| 🟢 Good (70-85%)          | 0.70-0.85   | • Ensemble methods<br>• Architecture optimization<br>• Edge case handling         | +3-5% mAP            |
| 🟢 Excellent (85%+)       | > 0.85      | • Model efficiency<br>• Inference optimization<br>• Production deployment         | Maintain performance |

### Optimization Commands by Performance Level

#### For mAP 50-70% (Your Current Range: 67.44%)

```bash
# Recommended: Advanced training with high precision profile
python main.py train-advanced -m faster_rcnn -d cattlebody -pr high_precision -a -es -me 300

# Alternative: Optimized basic training
python main.py train -m faster_rcnn -d cattlebody -e 200 -b 4 -lr 0.002

# Hyperparameter optimization
python main.py optimize -m faster_rcnn -d cattlebody -pr high_precision -tr 10
```

#### For mAP 30-50% (Needs Major Improvement)

```bash
# Aggressive optimization
python main.py train-advanced -m faster_rcnn -d cattlebody -pr high_precision -a -es -me 500

# Multiple optimization trials
python main.py optimize -m faster_rcnn -d cattlebody -tr 20 -me 200
```

#### For mAP 70%+ (Fine-tuning)

```bash
# Fine-tuning with careful parameters
python main.py train -m faster_rcnn -d cattlebody -e 100 -b 8 -lr 0.001

# Ensemble approach (train multiple models)
python main.py train-advanced -m faster_rcnn -d cattlebody -pr high_precision
python main.py train-advanced -m yolov8 -d cattlebody -pr high_precision
```

### Metrics Tracking and Analysis

The enhanced metrics system provides:

```bash
# View consolidated metrics
cat outputs/cattlebody/faster_rcnn/metrics/training_metrics.csv

# Enhanced final report with recommendations
cat outputs/cattlebody/faster_rcnn/metrics/enhanced_evaluation_report.txt

# Training curves visualization (automatically generated)
ls outputs/cattlebody/faster_rcnn/metrics/training_curves.png
```

### Performance Troubleshooting

| Issue                | Symptoms                  | Solution Commands                                                      |
| -------------------- | ------------------------- | ---------------------------------------------------------------------- |
| **Overfitting**      | Training acc ↑, val acc ↓ | `python main.py train-advanced -m faster_rcnn -d cattlebody -es -a`    |
| **Underfitting**     | Both train/val acc low    | `python main.py train -m faster_rcnn -d cattlebody -e 300 -lr 0.005`   |
| **Slow Convergence** | mAP plateaued early       | `python main.py train -m faster_rcnn -d cattlebody -lr 0.002 -b 4`     |
| **Memory Issues**    | CUDA out of memory        | `python main.py train -m faster_rcnn -d cattlebody -b 2 --device auto` |
| **Poor Precision**   | High false positives      | `python main.py evaluate -m faster_rcnn -d cattlebody -t 0.7`          |
| **Poor Recall**      | Missing detections        | `python main.py evaluate -m faster_rcnn -d cattlebody -t 0.3`          |

### Expected Performance Timeline

Based on your current 67.44% mAP, here's what to expect:

```
Week 1: Quick Improvements (67% → 72%)
• python main.py train -m faster_rcnn -d cattlebody -e 200 -b 4 -lr 0.002
• Expected: +5% mAP improvement

Week 2: Advanced Optimization (72% → 78%)
• python main.py train-advanced -m faster_rcnn -d cattlebody -pr high_precision -a -es
• Expected: +6% mAP improvement

Week 3: Fine-tuning (78% → 80%+)
• python main.py optimize -m faster_rcnn -d cattlebody -tr 10
• Expected: +2-3% mAP improvement to reach target
```

## 🎨 Visualization and Inference

```bash
# Visualize results
python main.py visualize -m faster_rcnn -d cattlebody

# Run inference on images
python main.py visualize -m faster_rcnn --input path/to/images/
```

## 🔍 Advanced Usage

### Batch Processing

```bash
# Train all models on all datasets
python src/scripts/train_all.py

# Evaluate all models
python src/scripts/evaluate_all.py
```

### Custom Dataset Processing

```bash
# Process new dataset
python main.py preprocess
    --dataset custom_dataset
    --split-ratio 0.8
    --images-dir path/to/images
    --labels-dir path/to/labels
```

### Multi-GPU Training

```bash
# Use specific GPU
python main.py train -m faster_rcnn -d cattlebody --device cuda:0

# Automatic GPU selection
python main.py train -m faster_rcnn -d cattlebody --device auto
```

## Development

### Adding New Models

1. Create model architecture in `src/models/new_model.py`
2. Create training script in `src/training/train_new_model.py`
3. Add configuration to `src/config/settings.py`:

```python
TRAINING_CONFIGS = {
    # ... existing configs
    "new_model": {
        "name": "New Model",
        "description": "Description of new model",
        "module": "src.training.train_new_model",
    }
}
```

4. The system will automatically recognize it!

### Adding New Datasets

1. Place dataset in `dataset/new_dataset/`
2. Update preprocessing script if needed
3. Run preprocessing:

```bash
python main.py preprocess --dataset new_dataset
```

## 🤝 Contributing

1. Keep all source code in `src/`
2. All outputs follow the pattern: `outputs/{dataset}/{model}/{type}/`
3. Update configurations in `src/config/`
4. Add comprehensive logging
5. Include evaluation metrics

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### 🛡️ **CUDA Error Prevention (NEW)**

#### **CUDA Device-Side Assert Errors**

If you encounter CUDA errors like `"block: [0,0,0], thread: [24,0,0] Assertion 't >= 0 && t < n_classes' failed"`:

```bash
# 1. IMMEDIATELY run diagnostics to identify the issue
python main.py debug --dataset-path your/dataset/path --validate-dataset

# 2. Check the diagnostic output:
#    - Look for "Recommended num_classes: X"
#    - Check "Label range validation" results
#    - Verify "Model compatibility testing"

# 3. Fix common issues:
# Issue: Wrong number of classes
python main.py train --dataset-path your/dataset --num-classes 3  # Use recommended value

# Issue: Invalid label ranges
# The diagnostic will show which labels are out of range - fix your dataset

# Issue: Background class problems
# System automatically handles +1 offset for background class
```

#### **Prevention Strategy**

```bash
# ALWAYS run this before ANY training:
python main.py debug --dataset-path your/dataset --validate-dataset

# Only train if you see: "✅ No critical issues found!"
```

#### **Diagnostic Output Interpretation**

```bash
✅ "No critical issues found!" → Safe to train
⚠️  "Label range issues detected" → Fix dataset labels first
❌ "CUDA error: device-side assert triggered" → Dataset has invalid labels
📊 "Recommended num_classes: 3" → Use this value in training
```

### Common Issues

#### Import Errors

```bash
# Make sure you're in the project root
cd /path/to/project1
python main.py
```

#### Memory Issues

```bash
# Reduce batch size
python main.py train -m faster_rcnn -d cattlebody -b 2

# Use CPU if GPU memory is insufficient
python main.py train -m faster_rcnn -d cattlebody --device cpu
```

#### Dataset Issues

```bash
# Validate dataset structure
python src/utils/data_validation.py --dataset cattlebody

# Reprocess dataset
python main.py preprocess --dataset cattlebody --force-reprocess
```

#### Model Loading Issues

```bash
# Check if model exists
ls outputs/cattlebody/faster_rcnn/models/

# Retrain if model is corrupted
python main.py train -m faster_rcnn -d cattlebody -e 1
```

### Getting Help

1. **Check logs**: Look in `outputs/{dataset}/{model}/logs/`
2. **Run debug tests**: `python main.py debug`
3. **Validate data**: `python src/utils/data_validation.py`
4. **Check memory**: `python src/utils/memory.py`
5. **Verify paths**: `python main.py --show-structure`

### Performance Tips

1. **GPU Training**: Always use `--device cuda` for faster training
2. **Batch Size**: Adjust based on GPU memory (start with 8, reduce if needed)
3. **Data Loading**: Use SSD storage for faster data loading
4. **Mixed Precision**: Automatically enabled for compatible GPUs
5. **Monitoring**: Check `outputs/{dataset}/{model}/metrics/` for progress

---

**Happy Training! 🐄🤖**

## 📚 Quick Reference Commands

### 🚀 **Robust & CUDA-Safe Commands (Recommended)**

```bash
# Complete robust workflow (works anywhere, prevents CUDA errors)
python main.py debug --dataset-path dataset/cattle --validate-dataset
python main.py train --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -e 50 -b 4
python main.py evaluate --dataset-path dataset/cattle -m faster_rcnn --validate-dataset

# Your standard quick training
python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# Advanced robust training (production-ready)
python main.py train-advanced --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -pr high_precision -a -es

# Multi-model training
python main.py train -m faster_rcnn -d cattle -e 20 -b 4 --device cuda:1
python main.py train -m yolov8 -d cattle -e 20 -b 8 --device cuda:0
python main.py train -m ultralytics -d cattlebody -e 20 -b 16 --device cuda

# Machine-portable commands (work on any machine)
python main.py debug --dataset-path /absolute/path/to/data --validate-dataset
python main.py train --dataset-path /absolute/path/to/data -m faster_rcnn --validate-dataset
```

### ⚡ **Quick Commands for Different Use Cases**

```bash
# 🔍 DEBUGGING: Always run first to prevent CUDA errors
python main.py debug --dataset-path dataset/cattle --validate-dataset

# 🧪 EXPERIMENTATION: Quick test training
python main.py train -m faster_rcnn -d cattle -e 5 -b 2 --device cuda:1

# 🎯 PRODUCTION: High-quality training
python main.py train-advanced --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -pr high_precision -a -es

# 📊 EVALUATION: Comprehensive analysis
python main.py evaluate --dataset-path dataset/cattle -m faster_rcnn --validate-dataset -t 0.5

# 🚀 OPTIMIZATION: Find best hyperparameters
python main.py optimize --dataset-path dataset/cattle -m faster_rcnn -tr 10


python main.py train -m yolov8 -d cattle -e 100 -b 8 --device cuda:0
```

```
`````
fusion model train

python -m src.training.train_fusion --batch_size 4 --resume "C:\Users\ASUS\Desktop\project 1\model_checkpoints\cattle_best_model.pth"

python -m src.training.train_fusion --batch_size 4 --resume "/home/john/coding/cattlebiometric/dataset/cattle/cattle_best_model.pth"
