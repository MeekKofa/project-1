````markdown
# Cattle Detection & Recognition System

A comprehensive machine learning pipeline for cattle detection and recognition using various deep learning architectures including Faster R-CNN, YOLOv8, and Ultralytics YOLO.

## 🏗️ Project Structure

```
project1/
├── main.py                 # 🚀 Main entry point - START HERE
├── requirements.txt        # 📦 Python dependencies
├── README.md              # 📖 This file
├──
├── config/                # ⚙️ Configuration files
│   ├── cattle.yaml        # YAML configuration
│   ├── hyperparameters.py # Training hyperparameters
│   └── paths.py           # Legacy path configuration
├──
├── data/                  # 📁 Raw data (if any)
├── dataset/               # 📊 Dataset files
│   ├── cattleface/        # Cattle face dataset
│   │   ├── CowfaceImage/  # Original images
│   │   └── Annotation/    # Annotation files
│   └── cattlebody/        # Cattle body dataset
├── processed_data/        # 🔄 Processed dataset
│   ├── cattleface/        # Face detection splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── cattlebody/        # Body detection splits
│       ├── train/
│       ├── val/
│       └── test/
├──
├── src/                   # 💻 Source code
│   ├── config/            # Configuration modules
│   │   ├── __init__.py
│   │   ├── settings.py    # Main configuration
│   │   ├── paths.py       # Path configurations
│   │   └── hyperparameters.py # Training hyperparameters
│   ├── models/            # 🧠 Model architectures
│   │   ├── faster_rcnn.py
│   │   ├── yolov8.py
│   │   └── fusion_model.py
│   ├── training/          # 🎯 Training scripts
│   │   ├── train_faster_rcnn.py
│   │   ├── train_yolov8.py
│   │   ├── train_ultralytics.py
│   │   └── utils.py
│   ├── evaluation/        # 📈 Evaluation scripts
│   │   ├── evaluate_model.py
│   │   └── metrics.py
│   ├── processing/        # Data processing
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── utils/             # 🛠️ Utility functions
│   │   ├── data_validation.py
│   │   ├── logging_utils.py
│   │   ├── model_validation.py
│   │   └── memory.py
│   └── scripts/           # 📜 Additional scripts
│       ├── train_all.py
│       ├── evaluate_all.py
│       └── inference.py
├──
└── outputs/               # 📤 All outputs organized by dataset/model
    ├── {dataset}/         # Dataset-specific outputs
    │   └── {model}/       # Model-specific outputs
    │       ├── models/    # Trained model weights
    │       ├── logs/      # Training and execution logs
    │       ├── metrics/   # Evaluation metrics (JSON + TXT)
    │       ├── images/    # Generated images/visualizations
    │       ├── results/   # Training results
    │       └── checkpoints/ # Model checkpoints
    └── legacy/            # Legacy output structure
        ├── models/
        ├── logs/
        ├── images/
        └── results/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Interactive Menu

```bash
python main.py
```

This launches an interactive menu for all operations.

### 3. Robust Dataset Configuration (NEW!)

The system now supports both traditional dataset names and direct dataset paths for maximum flexibility:

```bash
# Traditional method (backward compatible)
python main.py train -m faster_rcnn -d cattlebody

# NEW: Robust method using direct paths (works anywhere!)
python main.py train -m faster_rcnn --dataset-path /path/to/any/dataset

# With comprehensive validation (recommended)
python main.py train -m faster_rcnn --dataset-path dataset/cattle --validate-dataset
```

### 4. CUDA Error Prevention

The system automatically prevents common CUDA device-side assert errors:

```bash
# Debug your dataset before training (highly recommended)
python main.py debug --dataset-path dataset/cattle --validate-dataset

# The system will automatically detect and fix:
# - Class count mismatches
# - Label range issues
# - Invalid bounding boxes
# - Dataset compatibility problems
```

## � Command Reference

The system provides several commands with both short and long aliases for convenience. **NEW**: All training commands now support robust dataset configuration!

### � **NEW: Robust Dataset Configuration**

The system now supports two modes for maximum flexibility and portability:

#### **Mode 1: Robust Path-Based (Recommended)**

```bash
# Works anywhere - just specify the dataset path directly
python main.py train -m faster_rcnn --dataset-path /absolute/path/to/dataset
python main.py train -m faster_rcnn --dataset-path ./relative/path/to/dataset
python main.py train -m faster_rcnn --dataset-path dataset/cattle

# With validation and overrides
python main.py train -m faster_rcnn --dataset-path dataset/cattle --validate-dataset --num-classes 3
```

#### **Mode 2: Traditional Name-Based (Backward Compatible)**

```bash
# Uses predefined dataset names (still works)
python main.py train -m faster_rcnn -d cattlebody
python main.py train -m faster_rcnn -d cattleface
python main.py train -m faster_rcnn -d cattle
```

### 🛡️ **CUDA Error Prevention System**

**Before training, always run diagnostics to prevent CUDA device-side assert errors:**

```bash
# Comprehensive dataset diagnostic (HIGHLY RECOMMENDED)
python main.py debug --dataset-path dataset/cattle --validate-dataset --sample-size 10

# Quick diagnostic check
python main.py debug --dataset-path dataset/cattle

# Debug with traditional dataset names
python main.py debug -d cattle --validate-dataset
```

**The diagnostic system checks for:**

- ✅ CUDA compatibility and GPU status
- ✅ Dataset structure and file integrity
- ✅ Label range validation (prevents assert errors)
- ✅ Class count analysis and recommendations
- ✅ Bounding box format validation
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

| Model         | Description                          | Best For                |
| ------------- | ------------------------------------ | ----------------------- |
| `faster_rcnn` | Faster R-CNN with ResNet-50 backbone | High accuracy detection |
| `yolov8`      | YOLOv8 model                         | Real-time detection     |
| `ultralytics` | Ultralytics YOLO implementation      | Balanced speed/accuracy |

### Training Examples

```bash
# Your standard training command
python main.py train -m faster_rcnn -d cattle -e 2 -b 2 --device cuda:1

# Different models
python main.py train -m yolov8 -d cattle -e 50 -b 8 --device cuda:0
python main.py train -m ultralytics -d cattlebody -e 100 -b 16 --device cuda

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
```
````

```

```
