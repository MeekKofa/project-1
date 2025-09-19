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

## � Command Reference

The system provides several commands with both short and long aliases for convenience:

### 📋 All Available Commands

| Command          | Description                 | Key Features                                          |
| ---------------- | --------------------------- | ----------------------------------------------------- |
| `train`          | Train a model               | Basic training with customizable parameters           |
| `train-advanced` | Advanced training           | Optimized profiles, augmentation, early stopping      |
| `evaluate`       | Evaluate a trained model    | Comprehensive metrics and analysis                    |
| `preprocess`     | Preprocess datasets         | Split data into train/val/test sets                   |
| `optimize`       | Hyperparameter optimization | Automated parameter tuning                            |
| `cleanup`        | Clean old metric files      | Remove individual epoch files, keep consolidated data |
| `info`           | Show system information     | List models, datasets, and project structure          |

### 🏷️ Argument Aliases (Short Forms)

For faster command entry, you can use short aliases:

| Long Form           | Short | Description               | Example Values                               |
| ------------------- | ----- | ------------------------- | -------------------------------------------- |
| `--model`           | `-m`  | Model architecture        | `faster_rcnn`, `yolov8`, `ultralytics`       |
| `--dataset`         | `-d`  | Dataset name              | `cattlebody`, `cattleface`                   |
| `--epochs`          | `-e`  | Number of training epochs | `50`, `100`, `200`                           |
| `--batch-size`      | `-b`  | Training batch size       | `2`, `4`, `8`, `16`                          |
| `--learning-rate`   | `-lr` | Learning rate             | `0.001`, `0.002`, `0.01`                     |
| `--score-threshold` | `-t`  | Confidence threshold      | `0.5`, `0.7`, `0.8`                          |
| `--output-dir`      | `-o`  | Output directory          | `./custom_output/`                           |
| `--model-path`      | `-p`  | Specific model path       | `outputs/model.pth`                          |
| `--split-ratio`     | `-s`  | Train/val split ratio     | `0.7`, `0.8`, `0.9`                          |
| `--force`           | `-f`  | Force operation           | Used with cleanup/preprocess                 |
| `--profile`         | `-pr` | Training profile          | `default`, `high_precision`, `fast_training` |
| `--trials`          | `-tr` | Optimization trials       | `10`, `20`, `50`                             |
| `--max-epochs`      | `-me` | Maximum epochs            | `100`, `200`, `300`                          |
| `--augmentation`    | `-a`  | Enable data augmentation  | Boolean flag                                 |
| `--early-stopping`  | `-es` | Enable early stopping     | Boolean flag                                 |

### 📝 Command Examples with Explanations

#### 1. **Train Command** - `python main.py train`

Basic training with customizable parameters:

```bash
# Basic training (uses default parameters)
python main.py train -m faster_rcnn -d cattlebody

# Customized training with explanations:
python main.py train \
    -m faster_rcnn \          # Use Faster R-CNN model
    -d cattlebody \           # Train on cattle body dataset
    -e 100 \                  # Train for 100 epochs
    -b 4 \                    # Use batch size of 4
    -lr 0.002 \               # Set learning rate to 0.002
    --device cuda             # Use GPU for training
```

#### 2. **Train-Advanced Command** - `python main.py train-advanced`

Advanced training with optimization features:

```bash
# High precision training (recommended for best results)
python main.py train-advanced \
    -m faster_rcnn \          # Use Faster R-CNN model
    -d cattlebody \           # Train on cattle body dataset
    -pr high_precision \      # Use high precision profile
    -a \                      # Enable data augmentation
    -es \                     # Enable early stopping
    -me 300                   # Maximum 300 epochs

# Quick training for experiments
python main.py train-advanced -m faster_rcnn -d cattlebody -pr fast_training -me 50
```

#### 3. **Evaluate Command** - `python main.py evaluate`

Comprehensive model evaluation:

```bash
# Basic evaluation
python main.py evaluate -m faster_rcnn -d cattlebody

# Custom evaluation with specific parameters
python main.py evaluate \
    -m faster_rcnn \          # Evaluate Faster R-CNN model
    -d cattlebody \           # Use cattle body test set
    -b 4 \                    # Batch size 4 for evaluation
    -t 0.7 \                  # Confidence threshold 0.7
    -o ./custom_eval/         # Save results to custom directory
```

#### 4. **Preprocess Command** - `python main.py preprocess`

Data preprocessing and splitting:

```bash
# Basic preprocessing (80% train, 20% val)
python main.py preprocess -d cattlebody

# Custom split ratio
python main.py preprocess \
    -d cattlebody \           # Process cattle body dataset
    -s 0.7 \                  # 70% for training, 30% for val/test
    -f                        # Force reprocess even if exists
```

#### 5. **Optimize Command** - `python main.py optimize`

Hyperparameter optimization:

```bash
# Basic hyperparameter optimization
python main.py optimize -m faster_rcnn -d cattlebody

# Advanced optimization
python main.py optimize \
    -m faster_rcnn \          # Optimize Faster R-CNN
    -d cattlebody \           # Use cattle body dataset
    -pr high_precision \      # Use high precision profile
    -tr 20 \                  # Run 20 optimization trials
    -me 100                   # Max 100 epochs per trial
```

#### 6. **Cleanup Command** - `python main.py cleanup`

Clean up old metric files:

```bash
# Clean all metrics directories (dry run first)
python main.py cleanup --all --dry-run

# Actually clean up (removes individual epoch files)
python main.py cleanup --all -f

# Clean specific directory
python main.py cleanup --dir outputs/cattlebody/faster_rcnn/metrics/
```

#### 7. **Info Command** - `python main.py info`

System information and help:

```bash
# List available models and datasets
python main.py info -l

# Show project structure
python main.py info -s

# Show both
python main.py info -l -s
```

### 🎯 Recommended Workflows

#### **For Best Results (Recommended)**

```bash
# 1. Preprocess data
python main.py preprocess -d cattlebody -s 0.8

# 2. Train with advanced features
python main.py train-advanced -m faster_rcnn -d cattlebody -pr high_precision -a -es -me 300

# 3. Evaluate the model
python main.py evaluate -m faster_rcnn -d cattlebody -t 0.5

# 4. Clean up old files
python main.py cleanup --all -f
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

### Training Commands

#### Faster R-CNN

```bash
# Basic training
python main.py train -m faster_rcnn -d cattlebody

# Full training with custom parameters
python main.py train -m faster_rcnn -d cattlebody -e 50 -b 8 -lr 0.001 --device cuda

# Quick test training
python main.py train -m faster_rcnn -d cattlebody -e 1 -b 2
```

#### YOLOv8

```bash
# Basic training
python main.py train -m yolov8 -d cattlebody

# Advanced training
python main.py train -m yolov8 -d cattlebody -e 100 -b 16 -lr 0.01 --device cuda

# Multiple datasets
python main.py train -m yolov8 -d cattleface -e 50
```

#### Ultralytics YOLO

```bash
# Basic training
python main.py train -m ultralytics -d cattlebody

# Production training
python main.py train -m ultralytics -d cattlebody -e 200 -b 32 -lr 0.01 --device cuda
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

### Comprehensive Evaluation

```bash
# Evaluate specific model on specific dataset
python main.py evaluate -m faster_rcnn -d cattlebody

# Evaluate with custom parameters
python main.py evaluate -m faster_rcnn -d cattlebody -b 4 -t 0.7

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

## 📋 Training Recipes

### Quick Prototyping

```bash
# Fast iteration for development
python main.py train -m faster_rcnn -d cattlebody -e 1 -b 2
```

### Production Training

```bash
# High-quality model training
python main.py train -m faster_rcnn -d cattlebody -e 100 -b 16 -lr 0.001

python main.py train -m yolov8 -d cattle -e 2 -b 8 --device cuda:1
```

    --device cuda
````

### Comparative Training

```bash
# Train same dataset with different models
python main.py train -m faster_rcnn -d cattlebody -e 50
python main.py train -m yolov8 -d cattlebody -e 50
python main.py train -m ultralytics -d cattlebody -e 50

# Compare results
python main.py evaluate -m faster_rcnn -d cattlebody
python main.py evaluate -m yolov8 -d cattlebody
python main.py evaluate -m ultralytics -d cattlebody
```

### 🎯 Short Argument Aliases

For faster CLI usage, the system supports short aliases for common arguments:

| Long Form           | Short Form | Description          |
| ------------------- | ---------- | -------------------- |
| `--model`           | `-m`       | Model selection      |
| `--dataset`         | `-d`       | Dataset selection    |
| `--epochs`          | `-e`       | Number of epochs     |
| `--batch-size`      | `-b`       | Batch size           |
| `--learning-rate`   | `-lr`      | Learning rate        |
| `--score-threshold` | `-t`       | Confidence threshold |
| `--output-dir`      | `-o`       | Output directory     |
| `--model-path`      | `-p`       | Model path           |
| `--split-ratio`     | `-s`       | Dataset split ratio  |
| `--force`           | `-f`       | Force processing     |
| `--profile`         | `-pr`      | Training profile     |
| `--trials`          | `-tr`      | Optimization trials  |
| `--max-epochs`      | `-me`      | Maximum epochs       |
| `--augmentation`    | `-a`       | Enable augmentation  |
| `--early-stopping`  | `-es`      | Early stopping       |

**Examples:**

```bash
# Short form - much faster to type
python main.py train -m faster_rcnn -d cattlebody -e 50 -b 8 -lr 0.001

# Long form - more explicit
python main.py train --model faster_rcnn --dataset cattlebody --epochs 50 --batch-size 8 --learning-rate 0.001

# Mixed usage (both work)
python main.py train -m faster_rcnn --dataset cattlebody -e 50 --batch-size 8 -lr 0.001
```

## 🛠️ Configuration

### Hyperparameter Files

- **Faster R-CNN**: `src/config/hyperparameters.py` → `FASTER_RCNN_PARAMS`
- **YOLOv8**: `src/config/hyperparameters.py` → `YOLOV8_PARAMS`
- **Ultralytics**: `src/config/hyperparameters.py` → `ULTRALYTICS_PARAMS`

### Output Configuration

```python
# Systematic output structure: outputs/{dataset}/{model}/{type}/
get_systematic_output_dir("cattlebody", "faster_rcnn", "models")
# Returns: outputs/cattlebody/faster_rcnn/models/
```

## 📝 Logging

All activities are comprehensively logged:

```
outputs/{dataset}/{model}/logs/{model}_{dataset}.log
```

Example log locations:

- `outputs/cattlebody/faster_rcnn/logs/faster_rcnn_cattlebody.log`
- `outputs/cattleface/yolov8/logs/yolov8_cattleface.log`

## 🐛 Debug and Testing

### Debug Commands

```bash
# Run debug tests
python main.py debug

# Debug specific components
python src/utils/data_validation.py
python src/utils/model_validation.py
```

### Memory Management

```bash
# Monitor GPU memory during training
python main.py train --model faster_rcnn --dataset cattlebody --monitor-memory

# Force garbage collection
python main.py train --model faster_rcnn --dataset cattlebody --clean-memory
```

## 📊 Performance Benchmarks

### Expected Training Times (RTX 3080)

| Model        | Dataset    | Epochs | Batch Size | Time per Epoch | Total Time |
| ------------ | ---------- | ------ | ---------- | -------------- | ---------- |
| Faster R-CNN | cattlebody | 50     | 8          | ~4 min         | ~3.3 hours |
| YOLOv8       | cattlebody | 100    | 16         | ~2 min         | ~3.3 hours |
| Ultralytics  | cattlebody | 200    | 32         | ~1 min         | ~3.3 hours |

### Expected Performance Metrics

| Model        | Dataset    | mAP@0.5    | mAP@0.5:0.95 | FPS     |
| ------------ | ---------- | ---------- | ------------ | ------- |
| Faster R-CNN | cattlebody | ~0.75-0.85 | ~0.45-0.55   | ~15-25  |
| YOLOv8       | cattlebody | ~0.70-0.80 | ~0.40-0.50   | ~50-100 |
| Ultralytics  | cattlebody | ~0.72-0.82 | ~0.42-0.52   | ~60-120 |

## 🔧 Development

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

```bash
# Complete workflow
python main.py preprocess -d cattlebody
python main.py train -m faster_rcnn -d cattlebody -e 50
python main.py evaluate -m faster_rcnn -d cattlebody
python main.py visualize -m faster_rcnn -d cattlebody

# All models on one dataset
python main.py train -m faster_rcnn -d cattlebody -e 20
python main.py train -m yolov8 -d cattlebody -e 20
python main.py train -m ultralytics -d cattlebody -e 20

# One model on all datasets
python main.py train -m faster_rcnn -d cattlebody -e 20
python main.py train -m faster_rcnn -d cattleface -e 20

# Production ready
python main.py train -m faster_rcnn -d cattlebody -e 100 -b 16

python main.py train -m yolov8 -d cattle -e 2 -b 8 --device cuda:1
```

```

```
