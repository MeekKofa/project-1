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

## 📊 Data Processing

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
````

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
```
````
