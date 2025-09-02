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
│   └── cattleface/        # Cattle face dataset
│       ├── CowfaceImage/  # Original images
│       └── Annotation/    # Annotation files
├── processed_data/        # 🔄 Processed dataset
│   └── cattleface/        # Train/Val/Test splits
│       ├── train/
│       ├── val/
│       └── test/
├──
├── src/                   # 💻 Source code
│   ├── config.py          # Main configuration
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
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── utils/             # 🛠️ Utility functions
│   │   ├── data_validation.py
│   │   ├── logging_utils.py
│   │   ├── model_validation.py
│   │   └── ... (many more utilities)
│   ├── scripts/           # 📜 Additional scripts
│   │   ├── train_all.py
│   │   ├── evaluate_all.py
│   │   └── inference.py
│   └── examples/          # 💡 Example usage
│       └── test_model.py
├──
└── outputs/               # 📤 All outputs organized here
    ├── models/            # Trained model weights
    │   ├── faster_rcnn/
    │   ├── yolov8/
    │   └── ultralytics/
    ├── logs/              # Training and execution logs
    ├── images/            # Generated images/visualizations
    └── results/           # Training results and metrics
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py

python main.py train --model faster_rcnn --dataset cattlebody --epochs 10 --batch-size 8
```

This will launch an interactive menu where you can:

- ✅ Train different models (Faster R-CNN, YOLOv8, Ultralytics)
- ✅ Run model evaluation
- ✅ Visualize results
- ✅ Run debug tests
- ✅ View project structure

## 🎯 Available Training Options

### 1. Faster R-CNN

- **Description**: Faster R-CNN with ResNet-50 backbone for object detection
- **Module**: `src/training/train_faster_rcnn.py`
- **Output**: `outputs/models/faster_rcnn/`

### 2. YOLOv8

- **Description**: YOLOv8 model for real-time object detection
- **Module**: `src/training/train_yolov8.py`
- **Output**: `outputs/models/yolov8/`

### 3. Ultralytics YOLO

- **Description**: Ultralytics YOLO implementation
- **Module**: `src/training/train_ultralytics.py`
- **Output**: `outputs/models/ultralytics/`

## 📊 Dataset

The project works with cattle face detection datasets:

- **Location**: `dataset/cattleface/`
- **Processed Data**: `processed_data/cattleface/`
- **Splits**: train/val/test

## 🎛️ Configuration

All paths and settings are centralized in `src/config.py`:

- Output directories
- Dataset paths
- Training configurations
- Hyperparameters

## 📝 Logging

All activities are logged:

- **Main Log**: `outputs/logs/main.log`
- **Training Logs**: `outputs/logs/{model_name}.log`
- **Evaluation Logs**: `outputs/logs/evaluation.log`

## 🔧 Development

### Adding New Models

1. Create model architecture in `src/models/`
2. Create training script in `src/training/`
3. Add configuration to `src/config.py`
4. The main menu will automatically include it

### Customizing Outputs

All outputs are saved to the `outputs/` directory:

- Models: `outputs/models/`
- Logs: `outputs/logs/`
- Images: `outputs/images/`
- Results: `outputs/results/`

## 🐛 Debug and Testing

Run debug tests through the main menu or directly:

```bash
python src/run_debug_sample.py
```

## 📈 Evaluation

Evaluate trained models through the main menu or directly:

```bash
python src/evaluation/evaluate.py
```

## 🎨 Visualization

Generate visualizations through the main menu or directly:

```bash
python src/scripts/inference.py
```

## 🤝 Contributing

1. Keep all source code in `src/`
2. All outputs go to `outputs/`
3. Update `src/config.py` for new paths
4. Add new training options to `TRAINING_CONFIGS`

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure you're running from the project root
2. **Permission Errors**: Check file permissions in the project directory
3. **Missing Dependencies**: Run `pip install -r requirements.txt`
4. **Path Issues**: All paths are configured in `src/config.py`

### Getting Help:

- Check logs in `outputs/logs/`
- Run debug tests via main menu
- Verify project structure via main menu

---

**Happy Training! 🐄🤖**
