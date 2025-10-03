# 🏗️ New Architecture - Visual Overview

## 📊 System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CLI Arguments          Config Files          Quick Start Script │
│  ──────────────         ───────────           ──────────────── │
│  --model yolov8         yolov8.yaml           ./quick_start.sh  │
│  --optimizer adamw      high_perf.yaml        ↓                  │
│  --scheduler cosine     quick_test.yaml       Interactive Menu   │
│  --augment              custom.yaml           ↓                  │
│  --epochs 100           ...                   Generated Command  │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING CONFIGURATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  TrainingConfig (src/config/training_config.py)                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Parses CLI arguments                                  │   │
│  │  • Loads config files (YAML/JSON)                        │   │
│  │  • Merges configs (file + CLI overrides)                 │   │
│  │  • Validates all parameters                              │   │
│  │  • Auto-configures dataset paths                         │   │
│  │  • 60+ configurable parameters                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL TRAINER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  train_universal.py (src/scripts/train_universal.py)             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Setup logging and output directories                 │   │
│  │  2. Create model from registry                           │   │
│  │  3. Create dataloaders                                   │   │
│  │  4. Create optimizer (any type!)                         │   │
│  │  5. Create scheduler (any type!)                         │   │
│  │  6. Initialize trainer                                   │   │
│  │  7. Run training loop                                    │   │
│  │  8. Save results and checkpoints                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────┬──────────────┬──────────────┬──────────────┬───────────────┘
       │              │              │              │
       ↓              ↓              ↓              ↓
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  MODEL   │   │ DATASET  │   │ OPTIMIZER│   │ SCHEDULER│
│ REGISTRY │   │ LOADING  │   │ CREATION │   │ CREATION │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CORE ABSTRACTIONS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Registry Pattern        Model Base         Trainer Base          │
│  ───────────────         ──────────         ────────────          │
│  ModelRegistry           DetectionModelBase TrainerBase           │
│  DatasetRegistry         ↓                  ↓                     │
│  LossRegistry            - forward()        - train_epoch()       │
│  ↓                       - compute_loss()   - validate()          │
│  @register decorator     - freeze_backbone()- save_checkpoint()  │
│  build() factory         - get_model_info() - load_checkpoint()  │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODULAR COMPONENTS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  YOLOv8 Model (src/models/yolov8/)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  architecture.py          heads.py                        │   │
│  │  ────────────────          ────────                       │   │
│  │  YOLOv8Backbone            BoxRegressionHead              │   │
│  │  YOLOv8Model               ClassificationHead             │   │
│  │  ConvBlock                 ObjectnessHead                 │   │
│  │  CSPBlock                  YOLOv8Head (combines all)      │   │
│  │  SPPF                                                      │   │
│  │                                                           │   │
│  │  loss.py                   config.py                      │   │
│  │  ────────                  ──────────                     │   │
│  │  FocalLoss                 get_default_config()           │   │
│  │  YOLOv8Loss                update_config()                │   │
│  │  - compute_box_loss()                                     │   │
│  │  - compute_cls_loss()                                     │   │
│  │  - compute_obj_loss()                                     │   │
│  │                                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Detection Dataset (src/data/detection_dataset.py)               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Universal dataset for all models                       │   │
│  │  • Supports YOLO/COCO/VOC formats                         │   │
│  │  • Configurable augmentation                              │   │
│  │  • Auto-validation                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Box Utilities (src/utils/box_utils.py)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • box_iou() - Compute IoU                                │   │
│  │  • sanitize_boxes() - Clean invalid boxes                 │   │
│  │  • box_cxcywh_to_xyxy() - Format conversion               │   │
│  │  • clip_boxes_to_image() - Clip to bounds                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  outputs/{experiment_name}/                                       │
│  ├── config.yaml             ← Saved configuration               │
│  ├── checkpoints/            ← Model checkpoints                 │
│  │   ├── best_model.pth                                          │
│  │   └── checkpoint_epoch_*.pth                                  │
│  ├── logs/                   ← Training logs                     │
│  │   └── training.log                                            │
│  ├── metrics/                ← Performance metrics               │
│  │   ├── training_history.json                                   │
│  │   └── epoch_metrics.csv                                       │
│  ├── predictions/            ← Validation predictions            │
│  └── visualizations/         ← Training curves, plots            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

```
1. USER INPUT
   ↓
   CLI args or Config file
   ↓

2. CONFIGURATION
   ↓
   TrainingConfig parses and validates
   ↓

3. INITIALIZATION
   ↓
   ┌─────────────────────────────────────┐
   │  Model ← Registry.build('yolov8')   │
   │  Dataset ← DetectionDataset(...)    │
   │  Optimizer ← create_optimizer(...)  │
   │  Scheduler ← create_scheduler(...)  │
   └─────────────────────────────────────┘
   ↓

4. TRAINING LOOP
   ↓
   for epoch in range(num_epochs):
       ┌───────────────────────────────┐
       │  Forward pass                 │
       │  ↓                           │
       │  Compute loss                │
       │  ↓                           │
       │  Backward pass               │
       │  ↓                           │
       │  Optimizer step              │
       │  ↓                           │
       │  Scheduler step              │
       │  ↓                           │
       │  Validation (if interval)    │
       │  ↓                           │
       │  Save checkpoint (if interval)│
       └───────────────────────────────┘
   ↓

5. OUTPUT
   ↓
   Save final model, metrics, logs
```

## 🧩 Component Interaction

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  TrainingConfig                                            │
│  ─────────────                                             │
│  Contains ALL parameters                                   │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Model      │  │   Dataset    │  │   Training   │   │
│  │   Config     │  │   Config     │  │   Config     │   │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤   │
│  │ - model      │  │ - dataset_   │  │ - epochs     │   │
│  │ - num_       │  │   root       │  │ - batch_size │   │
│  │   classes    │  │ - image_size │  │ - learning_  │   │
│  │ - pretrained │  │ - augment    │  │   rate       │   │
│  │ - freeze     │  │ - label_     │  │ - optimizer  │   │
│  │              │  │   format     │  │ - scheduler  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                            │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Universal Trainer (train_universal.py)                    │
│  ───────────────────────────────────                       │
│                                                            │
│  def create_optimizer(model, config):                      │
│      if config['optimizer'] == 'adamw':                    │
│          return torch.optim.AdamW(...)                     │
│      elif config['optimizer'] == 'sgd':                    │
│          return torch.optim.SGD(...)                       │
│      ...                                                   │
│                                                            │
│  def create_scheduler(optimizer, config):                  │
│      if config['scheduler'] == 'cosine':                   │
│          return CosineAnnealingLR(...)                     │
│      elif config['scheduler'] == 'step':                   │
│          return StepLR(...)                                │
│      ...                                                   │
│                                                            │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Model Registry                                            │
│  ─────────────                                             │
│                                                            │
│  Registered Models:                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ yolov8   │  │ faster_  │  │ retina   │               │
│  │          │  │ rcnn     │  │ net      │               │
│  │ @register│  │ @register│  │ @register│               │
│  └──────────┘  └──────────┘  └──────────┘               │
│                                                            │
│  model = Registry.build('yolov8', num_classes=2)          │
│          ↓                                                 │
│  YOLOv8Model(num_classes=2)                               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 🎯 Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Configuration Priority (highest to lowest):            │
│                                                         │
│  1. Command-line arguments                              │
│     ↓ (overrides)                                      │
│  2. Config file (YAML/JSON)                             │
│     ↓ (overrides)                                      │
│  3. Default values                                      │
│                                                         │
│  Example:                                               │
│  ┌────────────────────────────────────────────────┐   │
│  │  Default:    learning_rate: 0.001               │   │
│  │  Config file: learning_rate: 0.002              │   │
│  │  CLI arg:     --learning-rate 0.005             │   │
│  │  ─────────────────────────────────────────────  │   │
│  │  Result:      learning_rate: 0.005 ✓            │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Extensibility

```
Adding a New Model:
┌────────────────────────────────────────────────┐
│                                                │
│  1. Create model file:                         │
│     src/models/new_model/architecture.py       │
│                                                │
│  2. Inherit from base:                         │
│     class NewModel(DetectionModelBase):        │
│         ...                                    │
│                                                │
│  3. Register model:                            │
│     @ModelRegistry.register('new_model')       │
│     class NewModel(DetectionModelBase):        │
│         ...                                    │
│                                                │
│  4. Use immediately:                           │
│     python train_universal.py \                │
│         --model new_model \                    │
│         --dataset-root dataset/...             │
│                                                │
│  ✅ Done! No other changes needed!             │
│                                                │
└────────────────────────────────────────────────┘


Adding a New Optimizer:
┌────────────────────────────────────────────────┐
│                                                │
│  1. Edit create_optimizer() function:          │
│     elif optimizer_type == 'new_opt':          │
│         return NewOptimizer(...)               │
│                                                │
│  2. Use immediately:                           │
│     python train_universal.py \                │
│         --optimizer new_opt \                  │
│         --optimizer-params '{...}'             │
│                                                │
│  ✅ Done! No other changes needed!             │
│                                                │
└────────────────────────────────────────────────┘
```

## 📊 Comparison: Old vs New Architecture

```
OLD SYSTEM:
┌──────────────────────────────────────────┐
│  train_yolov8.py (724 lines)             │
│  ┌────────────────────────────────────┐ │
│  │  EVERYTHING HARDCODED:             │ │
│  │  - Dataset paths                   │ │
│  │  - Hyperparameters                 │ │
│  │  - Optimizer (Adam only)           │ │
│  │  - No scheduler                    │ │
│  │  - Training loop                   │ │
│  │  - Validation loop                 │ │
│  │  - Checkpoint saving               │ │
│  │  - ...                             │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ❌ Monolithic                           │
│  ❌ Hardcoded                            │
│  ❌ Not reusable                         │
│  ❌ Hard to test                         │
│  ❌ Hard to extend                       │
└──────────────────────────────────────────┘


NEW SYSTEM:
┌──────────────────────────────────────────────────────────┐
│  Modular Components:                                      │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Registry │  │   Model  │  │  Dataset │  │ Trainer │ │
│  │  Pattern │  │   Base   │  │   Base   │  │  Base   │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│       ↓              ↓              ↓            ↓      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ YOLOv8   │  │ Detection│  │ Universal│  │ Config  │ │
│  │ Model    │  │ Dataset  │  │ Trainer  │  │ System  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│                                                           │
│  ✅ Modular                                               │
│  ✅ Configurable                                          │
│  ✅ Reusable                                              │
│  ✅ Testable                                              │
│  ✅ Extensible                                            │
└──────────────────────────────────────────────────────────┘
```

---

**This architecture follows:**

- ✅ **SOLID Principles** - Single responsibility, Open/closed, etc.
- ✅ **DRY Principle** - Don't Repeat Yourself
- ✅ **Clean Architecture** - Separation of concerns
- ✅ **Design Patterns** - Registry, Factory, Template Method
- ✅ **Zero Hardcoding** - Everything configurable

**Result: Professional, production-ready, maintainable code! 🚀**
