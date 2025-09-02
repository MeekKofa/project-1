# Faster R-CNN training parameters
FASTER_RCNN_PARAMS = {
    'batch_size': 4,           # Small batch for 4GB GPU
    'accumulation_steps': 4,   # Effective batch size = 4
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'num_epochs': 100,
    'warmup_epochs': 2,
    'img_size': 384,
    'clip_grad_norm': 0.1,
    'step_size': 5,        # Added missing step_size
    'gamma': 0.1          # Added missing gamma
}

# YOLOv8 training parameters
YOLOV8_PARAMS = {
    'batch_size': 2,  # Small batch for 4GB GPU
    'accumulation_steps': 8,  # Effective batch = 16
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'input_size': 384,
    'dropout': 0.1,
    'warmup_epochs': 3
}

# Fusion parameters
FUSION_PARAMS = {
    "conf_thresh": 0.5,
    "iou_thresh": 0.5
}