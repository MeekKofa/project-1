"""
Default configuration values.
These are used as fallback when values are not specified in config.yaml or CLI.
"""

DEFAULTS = {
    # Training defaults
    'training': {
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'optimizer': 'adam',
        'num_workers': 4,
        'checkpoint_freq': 10,
        'log_freq': 10,
        'mixed_precision': False,
        'seed': 42,
    },

    # Device settings
    'device': {
        'type': 'cuda',  # cuda, cpu, mps
        'gpu_id': 0,
    },

    # Data settings
    'data': {
        'img_size': 640,
        'augmentation': True,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
    },

    # Model-specific defaults
    'models': {
        'faster_rcnn': {
            'backbone_type': 'resnet50',
            'pretrained': True,
            'config': {
                'anchor_sizes': ((32, 64, 128, 256, 512),),
                'anchor_aspect_ratios': ((0.5, 1.0, 2.0),),
                'rpn_pre_nms_top_n_train': 2000,
                'rpn_pre_nms_top_n_test': 1000,
                'rpn_post_nms_top_n_train': 2000,
                'rpn_post_nms_top_n_test': 1000,
                'rpn_nms_thresh': 0.7,
                'box_score_thresh': 0.05,
                'box_nms_thresh': 0.5,
            },
        },

        'yolov8': {
            # Note: backbone_type is set per model variant in registry
            # (yolov8_resnet uses 'resnet50', yolov8_csp uses 'csp')
            'pretrained': True,  # Use pretrained weights for ResNet backbones
            'in_channels': 3,    # RGB images
            'base_channels': 64,  # Base channels for CSP backbone
            # Detection thresholds (passed via config dict)
            'config': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_det': 300,
                'max_proposals': 3000,
            }
        },
    },

    # Loss settings
    'loss': {
        'classification_weight': 1.0,
        'bbox_weight': 1.0,
        'objectness_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    },

    # Optimizer settings
    'optimizer_params': {
        'adam': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        },
        'adamw': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        },
        'sgd': {
            'momentum': 0.9,
            'nesterov': True,
        },
    },

    # Learning rate scheduler
    'scheduler': {
        'type': 'cosine',  # 'cosine', 'step', 'exponential'
        'warmup_epochs': 3,
        'warmup_lr': 1e-6,
        'min_lr': 1e-6,
        # For step scheduler
        'step_size': 30,
        'gamma': 0.1,
    },

    # Evaluation settings
    'evaluation': {
        'iou_threshold': 0.5,
        'conf_threshold': 0.5,
        'save_predictions': True,
        'max_visualizations': 100,
    },

    # Output settings
    'output': {
        'base_dir': 'outputs',
        'save_best': True,
        'save_latest': True,
        'generate_visualizations': True,
    },

    # Logging settings
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'save_to_file': True,
    },
}
