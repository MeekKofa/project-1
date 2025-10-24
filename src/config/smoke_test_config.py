"""Default configuration for smoke testing."""

config = {
    'smoke_test': {
        'max_iterations': 3,
        'timeout_seconds': 120,  # Increased timeout for first run
        'max_samples': 10,
        'image_size': 800,  # Standard size for Faster R-CNN
        'save_visualization': True,
        'verify_cuda': True
    },
    'models': {
        'fusion_model': {
            'num_classes': 2
        },
        'faster_rcnn': {
            'num_classes': 2
        }
    },
    'training': {
        'batch_size': 1,
        'learning_rate': 1e-4,
        'momentum': 0.9,
        'weight_decay': 1e-4
    }
}