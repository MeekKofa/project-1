import sys
import os
from pathlib import Path
from PIL import Image

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import traceback
from datetime import datetime
from src.loaders.cattle_loader import CattleDetectionDataset
from src.models.registry import get_model
from src.utils.visualization import visualize_detection
from src.config.smoke_test_config import config
import signal
from contextlib import contextmanager
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import threading
from contextlib import contextmanager
import time
import _thread

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Timeout context manager that works on Windows
    
    Args:
        seconds: Number of seconds before timeout occurs
    """
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutError(f"Timed out after {seconds} seconds!")
    finally:
        timer.cancel()

def run_smoke_test(timeout_seconds=60):
    """Run smoke test and save results
    
    Args:
        timeout_seconds: Maximum seconds to allow for the test (default: 60)
    """
    logger.info("Starting smoke test...")
    
    # Create output directory
    output_dir = Path("outputs/smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up error handling
    errors = []
    stage = "initialization"
    
    try:
        with timeout(timeout_seconds):
            # 1. Initialize dataset
            logger.info("Loading dataset...")
            stage = "dataset loading"
            
            # Use a tiny subset for smoke test
            # Custom transform that handles both image and target
            class DetectionTransform:
                def __init__(self, size):
                    self.size = size
                    # ImageNet normalization
                    self.normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                
                def __call__(self, image, target):
                    # Convert PIL Image to tensor
                    if isinstance(image, Image.Image):
                        image = transforms.ToTensor()(image)
                    
                    # Resize image
                    image = transforms.Resize((self.size, self.size))(image)
                    
                    # Normalize with ImageNet stats
                    image = self.normalize(image)
                    
                    # No modification to target
                    return image, target
            
            transform = DetectionTransform(size=800)  # Match Faster R-CNN's expected input size
            
            dataset = CattleDetectionDataset(
                root_dir="dataset/cattle",
                split="train",
                image_size=320,  # Smaller size for faster processing
                skip_analysis=True,  # Skip analysis for smoke test
                max_samples=10,  # Only use 10 images for smoke test
                transform=transform
            )
            
            # Create dataloader
            stage = "dataloader creation"
            logger.info("Creating dataloader...")
            def collate_fn(batch):
                # Separate images and targets
                images = [item[0] for item in batch]
                targets = [item[1] for item in batch]
                
                # Stack images
                images = torch.stack(images)
                
                return images, targets
                
            loader = DataLoader(
                dataset, 
                batch_size=1,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            # 2. Set up device and verify CUDA
            stage = "CUDA verification"
            logger.info("Setting up device...")
            
            # Check CUDA version and capabilities
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                cudnn_version = torch.backends.cudnn.version()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA version: {cuda_version}")
                logger.info(f"cuDNN version: {cudnn_version}")
                logger.info(f"GPU device: {device_name}")
                device = torch.device('cuda')
            else:
                logger.warning("CUDA not available, using CPU")
                device = torch.device('cpu')
            
            logger.info(f"Using device: {device}")
                
            # 3. Load model with detailed error tracking
            stage = "model loading"
            logger.info("Loading model...")
            
            # First verify imports
            stage = "module imports"
            try:
                # First try to create fusion model
                stage = "model creation"
                logger.info("Creating fusion model instance...")
                
                from src.models.fusion_model_simple import SimplerAdaptiveFusionDetector
                model = get_model('fusion_model', num_classes=2, config={})
                
                if model is None:
                    raise RuntimeError("Model creation returned None")
                    
                logger.info(f"Model created successfully: {type(model).__name__}")
                
                # Model verification
                model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
                logger.info(f"Model attributes: {model_attrs}")
                
                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"Model parameter count: {param_count:,}")
                
            except Exception as e:
                logger.warning(f"Failed to load fusion model: {str(e)}")
                logger.warning("Falling back to Faster R-CNN...")
                
                try:
                    from src.models.faster_rcnn.model import FasterRCNNModel
                    model = get_model('faster_rcnn', num_classes=2, config={})
                    if model is None:
                        raise RuntimeError("Fallback model creation returned None")
                    logger.info("Successfully loaded Faster R-CNN as fallback")
                except Exception as e:
                    error_msg = f"Failed to load fallback model: {str(e)}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    errors.append((stage, error_msg))
                    raise
            
            # Move model to device and set mode
            model = model.to(device)
            logger.info(f"Model moved to {device}")
            
            model.train()
            logger.info("Model set to training mode")
            
            logger.info(f"Model loaded successfully: {type(model).__name__}")
        
        # Get a batch
        logger.info("Running inference...")
        images, targets = next(iter(loader))
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        logger.info(f"Loaded batch: images shape {images.shape}")
        
        # Run inference
        with torch.no_grad():
            loss_dict = model(images, targets)
        
        # Log results
        logger.info("\nSmoke Test Results:")
        logger.info("-" * 50)
        logger.info("Model Device: %s", device)
        logger.info("Input Image Shape: %s", images[0].shape)
        logger.info("Number of Targets: %d", len(targets[0]['boxes']))
        
        logger.info("\nLoss Values:")
        for k, v in loss_dict.items():
            logger.info(f"{k}: {float(v):.4f}")
        
        # Save visualization
        img = images[0].cpu()
        boxes = targets[0]['boxes'].cpu()
        labels = targets[0]['labels'].cpu()
        
        output_path = output_dir / f"smoke_test_{datetime.now():%Y%m%d_%H%M%S}.png"
        visualize_detection(
            img, 
            boxes,
            labels,
            output_path=output_path
        )
        
        logger.info(f"\nVisualization saved to: {output_path}")
        logger.info("\n✓ Smoke test completed successfully!")
        
    except Exception as e:
        logger.error(f"Smoke test failed: {str(e)}", exc_info=True)
        raise

def main():
    try:
        run_smoke_test(timeout_seconds=60)
    except KeyboardInterrupt:
        logger.warning("\nSmoke test interrupted by user")
    except Exception as e:
        logger.error(f"Smoke test failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
