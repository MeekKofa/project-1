"""
Test script to validate cattleface dataset loading and visualization
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.loaders.cattle_loader import CattleDetectionDataset
from src.utils.visualization import visualize_detection
from src.config.face_dataset_config import FaceDatasetConfig
from src.loaders.detection_transforms import Compose, ToTensor

def get_transform():
    """Get basic transforms for dataset."""
    return Compose([
        ToTensor(),  # Convert PIL Image to tensor while preserving target
    ])

def visualize_samples(dataset, num_samples=4):
    """Visualize random samples from dataset"""
    plt.figure(figsize=(16, 16))
    
    for i in range(num_samples):
        plt.subplot(2, 2, i+1)
        
        # Get random sample
        idx = torch.randint(len(dataset), (1,)).item()
        image, target = dataset[idx]
        
        # Visualize detection
        visualize_detection(
            image=image,
            boxes=target['boxes'],
            labels=target['labels'],
            score_threshold=0.0  # Show all boxes since these are ground truth
        )
        plt.title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Define paths
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        dataset_root = project_root / "dataset" / "cattleface"
        
        print(f"Loading dataset from: {dataset_root}")
        
        # Initialize and analyze dataset configuration
        config = FaceDatasetConfig(
            dataset_root=str(dataset_root),
            image_dir="CowfaceImage",
            annotation_dir="Annotation"
        )
        
        # Print dataset summary
        config.print_summary()
        
        # Validate dataset
        issues = config.validate_dataset()
        if issues:
            print("\nWarning: Found the following issues:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("\nNo issues found in dataset configuration.")

        # Get class configuration from the yaml file
        import yaml
        config_path = project_root / "src" / "config" / "cattleface.yaml"
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Generate class names for all 384 classes - just using numeric IDs
        class_names = [str(i) for i in range(384)]

        # Load the dataset
        dataset = CattleDetectionDataset(
            root_dir=str(dataset_root),
            split='train',  # Use train as default since we have all images in one folder
            image_size=640,  # Default image size
            format="yolo",
            class_names=class_names,  # Explicit class names
            num_classes=384,  # Explicit number of classes
            image_dir="CowfaceImage",
            annotation_dir="Annotation",
            transform=get_transform()  # Add basic transforms for tensor conversion
        )
    
        print(f"\nSuccessfully loaded dataset with {len(dataset)} samples")
        
        # Print first few samples
        print("\nFirst few samples:")
        for i in range(min(3, len(dataset))):
            image, target = dataset[i]
            print(f"Sample {i}:")
            print(f"Image type: {type(image)}")
            print(f"Image shape: {image.shape}")
            print(f"Number of boxes: {len(target['boxes'])}")
            print(f"Labels type: {type(target['labels'])}")
            print(f"Boxes type: {type(target['boxes'])}")
            print(f"Labels: {target['labels'].tolist()}")
        
        # Visualize random samples
        print("\nVisualizing random samples...")
        visualize_samples(dataset)
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()