"""
Dataset analysis utilities for cattle detection.
Provides tools for analyzing class distribution, box sizes, and data quality.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
import random
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

def analyze_class_distribution(label_paths: List[Path]) -> Dict[int, int]:
    """
    Analyze class distribution in the dataset.
    
    Args:
        label_paths: List of paths to label files
        
    Returns:
        Dictionary mapping class IDs to their counts
    """
    class_counts = {}
    total_boxes = 0
    
    for label_file in tqdm(label_paths, desc="Analyzing class distribution"):
        if not label_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            for line in f:
                try:
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_boxes += 1
                except (ValueError, IndexError):
                    logger.warning(f"Invalid label format in {label_file}")
                    continue
    
    # Log distribution
    logger.info(f"Total boxes: {total_boxes}")
    for class_id, count in class_counts.items():
        percentage = (count / total_boxes) * 100 if total_boxes > 0 else 0
        logger.info(f"Class {class_id}: {count} boxes ({percentage:.2f}%)")
        
    # Plot distribution
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Boxes')
    plt.savefig('class_distribution.png')
    plt.close()
    
    return class_counts

def analyze_box_sizes(label_paths: List[Path], plot: bool = True) -> Dict[str, float]:
    """
    Analyze bounding box size distribution.
    
    Args:
        label_paths: List of paths to label files
        plot: Whether to generate distribution plots
        
    Returns:
        Dictionary with size statistics
    """
    sizes = []
    for label_file in tqdm(label_paths, desc="Analyzing box sizes"):
        if not label_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    w, h = float(parts[3]), float(parts[4])
                    area = w * h
                    sizes.append(area)
                except (ValueError, IndexError):
                    logger.warning(f"Invalid box format in {label_file}")
                    continue
    
    if not sizes:
        logger.error("No valid boxes found for analysis")
        return {}
        
    sizes = np.array(sizes)
    stats = {
        'min_area': sizes.min(),
        'max_area': sizes.max(),
        'mean_area': sizes.mean(),
        'median_area': np.median(sizes),
        'std_dev': sizes.std()
    }
    
    logger.info("Box size statistics:")
    for key, value in stats.items():
        logger.info(f"- {key}: {value:.6f}")
        
    if plot:
        plt.figure(figsize=(10, 5))
        plt.hist(sizes, bins=50)
        plt.title('Box Size Distribution')
        plt.xlabel('Box Area (normalized)')
        plt.ylabel('Count')
        plt.savefig('box_size_distribution.png')
        plt.close()
        
    return stats

def validate_annotations(
    label_paths: List[Path],
    img_dir: Path,
    min_box_size: float = 0.0001
) -> Dict[str, int]:
    """
    Validate dataset annotations for common issues.
    
    Args:
        label_paths: List of paths to label files
        img_dir: Directory containing images
        min_box_size: Minimum allowed box area
        
    Returns:
        Dictionary with counts of various issues
    """
    issues = {
        'invalid_coords': 0,
        'zero_size': 0,
        'out_of_bounds': 0,
        'missing_images': 0,
        'too_small': 0,
        'invalid_format': 0
    }
    
    for label_file in tqdm(label_paths, desc="Validating annotations"):
        img_file = img_dir / label_file.stem.replace('_txt', '_jpg')
        
        if not img_file.with_suffix('.jpg').exists():
            issues['missing_images'] += 1
            logger.warning(f"Missing image for {label_file}")
            continue
            
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues['invalid_format'] += 1
                        logger.warning(f"Invalid format in {label_file}:{line_num}")
                        continue
                        
                    class_id, x, y, w, h = map(float, parts)
                    
                    # Check coordinates
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        issues['out_of_bounds'] += 1
                        logger.warning(f"Out of bounds coordinates in {label_file}:{line_num}")
                    
                    # Check size
                    if w <= 0 or h <= 0:
                        issues['zero_size'] += 1
                        logger.warning(f"Zero size box in {label_file}:{line_num}")
                    elif w * h < min_box_size:
                        issues['too_small'] += 1
                        logger.warning(f"Box too small in {label_file}:{line_num}")
                        
                except ValueError:
                    issues['invalid_coords'] += 1
                    logger.warning(f"Invalid coordinate values in {label_file}:{line_num}")
    
    logger.info("Dataset validation results:")
    for issue, count in issues.items():
        logger.info(f"- {issue}: {count}")
        
    return issues

def visualize_samples(
    dataset_path: Path,
    num_samples: int = 5,
    output_dir: Optional[Path] = None
) -> None:
    """
    Visualize random samples from dataset with annotations.
    
    Args:
        dataset_path: Path to dataset root
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations (default: dataset_path/visualizations)
    """
    img_dir = dataset_path / 'train/images'
    label_dir = dataset_path / 'train/labels'
    output_dir = output_dir or dataset_path / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = list(img_dir.glob('*.jpg'))
    if not img_files:
        logger.error(f"No images found in {img_dir}")
        return
        
    random.shuffle(img_files)
    
    for img_file in img_files[:num_samples]:
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"Could not read image: {img_file}")
            continue
            
        h, w = img.shape[:2]
        
        # Load corresponding labels
        label_file = label_dir / (img_file.stem + '.txt')
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    try:
                        cls_id, x, y, w_norm, h_norm = map(float, line.strip().split())
                        
                        # Convert normalized coords to pixel coords
                        x1 = int((x - w_norm/2) * w)
                        y1 = int((y - h_norm/2) * h)
                        x2 = int((x + w_norm/2) * w)
                        y2 = int((y + h_norm/2) * h)
                        
                        # Draw box
                        color = (0,255,0) if cls_id == 0 else (0,0,255)
                        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(img, f'{int(cls_id)}', (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                  
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid label format in {label_file}")
                        continue
                        
        output_path = output_dir / f'sample_{img_file.stem}.jpg'
        cv2.imwrite(str(output_path), img)
        
    logger.info(f"Saved {num_samples} visualizations to {output_dir}")