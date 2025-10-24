"""
Clean up dataset by removing label files that don't have matching images.
"""

import os
import glob
from pathlib import Path
import logging

def cleanup_dataset(dataset_path: str):
    """
    Remove label files that don't have corresponding images.
    
    Args:
        dataset_path: Path to dataset root containing train/val folders
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = Path(dataset_path) / split
        if not split_path.exists():
            continue
            
        labels_path = split_path / 'labels'
        images_path = split_path / 'images'
        
        if not labels_path.exists() or not images_path.exists():
            logger.warning(f"Skipping {split} - missing labels or images directory")
            continue
            
        # Get all label files
        label_files = list(labels_path.glob('*.txt'))
        removed = 0
        
        for label_file in label_files:
            # Get corresponding image file
            img_stem = label_file.stem
            if '.rf.' in img_stem:  # Handle augmented filenames
                img_stem = img_stem.split('.rf.')[0]
                
            # Check for both jpg and png
            img_file_jpg = images_path / f"{img_stem}.jpg"
            img_file_png = images_path / f"{img_stem}.png"
            
            if not img_file_jpg.exists() and not img_file_png.exists():
                label_file.unlink()  # Remove label file
                removed += 1
                
        logger.info(f"{split}: Removed {removed} label files without matching images")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/cattle',
                       help='Path to dataset root containing train/val folders')
    args = parser.parse_args()
    
    cleanup_dataset(args.dataset)