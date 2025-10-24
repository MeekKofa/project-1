"""
Dataset cleanup utility to remove orphaned label files.
"""

import os
from pathlib import Path
import logging
from typing import List, Tuple, Set
from tqdm import tqdm

logger = logging.getLogger(__name__)

def find_orphaned_labels(dataset_dir: str) -> Tuple[List[Path], int, int]:
    """
    Find label files that don't have corresponding images.
    
    Args:
        dataset_dir: Path to dataset directory containing 'train', 'val', etc.
        
    Returns:
        Tuple of (orphaned label files, total labels, total images)
    """
    orphaned_labels = []
    total_labels = 0
    total_images = 0
    
    # Supported image extensions
    img_extensions = {'.jpg', '.jpeg', '.png'}
    
    for split in ['train', 'val', 'test']:
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            continue
            
        # Get all image and label files
        images = {f.stem for f in split_dir.glob('**/*.*') if f.suffix.lower() in img_extensions}
        labels = set(f for f in split_dir.rglob('*.txt'))
        
        total_images += len(images)
        total_labels += len(labels)
        
        # Find orphaned labels
        for label_path in labels:
            # Skip special files
            if label_path.name == 'classes.txt':
                continue
                
            # Check if corresponding image exists
            if label_path.stem not in images:
                orphaned_labels.append(label_path)
                
    return orphaned_labels, total_labels, total_images

def remove_orphaned_labels(orphaned_labels: List[Path], dry_run: bool = True) -> int:
    """
    Remove orphaned label files.
    
    Args:
        orphaned_labels: List of paths to orphaned label files
        dry_run: If True, only print what would be done
        
    Returns:
        Number of files removed
    """
    removed = 0
    
    for label_path in tqdm(orphaned_labels, desc="Removing orphaned labels"):
        if dry_run:
            logger.info(f"Would remove: {label_path}")
            removed += 1
        else:
            try:
                label_path.unlink()
                removed += 1
            except Exception as e:
                logger.error(f"Failed to remove {label_path}: {e}")
                
    return removed

def cleanup_dataset(dataset_dir: str, dry_run: bool = True) -> None:
    """
    Clean up dataset by removing orphaned label files.
    
    Args:
        dataset_dir: Path to dataset directory
        dry_run: If True, only print what would be done
    """
    logger.info(f"Analyzing dataset in: {dataset_dir}")
    
    orphaned_labels, total_labels, total_images = find_orphaned_labels(dataset_dir)
    
    logger.info(f"Found {len(orphaned_labels)} orphaned labels")
    logger.info(f"Total labels: {total_labels}")
    logger.info(f"Total images: {total_images}")
    
    if orphaned_labels:
        if dry_run:
            logger.info("Dry run - no files will be removed")
        removed = remove_orphaned_labels(orphaned_labels, dry_run)
        logger.info(f"{'Would remove' if dry_run else 'Removed'} {removed} orphaned label files")
    else:
        logger.info("No orphaned labels found")

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Clean up dataset by removing orphaned label files')
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually remove files instead of just printing')
    
    args = parser.parse_args()
    
    cleanup_dataset(args.dataset_dir, dry_run=not args.no_dry_run)