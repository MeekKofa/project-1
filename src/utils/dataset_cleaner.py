"""
Enhanced dataset cleanup utility.
Handles:
1. Removal of augmented files without labels
2. Validation of label quality
3. Dataset statistics generation
"""

import os
from pathlib import Path
import shutil
import json
import logging
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DatasetCleaner:
    def __init__(self, dataset_dir: str, backup_dir: str = None):
        """
        Initialize dataset cleaner.
        
        Args:
            dataset_dir: Root directory of dataset
            backup_dir: Directory to store removed files (optional)
        """
        self.dataset_dir = Path(dataset_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.splits = ['train', 'val', 'test']
        
    def identify_augmented_files(self) -> Dict[str, List[Path]]:
        """Find all augmented files (.rf.) in dataset."""
        augmented_files = {}
        
        for split in self.splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                continue
                
            # Find all augmented images
            augmented = []
            for ext in self.img_extensions:
                augmented.extend(split_dir.rglob(f"*rf.*{ext}"))
            
            augmented_files[split] = augmented
            
        return augmented_files
        
    def find_orphaned_files(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Find files without corresponding pairs (images without labels and vice versa).
        """
        orphans = {}
        
        for split in self.splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                continue
                
            # Get all image and label files
            images = {f.stem: f for f in split_dir.rglob('*.*') 
                     if f.suffix.lower() in self.img_extensions}
            labels = {f.stem: f for f in split_dir.rglob('*.txt')
                     if f.name != 'classes.txt'}
                     
            # Find mismatches
            img_stems = set(images.keys())
            label_stems = set(labels.keys())
            
            orphans[split] = {
                'unlabeled_images': [images[stem] for stem in (img_stems - label_stems)],
                'orphaned_labels': [labels[stem] for stem in (label_stems - img_stems)]
            }
            
        return orphans
        
    def validate_labels(self) -> Dict[str, List[str]]:
        """Check label files for potential issues."""
        invalid_labels = {}
        
        for split in self.splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                continue
                
            invalid = []
            for label_file in tqdm(split_dir.rglob('*.txt'), 
                                 desc=f"Validating {split} labels"):
                if label_file.name == 'classes.txt':
                    continue
                    
                try:
                    with open(label_file) as f:
                        lines = f.readlines()
                        
                    # Validate YOLO format
                    for i, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:  # class x y w h
                            invalid.append(f"{label_file}: Line {i} has wrong format")
                            continue
                            
                        try:
                            # Validate class id
                            class_id = int(parts[0])
                            if class_id < 0:
                                invalid.append(f"{label_file}: Line {i} has invalid class {class_id}")
                            
                            # Validate coordinates (should be between 0 and 1)
                            coords = [float(x) for x in parts[1:]]
                            if not all(0 <= x <= 1 for x in coords):
                                invalid.append(f"{label_file}: Line {i} has invalid coordinates")
                                
                        except ValueError:
                            invalid.append(f"{label_file}: Line {i} has invalid values")
                            
                except Exception as e:
                    invalid.append(f"{label_file}: Failed to read - {str(e)}")
                    
            invalid_labels[split] = invalid
            
        return invalid_labels
        
    def backup_files(self, files: List[Path]) -> None:
        """Backup files before removal."""
        if not self.backup_dir:
            return
            
        for file in files:
            # Create relative path structure in backup
            rel_path = file.relative_to(self.dataset_dir)
            backup_path = self.backup_dir / rel_path
            
            # Create directories if needed
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(file, backup_path)
            
    def remove_files(self, files: List[Path]) -> None:
        """Remove files from dataset."""
        for file in files:
            try:
                file.unlink()
            except Exception as e:
                logger.error(f"Failed to remove {file}: {e}")
                
    def generate_statistics(self) -> Dict:
        """Generate detailed dataset statistics."""
        stats = {}
        
        for split in self.splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                continue
                
            # Count images and labels
            images = list(f for f in split_dir.rglob('*.*') 
                        if f.suffix.lower() in self.img_extensions)
            labels = list(f for f in split_dir.rglob('*.txt')
                         if f.name != 'classes.txt')
                         
            # Analyze image sizes
            img_sizes = []
            for img_path in tqdm(images, desc=f"Analyzing {split} images"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_sizes.append(img.shape[:2])
                except Exception:
                    continue
                    
            # Analyze labels
            boxes_per_image = []
            class_dist = {}
            
            for label_path in labels:
                try:
                    with open(label_path) as f:
                        lines = f.readlines()
                        boxes_per_image.append(len(lines))
                        
                        for line in lines:
                            class_id = int(line.split()[0])
                            class_dist[class_id] = class_dist.get(class_id, 0) + 1
                except Exception:
                    continue
                    
            # Convert numpy types to standard Python types for JSON serialization
            stats[split] = {
                'num_images': len(images),
                'num_labels': len(labels),
                'image_sizes': {
                    'min': tuple(int(x) for x in np.min(img_sizes, axis=0)) if img_sizes else None,
                    'max': tuple(int(x) for x in np.max(img_sizes, axis=0)) if img_sizes else None,
                    'mean': tuple(float(x) for x in np.mean(img_sizes, axis=0)) if img_sizes else None
                },
                'boxes_per_image': {
                    'min': min(boxes_per_image) if boxes_per_image else 0,
                    'max': max(boxes_per_image) if boxes_per_image else 0,
                    'mean': float(np.mean(boxes_per_image)) if boxes_per_image else 0
                },
                'class_distribution': {int(k): int(v) for k, v in class_dist.items()}
            }
            
        return stats
        
    def clean_dataset(self, remove_augmented: bool = True, 
                     remove_orphaned: bool = True) -> Dict:
        """
        Clean the dataset by removing problematic files.
        
        Args:
            remove_augmented: Whether to remove augmented files
            remove_orphaned: Whether to remove orphaned files
            
        Returns:
            Dict with cleanup statistics
        """
        logger.info("Starting dataset cleanup...")
        
        # Find problematic files
        augmented = self.identify_augmented_files()
        orphans = self.find_orphaned_files()
        invalid = self.validate_labels()
        
        # Initialize removal list
        to_remove = []
        
        # Add augmented files if requested
        if remove_augmented:
            for split_files in augmented.values():
                to_remove.extend(split_files)
                
        # Add orphaned files if requested
        if remove_orphaned:
            for split_data in orphans.values():
                to_remove.extend(split_data['unlabeled_images'])
                to_remove.extend(split_data['orphaned_labels'])
                
        # Backup files if backup directory is set
        if self.backup_dir:
            logger.info(f"Backing up {len(to_remove)} files...")
            self.backup_files(to_remove)
            
        # Remove files
        logger.info(f"Removing {len(to_remove)} files...")
        self.remove_files(to_remove)
        
        # Generate final statistics
        logger.info("Generating final statistics...")
        stats = self.generate_statistics()
        
        return {
            'files_removed': len(to_remove),
            'augmented_removed': sum(len(x) for x in augmented.values()),
            'orphaned_removed': sum(
                len(x['unlabeled_images']) + len(x['orphaned_labels']) 
                for x in orphans.values()
            ),
            'invalid_labels': sum(len(x) for x in invalid.values()),
            'final_statistics': stats
        }

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Clean up dataset by removing problematic files')
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('--backup-dir', help='Directory to backup removed files')
    parser.add_argument('--keep-augmented', action='store_true', 
                       help='Keep augmented files')
    parser.add_argument('--keep-orphaned', action='store_true',
                       help='Keep orphaned files')
    
    args = parser.parse_args()
    
    cleaner = DatasetCleaner(args.dataset_dir, args.backup_dir)
    results = cleaner.clean_dataset(
        remove_augmented=not args.keep_augmented,
        remove_orphaned=not args.keep_orphaned
    )
    
    # Save results
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'cleanup_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Cleanup complete. Results saved to {output_dir / 'cleanup_results.json'}")

if __name__ == '__main__':
    main()