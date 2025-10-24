"""
Comprehensive dataset analysis tool.
"""

import os
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.splits = ['train', 'val', 'test']
        
    def analyze_split(self, split: str) -> Dict:
        """Analyze a specific dataset split."""
        split_dir = self.dataset_dir / split
        if not split_dir.exists():
            return {}
            
        # Get all image and label files
        images = set(f for f in split_dir.rglob('*.*') 
                    if f.suffix.lower() in self.img_extensions)
        labels = set(f for f in split_dir.rglob('*.txt')
                    if f.name != 'classes.txt')
                    
        # Match by stem
        image_stems = {f.stem for f in images}
        label_stems = {f.stem for f in labels}
        
        # Find mismatches
        unlabeled_images = image_stems - label_stems
        orphaned_labels = label_stems - image_stems
        
        # Analyze image sizes
        image_sizes = []
        corrupt_images = []
        for img_path in tqdm(images, desc=f"Analyzing {split} images"):
            try:
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
            except Exception as e:
                corrupt_images.append(str(img_path))
                
        # Analyze labels
        boxes_per_image = defaultdict(int)
        class_distribution = defaultdict(int)
        invalid_labels = []
        
        for label_path in tqdm(labels, desc=f"Analyzing {split} labels"):
            try:
                with open(label_path) as f:
                    lines = f.readlines()
                    boxes_per_image[label_path.stem] = len(lines)
                    for line in lines:
                        try:
                            class_id = int(line.split()[0])
                            class_distribution[class_id] += 1
                        except (ValueError, IndexError):
                            invalid_labels.append(str(label_path))
                            break
            except Exception as e:
                invalid_labels.append(str(label_path))
        
        return {
            'total_images': len(images),
            'total_labels': len(labels),
            'unlabeled_images': list(unlabeled_images),
            'orphaned_labels': list(orphaned_labels),
            'corrupt_images': corrupt_images,
            'invalid_labels': invalid_labels,
            'image_sizes': image_sizes,
            'boxes_per_image': dict(boxes_per_image),
            'class_distribution': dict(class_distribution)
        }
    
    def analyze_dataset(self) -> Dict:
        """Analyze entire dataset across all splits."""
        results = {}
        for split in self.splits:
            results[split] = self.analyze_split(split)
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a detailed markdown report from analysis results."""
        report = ["# Dataset Analysis Report\n"]
        
        # Overall statistics
        total_images = sum(split['total_images'] for split in results.values())
        total_labels = sum(split['total_labels'] for split in results.values())
        
        report.append("## Overall Statistics\n")
        report.append(f"- Total Images: {total_images}")
        report.append(f"- Total Labels: {total_labels}\n")
        
        # Split-wise analysis
        for split, data in results.items():
            if not data:  # Skip empty splits
                continue
                
            report.append(f"## {split.capitalize()} Split\n")
            report.append(f"- Images: {data['total_images']}")
            report.append(f"- Labels: {data['total_labels']}")
            
            if data['unlabeled_images']:
                report.append(f"- Unlabeled Images: {len(data['unlabeled_images'])}")
            if data['orphaned_labels']:
                report.append(f"- Orphaned Labels: {len(data['orphaned_labels'])}")
            if data['corrupt_images']:
                report.append(f"- Corrupt Images: {len(data['corrupt_images'])}")
            if data['invalid_labels']:
                report.append(f"- Invalid Labels: {len(data['invalid_labels'])}")
            
            # Image size statistics
            if data['image_sizes']:
                widths, heights = zip(*data['image_sizes'])
                report.append("\nImage Size Statistics:")
                report.append(f"- Width range: {min(widths)}-{max(widths)}")
                report.append(f"- Height range: {min(heights)}-{max(heights)}")
                
            # Box statistics
            if data['boxes_per_image']:
                boxes = list(data['boxes_per_image'].values())
                report.append("\nBounding Box Statistics:")
                report.append(f"- Average boxes per image: {np.mean(boxes):.2f}")
                report.append(f"- Range: {min(boxes)}-{max(boxes)}")
            
            report.append("\n")  # Add spacing between sections
            
        return "\n".join(report)
    
    def plot_statistics(self, results: Dict, output_dir: str) -> None:
        """Generate visualizations of dataset statistics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Combine data across splits
        all_boxes = []
        all_classes = defaultdict(int)
        
        for split_data in results.values():
            if not split_data:
                continue
            all_boxes.extend(split_data['boxes_per_image'].values())
            for class_id, count in split_data['class_distribution'].items():
                all_classes[class_id] += count
        
        # Plot box distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_boxes, bins=30)
        plt.title('Distribution of Boxes per Image')
        plt.xlabel('Number of Boxes')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'box_distribution.png')
        plt.close()
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        classes = sorted(all_classes.keys())
        counts = [all_classes[c] for c in classes]
        plt.bar(classes, counts)
        plt.title('Class Distribution')
        plt.xlabel('Class ID')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'class_distribution.png')
        plt.close()

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Analyze dataset structure and quality')
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('--output-dir', default='./analysis_output',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(args.dataset_dir)
    results = analyzer.analyze_dataset()
    
    # Generate report
    report = analyzer.generate_report(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'dataset_analysis.md', 'w') as f:
        f.write(report)
    
    # Save raw results
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate plots
    analyzer.plot_statistics(results, output_dir)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()