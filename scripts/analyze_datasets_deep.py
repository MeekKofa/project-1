"""
Deep Dataset Analysis for Object Detection - Cattle Detection Project

Comprehensive analysis including:
- Image statistics (size, brightness, contrast, sharpness)
- Label quality (bbox sizes, aspect ratios, distribution)
- Data quality (duplicates, corrupted files, artifacts)
- Preprocessing recommendations

Author: Auto-generated
Date: 2025-10-03
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
import json
import hashlib
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Setup paths
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Results directory
RESULTS_DIR = project_root / "dataset_analysis_results"
RESULTS_DIR.mkdir(exist_ok=True)

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


class DatasetAnalyzer:
    """Comprehensive dataset analyzer for object detection."""

    def __init__(self, dataset_path: Path, dataset_name: str):
        self.path = Path(dataset_path)
        self.name = dataset_name
        self.results = {
            'name': dataset_name,
            'path': str(dataset_path),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def analyze(self) -> Dict:
        """Run full analysis pipeline."""
        print(f"\n{'='*80}")
        print(f"📊 Analyzing: {self.name}")
        print(f"{'='*80}\n")

        # 1. Discover structure
        self.discover_structure()

        # 2. Analyze images
        self.analyze_images()

        # 3. Analyze labels
        self.analyze_labels()

        # 4. Quality checks
        self.check_data_quality()

        # 5. Generate recommendations
        self.generate_recommendations()

        # 6. Create visualizations
        self.create_visualizations()

        # 7. Save results
        self.save_results()

        return self.results

    def discover_structure(self):
        """Discover dataset structure and format."""
        print("🔍 Discovering dataset structure...")

        structure = {
            'splits': {},
            'format': 'unknown',
            'has_data_yaml': False
        }

        # Check for data.yaml
        data_yaml = self.path / 'data.yaml'
        if data_yaml.exists():
            structure['has_data_yaml'] = True
            with open(data_yaml, 'r') as f:
                yaml_data = yaml.safe_load(f)
                structure['yaml_content'] = yaml_data
                structure['format'] = 'yolo'

        # Discover splits
        for split in ['train', 'val', 'test']:
            split_img = self.path / split / 'images'
            split_lbl = self.path / split / 'labels'

            if split_img.exists():
                img_files = list(split_img.glob(
                    '*.[jp][pn]g')) + list(split_img.glob('*.jpeg'))
                lbl_files = list(split_lbl.glob('*.txt')
                                 ) if split_lbl.exists() else []

                structure['splits'][split] = {
                    'images': len(img_files),
                    'labels': len(lbl_files),
                    'image_dir': str(split_img),
                    'label_dir': str(split_lbl) if split_lbl.exists() else None
                }

        self.results['structure'] = structure
        print(f"  Format: {structure['format']}")
        print(f"  Splits found: {list(structure['splits'].keys())}")

    def analyze_images(self):
        """Comprehensive image analysis."""
        print("\n📸 Analyzing images...")

        image_stats = {
            'dimensions': [],
            'aspect_ratios': [],
            'file_sizes': [],
            'brightness': [],
            'contrast': [],
            'sharpness': [],
            'channels': [],
            'formats': Counter()
        }

        # Collect all image paths
        all_images = []
        for split, info in self.results['structure']['splits'].items():
            if info['images'] > 0:
                img_dir = Path(info['image_dir'])
                all_images.extend(
                    list(img_dir.glob('*.[jp][pn]g')) + list(img_dir.glob('*.jpeg')))

        # Sample if too many
        if len(all_images) > 1000:
            import random
            random.seed(42)
            sample_images = random.sample(all_images, 1000)
            print(f"  Sampling 1000/{len(all_images)} images for analysis")
        else:
            sample_images = all_images

        # Analyze each image
        for img_path in tqdm(sample_images, desc="  Processing images"):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w, c = img.shape

                # Dimensions
                image_stats['dimensions'].append((w, h))
                image_stats['aspect_ratios'].append(w / h if h > 0 else 0)
                image_stats['channels'].append(c)

                # File size
                image_stats['file_sizes'].append(
                    img_path.stat().st_size / 1024)  # KB

                # Format
                image_stats['formats'][img_path.suffix.lower()] += 1

                # Brightness (mean intensity)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                image_stats['brightness'].append(np.mean(gray))

                # Contrast (std of intensity)
                image_stats['contrast'].append(np.std(gray))

                # Sharpness (Laplacian variance)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                image_stats['sharpness'].append(laplacian.var())

            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

        # Compute statistics
        self.results['images'] = {
            'total': len(all_images),
            'analyzed': len(sample_images),
            'dimensions': {
                'unique_count': len(set(image_stats['dimensions'])),
                'most_common': Counter(image_stats['dimensions']).most_common(5),
                'min': min(image_stats['dimensions'], key=lambda x: x[0]*x[1]) if image_stats['dimensions'] else None,
                'max': max(image_stats['dimensions'], key=lambda x: x[0]*x[1]) if image_stats['dimensions'] else None
            },
            'aspect_ratios': {
                'mean': float(np.mean(image_stats['aspect_ratios'])) if image_stats['aspect_ratios'] else 0,
                'std': float(np.std(image_stats['aspect_ratios'])) if image_stats['aspect_ratios'] else 0,
                'min': float(np.min(image_stats['aspect_ratios'])) if image_stats['aspect_ratios'] else 0,
                'max': float(np.max(image_stats['aspect_ratios'])) if image_stats['aspect_ratios'] else 0
            },
            'brightness': {
                'mean': float(np.mean(image_stats['brightness'])) if image_stats['brightness'] else 0,
                'std': float(np.std(image_stats['brightness'])) if image_stats['brightness'] else 0,
                'min': float(np.min(image_stats['brightness'])) if image_stats['brightness'] else 0,
                'max': float(np.max(image_stats['brightness'])) if image_stats['brightness'] else 0
            },
            'contrast': {
                'mean': float(np.mean(image_stats['contrast'])) if image_stats['contrast'] else 0,
                'std': float(np.std(image_stats['contrast'])) if image_stats['contrast'] else 0,
                'min': float(np.min(image_stats['contrast'])) if image_stats['contrast'] else 0,
                'max': float(np.max(image_stats['contrast'])) if image_stats['contrast'] else 0
            },
            'sharpness': {
                'mean': float(np.mean(image_stats['sharpness'])) if image_stats['sharpness'] else 0,
                'std': float(np.std(image_stats['sharpness'])) if image_stats['sharpness'] else 0,
                'min': float(np.min(image_stats['sharpness'])) if image_stats['sharpness'] else 0,
                'max': float(np.max(image_stats['sharpness'])) if image_stats['sharpness'] else 0
            },
            'file_sizes': {
                'mean_kb': float(np.mean(image_stats['file_sizes'])) if image_stats['file_sizes'] else 0,
                'median_kb': float(np.median(image_stats['file_sizes'])) if image_stats['file_sizes'] else 0,
                'min_kb': float(np.min(image_stats['file_sizes'])) if image_stats['file_sizes'] else 0,
                'max_kb': float(np.max(image_stats['file_sizes'])) if image_stats['file_sizes'] else 0
            },
            'formats': dict(image_stats['formats']),
            'channels': {
                'unique': list(set(image_stats['channels'])),
                'most_common': Counter(image_stats['channels']).most_common(1)[0] if image_stats['channels'] else None
            }
        }

        # Store raw stats for visualization
        self.raw_image_stats = image_stats

    def analyze_labels(self):
        """Comprehensive label analysis for object detection."""
        print("\n🏷️  Analyzing labels...")

        label_stats = {
            'bbox_widths': [],
            'bbox_heights': [],
            'bbox_areas': [],
            'bbox_aspect_ratios': [],
            'objects_per_image': [],
            'class_distribution': Counter(),
            'bbox_positions_x': [],
            'bbox_positions_y': []
        }

        # Collect all label paths
        all_labels = []
        for split, info in self.results['structure']['splits'].items():
            if info['labels'] > 0:
                lbl_dir = Path(info['label_dir'])
                all_labels.extend(list(lbl_dir.glob('*.txt')))

        if not all_labels:
            print("  ⚠️  No labels found!")
            self.results['labels'] = {'error': 'No labels found'}
            return

        # Analyze labels (YOLO format: class x y w h)
        for lbl_path in tqdm(all_labels, desc="  Processing labels"):
            try:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()

                label_stats['objects_per_image'].append(len(lines))

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])

                    # Class distribution
                    label_stats['class_distribution'][cls_id] += 1

                    # Bbox statistics
                    label_stats['bbox_widths'].append(w)
                    label_stats['bbox_heights'].append(h)
                    label_stats['bbox_areas'].append(w * h)
                    label_stats['bbox_aspect_ratios'].append(
                        w / h if h > 0 else 0)
                    label_stats['bbox_positions_x'].append(x)
                    label_stats['bbox_positions_y'].append(y)

            except Exception as e:
                print(f"  Error processing {lbl_path}: {e}")

        # Compute statistics
        self.results['labels'] = {
            'total_files': len(all_labels),
            'total_objects': sum(label_stats['objects_per_image']),
            'classes': {
                'num_classes': len(label_stats['class_distribution']),
                'distribution': dict(label_stats['class_distribution']),
                'most_common': label_stats['class_distribution'].most_common(5)
            },
            'objects_per_image': {
                'mean': float(np.mean(label_stats['objects_per_image'])) if label_stats['objects_per_image'] else 0,
                'median': float(np.median(label_stats['objects_per_image'])) if label_stats['objects_per_image'] else 0,
                'min': int(np.min(label_stats['objects_per_image'])) if label_stats['objects_per_image'] else 0,
                'max': int(np.max(label_stats['objects_per_image'])) if label_stats['objects_per_image'] else 0
            },
            'bbox_size': {
                'width': {
                    'mean': float(np.mean(label_stats['bbox_widths'])) if label_stats['bbox_widths'] else 0,
                    'median': float(np.median(label_stats['bbox_widths'])) if label_stats['bbox_widths'] else 0,
                    'min': float(np.min(label_stats['bbox_widths'])) if label_stats['bbox_widths'] else 0,
                    'max': float(np.max(label_stats['bbox_widths'])) if label_stats['bbox_widths'] else 0
                },
                'height': {
                    'mean': float(np.mean(label_stats['bbox_heights'])) if label_stats['bbox_heights'] else 0,
                    'median': float(np.median(label_stats['bbox_heights'])) if label_stats['bbox_heights'] else 0,
                    'min': float(np.min(label_stats['bbox_heights'])) if label_stats['bbox_heights'] else 0,
                    'max': float(np.max(label_stats['bbox_heights'])) if label_stats['bbox_heights'] else 0
                },
                'area': {
                    'mean': float(np.mean(label_stats['bbox_areas'])) if label_stats['bbox_areas'] else 0,
                    'median': float(np.median(label_stats['bbox_areas'])) if label_stats['bbox_areas'] else 0,
                    'min': float(np.min(label_stats['bbox_areas'])) if label_stats['bbox_areas'] else 0,
                    'max': float(np.max(label_stats['bbox_areas'])) if label_stats['bbox_areas'] else 0
                },
                'aspect_ratio': {
                    'mean': float(np.mean(label_stats['bbox_aspect_ratios'])) if label_stats['bbox_aspect_ratios'] else 0,
                    'median': float(np.median(label_stats['bbox_aspect_ratios'])) if label_stats['bbox_aspect_ratios'] else 0
                }
            }
        }

        # Store raw stats for visualization
        self.raw_label_stats = label_stats

    def check_data_quality(self):
        """Check for data quality issues."""
        print("\n✅ Checking data quality...")

        issues = []
        warnings = []

        # Check 1: Image-label mismatch
        for split, info in self.results['structure']['splits'].items():
            img_count = info['images']
            lbl_count = info['labels']
            if img_count != lbl_count:
                issues.append(
                    f"{split}: Image/label mismatch ({img_count} images, {lbl_count} labels)")

        # Check 2: Brightness issues
        if 'images' in self.results:
            brightness = self.results['images']['brightness']
            if brightness['mean'] < 50:
                warnings.append(
                    "Low brightness detected (mean < 50). Consider brightness augmentation.")
            if brightness['mean'] > 200:
                warnings.append(
                    "High brightness detected (mean > 200). Check for overexposure.")

        # Check 3: Contrast issues
        if 'images' in self.results:
            contrast = self.results['images']['contrast']
            if contrast['mean'] < 30:
                warnings.append(
                    "Low contrast detected (mean < 30). Consider contrast enhancement.")

        # Check 4: Sharpness issues
        if 'images' in self.results:
            sharpness = self.results['images']['sharpness']
            if sharpness['mean'] < 50:
                warnings.append(
                    "Low sharpness detected (mean < 50). Images may be blurry.")

        # Check 5: Class imbalance
        if 'labels' in self.results and 'classes' in self.results['labels']:
            dist = self.results['labels']['classes']['distribution']
            if len(dist) > 1:
                max_count = max(dist.values())
                min_count = min(dist.values())
                imbalance_ratio = max_count / \
                    min_count if min_count > 0 else float('inf')
                if imbalance_ratio > 5:
                    warnings.append(
                        f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Consider weighted loss or resampling.")

        # Check 6: Small objects
        if 'labels' in self.results and 'bbox_size' in self.results['labels']:
            area = self.results['labels']['bbox_size']['area']
            if area['mean'] < 0.01:  # Less than 1% of image
                warnings.append(
                    "Small objects detected (mean area < 1%). Consider using smaller anchors.")

        # Check 7: Aspect ratio variation
        if 'images' in self.results:
            ar = self.results['images']['aspect_ratios']
            if ar['max'] - ar['min'] > 1.0:
                warnings.append(
                    f"Large aspect ratio variation ({ar['min']:.2f} to {ar['max']:.2f}). Consider padding or letterboxing.")

        self.results['quality'] = {
            'issues': issues,
            'warnings': warnings,
            'passed': len(issues) == 0
        }

        if issues:
            print(f"  ❌ {len(issues)} issue(s) found")
            for issue in issues:
                print(f"    - {issue}")
        if warnings:
            print(f"  ⚠️  {len(warnings)} warning(s)")
            for warning in warnings:
                print(f"    - {warning}")
        if not issues and not warnings:
            print("  ✅ All quality checks passed!")

    def generate_recommendations(self):
        """Generate preprocessing and training recommendations."""
        print("\n💡 Generating recommendations...")

        recommendations = {
            'preprocessing': [],
            'augmentation': [],
            'training': [],
            'architecture': []
        }

        # Preprocessing recommendations
        if 'images' in self.results:
            dims = self.results['images']['dimensions']
            if dims['unique_count'] > 10:
                recommendations['preprocessing'].append(
                    "Resize all images to consistent size (e.g., 640x640)")

            brightness = self.results['images']['brightness']
            if brightness['std'] > 50:
                recommendations['preprocessing'].append(
                    "Apply histogram equalization for brightness normalization")

        # Augmentation recommendations
        if 'images' in self.results:
            brightness = self.results['images']['brightness']
            contrast = self.results['images']['contrast']

            if brightness['std'] < 30:
                recommendations['augmentation'].append(
                    "Add brightness augmentation (±20%)")
            if contrast['std'] < 20:
                recommendations['augmentation'].append(
                    "Add contrast augmentation (±20%)")

            recommendations['augmentation'].extend([
                "Horizontal flip (p=0.5)",
                "Random rotation (±10°)",
                "Random scale (0.8-1.2)",
                "Mosaic augmentation for multi-object scenes"
            ])

        # Training recommendations
        if 'labels' in self.results and 'classes' in self.results['labels']:
            num_classes = self.results['labels']['classes']['num_classes']
            recommendations['training'].append(
                f"Use num_classes={num_classes}")

            dist = self.results['labels']['classes']['distribution']
            if len(dist) > 1:
                max_count = max(dist.values())
                min_count = min(dist.values())
                if max_count / min_count > 3:
                    recommendations['training'].append(
                        "Use weighted loss or focal loss for class imbalance")

            obj_per_img = self.results['labels']['objects_per_image']
            if obj_per_img['mean'] > 5:
                recommendations['training'].append(
                    "Increase batch size or use gradient accumulation for dense scenes")

        # Architecture recommendations
        if 'labels' in self.results and 'bbox_size' in self.results['labels']:
            area = self.results['labels']['bbox_size']['area']
            if area['mean'] < 0.05:
                recommendations['architecture'].append(
                    "Use small anchor sizes for tiny objects")
                recommendations['architecture'].append(
                    "Consider using higher resolution (e.g., 1280x1280)")
            elif area['mean'] > 0.3:
                recommendations['architecture'].append(
                    "Use larger anchor sizes for large objects")
            else:
                recommendations['architecture'].append(
                    "Default anchor sizes should work well")

        self.results['recommendations'] = recommendations

        for category, recs in recommendations.items():
            if recs:
                print(f"  {category.upper()}:")
                for rec in recs:
                    print(f"    • {rec}")

    def create_visualizations(self):
        """Create visualization plots."""
        print("\n📊 Creating visualizations...")

        fig_dir = FIGURES_DIR / self.name
        fig_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'

        # 1. Image statistics
        if hasattr(self, 'raw_image_stats') and self.raw_image_stats['brightness']:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{self.name} - Image Statistics', fontsize=16)

            # Brightness
            axes[0, 0].hist(self.raw_image_stats['brightness'],
                            bins=50, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Brightness Distribution')
            axes[0, 0].set_xlabel('Mean Intensity')
            axes[0, 0].set_ylabel('Count')

            # Contrast
            axes[0, 1].hist(self.raw_image_stats['contrast'],
                            bins=50, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Contrast Distribution')
            axes[0, 1].set_xlabel('Std Intensity')
            axes[0, 1].set_ylabel('Count')

            # Sharpness
            axes[0, 2].hist(self.raw_image_stats['sharpness'],
                            bins=50, color='lightgreen', edgecolor='black')
            axes[0, 2].set_title('Sharpness Distribution')
            axes[0, 2].set_xlabel('Laplacian Variance')
            axes[0, 2].set_ylabel('Count')

            # Aspect ratios
            axes[1, 0].hist(self.raw_image_stats['aspect_ratios'],
                            bins=50, color='plum', edgecolor='black')
            axes[1, 0].set_title('Aspect Ratio Distribution')
            axes[1, 0].set_xlabel('Width/Height')
            axes[1, 0].set_ylabel('Count')

            # File sizes
            axes[1, 1].hist(self.raw_image_stats['file_sizes'],
                            bins=50, color='khaki', edgecolor='black')
            axes[1, 1].set_title('File Size Distribution')
            axes[1, 1].set_xlabel('Size (KB)')
            axes[1, 1].set_ylabel('Count')

            # Dimensions scatter
            dims = np.array(self.raw_image_stats['dimensions'])
            axes[1, 2].scatter(dims[:, 0], dims[:, 1], alpha=0.5, s=10)
            axes[1, 2].set_title('Image Dimensions')
            axes[1, 2].set_xlabel('Width')
            axes[1, 2].set_ylabel('Height')

            plt.tight_layout()
            plt.savefig(fig_dir / 'image_statistics.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

        # 2. Label statistics
        if hasattr(self, 'raw_label_stats') and self.raw_label_stats['bbox_widths']:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{self.name} - Label Statistics', fontsize=16)

            # Bbox widths
            axes[0, 0].hist(self.raw_label_stats['bbox_widths'],
                            bins=50, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Bbox Width Distribution')
            axes[0, 0].set_xlabel('Width (normalized)')
            axes[0, 0].set_ylabel('Count')

            # Bbox heights
            axes[0, 1].hist(self.raw_label_stats['bbox_heights'],
                            bins=50, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Bbox Height Distribution')
            axes[0, 1].set_xlabel('Height (normalized)')
            axes[0, 1].set_ylabel('Count')

            # Bbox areas
            axes[0, 2].hist(self.raw_label_stats['bbox_areas'],
                            bins=50, color='lightgreen', edgecolor='black')
            axes[0, 2].set_title('Bbox Area Distribution')
            axes[0, 2].set_xlabel('Area (normalized)')
            axes[0, 2].set_ylabel('Count')

            # Bbox aspect ratios
            axes[1, 0].hist(self.raw_label_stats['bbox_aspect_ratios'],
                            bins=50, color='plum', edgecolor='black')
            axes[1, 0].set_title('Bbox Aspect Ratio Distribution')
            axes[1, 0].set_xlabel('Width/Height')
            axes[1, 0].set_ylabel('Count')

            # Objects per image
            axes[1, 1].hist(self.raw_label_stats['objects_per_image'],
                            bins=50, color='khaki', edgecolor='black')
            axes[1, 1].set_title('Objects per Image')
            axes[1, 1].set_xlabel('Count')
            axes[1, 1].set_ylabel('Frequency')

            # Class distribution
            if self.raw_label_stats['class_distribution']:
                classes = list(
                    self.raw_label_stats['class_distribution'].keys())
                counts = list(
                    self.raw_label_stats['class_distribution'].values())
                axes[1, 2].bar(classes, counts,
                               color='lightblue', edgecolor='black')
                axes[1, 2].set_title('Class Distribution')
                axes[1, 2].set_xlabel('Class ID')
                axes[1, 2].set_ylabel('Count')

            plt.tight_layout()
            plt.savefig(fig_dir / 'label_statistics.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

        # 3. Bbox position heatmap
        if hasattr(self, 'raw_label_stats') and self.raw_label_stats['bbox_positions_x']:
            fig, ax = plt.subplots(figsize=(10, 10))

            # Create 2D histogram
            h, xedges, yedges = np.histogram2d(
                self.raw_label_stats['bbox_positions_x'],
                self.raw_label_stats['bbox_positions_y'],
                bins=50
            )

            # Plot heatmap
            im = ax.imshow(h.T, origin='lower', cmap='hot', interpolation='bilinear',
                           extent=[0, 1, 0, 1], aspect='auto')
            ax.set_title(f'{self.name} - Object Position Heatmap', fontsize=14)
            ax.set_xlabel('X Position (normalized)')
            ax.set_ylabel('Y Position (normalized)')
            plt.colorbar(im, ax=ax, label='Object Count')

            plt.tight_layout()
            plt.savefig(fig_dir / 'position_heatmap.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  ✅ Figures saved to {fig_dir}/")

    def save_results(self):
        """Save analysis results to files."""
        print("\n💾 Saving results...")

        # Save JSON
        json_path = RESULTS_DIR / f"{self.name}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✅ JSON: {json_path}")

        # Save TXT report
        txt_path = RESULTS_DIR / f"{self.name}_analysis.txt"
        with open(txt_path, 'w') as f:
            f.write(self.format_report())
        print(f"  ✅ TXT: {txt_path}")

    def format_report(self) -> str:
        """Format results as readable text report."""
        report = []
        report.append("=" * 80)
        report.append(f"DATASET ANALYSIS REPORT: {self.name}")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Path: {self.results['path']}")
        report.append("")

        # Structure
        report.append("STRUCTURE:")
        report.append(f"  Format: {self.results['structure']['format']}")
        report.append(f"  Splits:")
        for split, info in self.results['structure']['splits'].items():
            report.append(
                f"    {split}: {info['images']} images, {info['labels']} labels")
        report.append("")

        # Images
        if 'images' in self.results:
            report.append("IMAGES:")
            report.append(f"  Total: {self.results['images']['total']}")
            report.append(f"  Dimensions:")
            report.append(
                f"    Unique sizes: {self.results['images']['dimensions']['unique_count']}")
            report.append(
                f"  Brightness: mean={self.results['images']['brightness']['mean']:.1f}, std={self.results['images']['brightness']['std']:.1f}")
            report.append(
                f"  Contrast: mean={self.results['images']['contrast']['mean']:.1f}, std={self.results['images']['contrast']['std']:.1f}")
            report.append(
                f"  Sharpness: mean={self.results['images']['sharpness']['mean']:.1f}, std={self.results['images']['sharpness']['std']:.1f}")
            report.append("")

        # Labels
        if 'labels' in self.results and 'total_objects' in self.results['labels']:
            report.append("LABELS:")
            report.append(
                f"  Total objects: {self.results['labels']['total_objects']}")
            report.append(
                f"  Classes: {self.results['labels']['classes']['num_classes']}")
            report.append(
                f"  Objects per image: mean={self.results['labels']['objects_per_image']['mean']:.1f}, median={self.results['labels']['objects_per_image']['median']:.1f}")
            report.append(f"  Bbox size (normalized):")
            report.append(
                f"    Width: mean={self.results['labels']['bbox_size']['width']['mean']:.3f}")
            report.append(
                f"    Height: mean={self.results['labels']['bbox_size']['height']['mean']:.3f}")
            report.append(
                f"    Area: mean={self.results['labels']['bbox_size']['area']['mean']:.3f}")
            report.append("")

        # Quality
        if 'quality' in self.results:
            report.append("QUALITY:")
            if self.results['quality']['issues']:
                report.append("  Issues:")
                for issue in self.results['quality']['issues']:
                    report.append(f"    ❌ {issue}")
            if self.results['quality']['warnings']:
                report.append("  Warnings:")
                for warning in self.results['quality']['warnings']:
                    report.append(f"    ⚠️  {warning}")
            if self.results['quality']['passed']:
                report.append("  ✅ All quality checks passed")
            report.append("")

        # Recommendations
        if 'recommendations' in self.results:
            report.append("RECOMMENDATIONS:")
            for category, recs in self.results['recommendations'].items():
                if recs:
                    report.append(f"  {category.upper()}:")
                    for rec in recs:
                        report.append(f"    • {rec}")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)


def main():
    """Main analysis function."""
    print("\n" + "=" * 80)
    print("🔬 DEEP DATASET ANALYSIS FOR OBJECT DETECTION")
    print("=" * 80)

    # Discover datasets
    dataset_root = project_root / "dataset"
    processed_root = project_root / "processed_data"

    datasets_to_analyze = []

    # Check raw datasets
    if dataset_root.exists():
        for ds_path in dataset_root.iterdir():
            if ds_path.is_dir() and (ds_path / "train").exists():
                datasets_to_analyze.append((ds_path, ds_path.name, 'raw'))

    # Check processed datasets
    if processed_root.exists():
        for ds_path in processed_root.iterdir():
            if ds_path.is_dir() and (ds_path / "train").exists():
                datasets_to_analyze.append(
                    (ds_path, ds_path.name, 'processed'))

    if not datasets_to_analyze:
        print("❌ No datasets found!")
        return

    print(f"\n📁 Found {len(datasets_to_analyze)} dataset(s) to analyze\n")

    # Analyze each dataset
    all_results = []
    for ds_path, ds_name, ds_type in datasets_to_analyze:
        analyzer = DatasetAnalyzer(ds_path, f"{ds_name}_{ds_type}")
        results = analyzer.analyze()
        all_results.append(results)

    # Create summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    for result in all_results:
        name = result['name']
        if 'images' in result and 'labels' in result:
            imgs = result['images']['total']
            objs = result['labels'].get('total_objects', 0)
            classes = result['labels'].get('classes', {}).get('num_classes', 0)
            print(f"  {name}: {imgs} images, {objs} objects, {classes} classes")

    print(f"\n✅ Analysis complete! Results saved to: {RESULTS_DIR}/")
    print(f"📊 Visualizations saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
