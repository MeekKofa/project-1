"""
Robust Dataset Preprocessing Script

This script preprocesses raw datasets using configuration from config.yaml
and insights from dataset_analysis_results. It handles:
- Image resizing with letterboxing
- Label format conversion
- Quality filtering (invalid boxes, mismatched images)
- Data splitting
- Format normalization

Usage:
    python preprocess_dataset.py --dataset cattle --split raw
    python preprocess_dataset.py --dataset cattlebody --split raw --force
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import yaml
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """Robust dataset preprocessor using analysis results and config."""

    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize preprocessor with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path.cwd()
        self.analysis_dir = self.project_root / "dataset_analysis_results"

    def load_dataset_analysis(self, dataset_name: str, split: str) -> Optional[Dict]:
        """Load analysis results for a dataset."""
        analysis_file = self.analysis_dir / \
            f"{dataset_name}_{split}_analysis.json"

        if not analysis_file.exists():
            logger.warning(f"No analysis file found: {analysis_file}")
            return None

        with open(analysis_file, 'r') as f:
            return json.load(f)

    def preprocess_dataset(
        self,
        dataset_name: str,
        split: str = "raw",
        force: bool = False
    ) -> bool:
        """
        Preprocess a dataset.

        Args:
            dataset_name: Name of dataset (cattle, cattlebody, cattleface)
            split: Which split to use (raw, processed)
            force: Overwrite existing processed data

        Returns:
            bool: Success status
        """
        logger.info(f"="*80)
        logger.info(f"Preprocessing: {dataset_name} ({split})")
        logger.info(f"="*80)

        # Load analysis results
        analysis = self.load_dataset_analysis(dataset_name, split)
        if analysis is None:
            logger.error(f"Cannot proceed without analysis results!")
            return False

        # Check dataset issues
        issues = analysis.get('quality', {}).get('issues', [])
        if issues:
            logger.warning(f"Dataset has {len(issues)} issue(s):")
            for issue in issues:
                logger.warning(f"  - {issue}")

        # Setup paths
        if split == "raw":
            input_dir = self.project_root / "dataset" / dataset_name
        else:
            input_dir = self.project_root / "processed_data" / dataset_name

        output_dir = self.project_root / "processed_data" / \
            f"{dataset_name}_preprocessed"

        if output_dir.exists() and not force:
            logger.info(f"Output directory already exists: {output_dir}")
            logger.info("Use --force to overwrite")
            return True

        # Remove existing output if force
        if output_dir.exists() and force:
            logger.info(f"Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)

        # Create output structure
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get preprocessing config
        preprocess_cfg = self.config.get('preprocess', {})
        target_size = tuple(preprocess_cfg.get('target_size', [640, 640]))
        maintain_aspect = preprocess_cfg.get('maintain_aspect', True)
        min_bbox_size = preprocess_cfg.get('min_bbox_size', 0.001)
        filter_invalid = preprocess_cfg.get('filter_invalid_boxes', True)

        logger.info(f"Preprocessing config:")
        logger.info(f"  Target size: {target_size}")
        logger.info(f"  Maintain aspect: {maintain_aspect}")
        logger.info(f"  Min bbox size: {min_bbox_size}")
        logger.info(f"  Filter invalid boxes: {filter_invalid}")

        # Process each split
        structure = analysis.get('structure', {})
        splits_info = structure.get('splits', {})

        total_processed = 0
        total_filtered = 0

        for split_name, split_info in splits_info.items():
            logger.info(f"\n📂 Processing {split_name} split...")

            image_dir = split_info.get('image_dir')
            label_dir = split_info.get('label_dir')

            if not image_dir or not Path(image_dir).exists():
                logger.warning(
                    f"  Skipping {split_name}: image directory not found")
                continue

            # Create output directories
            out_split = output_dir / split_name
            out_images = out_split / "images"
            out_labels = out_split / "labels"
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            # Get all images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(Path(image_dir).glob(ext))

            logger.info(f"  Found {len(image_files)} images")

            processed = 0
            filtered = 0

            for img_path in tqdm(image_files, desc=f"  Processing {split_name}"):
                try:
                    # Process image and labels
                    result = self._process_image_label_pair(
                        img_path=img_path,
                        label_dir=label_dir if label_dir else None,
                        output_images=out_images,
                        output_labels=out_labels,
                        target_size=target_size,
                        maintain_aspect=maintain_aspect,
                        min_bbox_size=min_bbox_size,
                        filter_invalid=filter_invalid
                    )

                    if result:
                        processed += 1
                    else:
                        filtered += 1

                except Exception as e:
                    logger.error(f"  Error processing {img_path.name}: {e}")
                    filtered += 1

            logger.info(
                f"  ✅ {split_name}: {processed} processed, {filtered} filtered")
            total_processed += processed
            total_filtered += filtered

        # Create data.yaml for YOLO format
        self._create_data_yaml(output_dir, analysis)

        # Create preprocessing summary
        self._create_summary(
            output_dir,
            dataset_name,
            split,
            total_processed,
            total_filtered,
            analysis
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Preprocessing complete!")
        logger.info(f"  Total processed: {total_processed}")
        logger.info(f"  Total filtered: {total_filtered}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"{'='*80}\n")

        return True

    def _process_image_label_pair(
        self,
        img_path: Path,
        label_dir: Optional[Path],
        output_images: Path,
        output_labels: Path,
        target_size: Tuple[int, int],
        maintain_aspect: bool,
        min_bbox_size: float,
        filter_invalid: bool
    ) -> bool:
        """Process a single image-label pair."""

        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        if maintain_aspect:
            # Letterbox resize
            img_resized, scale, pad_x, pad_y = self._letterbox_resize(
                img, target_size
            )
        else:
            # Simple resize
            img_resized = img.resize(target_size, Image.BILINEAR)
            scale = (target_size[0] / orig_w, target_size[1] / orig_h)
            pad_x, pad_y = 0, 0

        # Load and transform labels
        label_path = None
        if label_dir:
            label_path = label_dir / (img_path.stem + '.txt')

        boxes_valid = []

        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        # Convert to pixel coordinates (original image)
                        cx_px = cx * orig_w
                        cy_px = cy * orig_h
                        w_px = w * orig_w
                        h_px = h * orig_h

                        # Apply letterbox transformation
                        if maintain_aspect:
                            cx_new = (cx_px * scale[0]) + pad_x
                            cy_new = (cy_px * scale[1]) + pad_y
                            w_new = w_px * scale[0]
                            h_new = h_px * scale[1]
                        else:
                            cx_new = cx_px * scale[0]
                            cy_new = cy_px * scale[1]
                            w_new = w_px * scale[0]
                            h_new = h_px * scale[1]

                        # Normalize to new image size
                        cx_norm = cx_new / target_size[0]
                        cy_norm = cy_new / target_size[1]
                        w_norm = w_new / target_size[0]
                        h_norm = h_new / target_size[1]

                        # Filter invalid boxes
                        if filter_invalid:
                            # Check if box is valid
                            if (cx_norm < 0 or cx_norm > 1 or
                                cy_norm < 0 or cy_norm > 1 or
                                    w_norm <= 0 or h_norm <= 0):
                                continue

                            # Check if box is too small
                            if w_norm * h_norm < min_bbox_size:
                                continue

                            # Clip to valid range
                            cx_norm = np.clip(cx_norm, 0, 1)
                            cy_norm = np.clip(cy_norm, 0, 1)
                            w_norm = np.clip(w_norm, 0, 1 - cx_norm + w_norm/2)
                            h_norm = np.clip(h_norm, 0, 1 - cy_norm + h_norm/2)

                        boxes_valid.append(
                            [class_id, cx_norm, cy_norm, w_norm, h_norm])

                    except (ValueError, IndexError):
                        continue

        # Save processed image
        output_img_path = output_images / img_path.name
        img_resized.save(output_img_path, quality=95)

        # Save processed labels
        if boxes_valid:
            output_label_path = output_labels / (img_path.stem + '.txt')
            with open(output_label_path, 'w') as f:
                for box in boxes_valid:
                    f.write(
                        f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

        return True

    def _letterbox_resize(
        self,
        img: Image.Image,
        target_size: Tuple[int, int]
    ) -> Tuple[Image.Image, Tuple[float, float], float, float]:
        """
        Resize image with letterboxing to maintain aspect ratio.

        Returns:
            resized_img: Letterboxed image
            scale: (scale_x, scale_y)
            pad_x: Horizontal padding
            pad_y: Vertical padding
        """
        orig_w, orig_h = img.size
        target_w, target_h = target_size

        # Calculate scale to fit image in target size
        scale = min(target_w / orig_w, target_h / orig_h)

        # Resize image
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create canvas and paste resized image
        canvas = Image.new('RGB', target_size, (114, 114, 114))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas.paste(img_resized, (pad_x, pad_y))

        return canvas, (scale, scale), pad_x, pad_y

    def _create_data_yaml(self, output_dir: Path, analysis: Dict):
        """Create data.yaml file for YOLO format."""
        # Extract class information
        labels = analysis.get('labels', {})
        if 'error' in labels:
            logger.warning("No labels found, skipping data.yaml creation")
            return

        classes = labels.get('classes', {})
        num_classes = classes.get('num_classes', 0)

        # Get class names from original data.yaml if available
        structure = analysis.get('structure', {})
        yaml_content = structure.get('yaml_content', {})
        class_names = yaml_content.get(
            'names', [f'class_{i}' for i in range(num_classes)])

        # Create data.yaml
        data_yaml = {
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': num_classes,
            'names': class_names
        }

        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(f"  Created data.yaml: {num_classes} classes")

    def _create_summary(
        self,
        output_dir: Path,
        dataset_name: str,
        split: str,
        processed: int,
        filtered: int,
        analysis: Dict
    ):
        """Create preprocessing summary."""
        summary = {
            'dataset_name': dataset_name,
            'source_split': split,
            'preprocessing_config': self.config.get('preprocess', {}),
            'statistics': {
                'total_processed': processed,
                'total_filtered': filtered,
                'filter_rate': filtered / (processed + filtered) if (processed + filtered) > 0 else 0
            },
            'source_analysis': {
                'issues': analysis.get('quality', {}).get('issues', []),
                'warnings': analysis.get('quality', {}).get('warnings', []),
                'recommendations': analysis.get('recommendations', {})
            }
        }

        summary_path = output_dir / 'preprocessing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also create human-readable version
        txt_path = output_dir / 'preprocessing_summary.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"PREPROCESSING SUMMARY\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Dataset: {dataset_name} ({split})\n")
            f.write(f"Output: {output_dir}\n\n")
            f.write(f"Statistics:\n")
            f.write(f"  Processed: {processed}\n")
            f.write(f"  Filtered: {filtered}\n")
            f.write(
                f"  Filter Rate: {summary['statistics']['filter_rate']:.2%}\n\n")
            f.write(f"Source Issues:\n")
            for issue in summary['source_analysis']['issues']:
                f.write(f"  - {issue}\n")
            f.write(f"\nSource Warnings:\n")
            for warning in summary['source_analysis']['warnings']:
                f.write(f"  - {warning}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for object detection training"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['cattle', 'cattlebody', 'cattleface'],
        help='Dataset name'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='raw',
        choices=['raw', 'processed'],
        help='Which split to use'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing processed data'
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = DatasetPreprocessor(config_path=args.config)

    # Run preprocessing
    success = preprocessor.preprocess_dataset(
        dataset_name=args.dataset,
        split=args.split,
        force=args.force
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
