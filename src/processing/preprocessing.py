"""
Dataset preprocessing module for converting raw datasets to training-ready format.

This module handles the conversion of raw cattle detection datasets into
the standardized format required for training.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
import random
from PIL import Image
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def convert_dataset_to_training_format(
    dataset_name: str,
    project_root: Optional[str] = None,
    force: bool = False
) -> bool:
    """
    Convert a raw dataset to training-ready format.

    Args:
        dataset_name: Name of the dataset (e.g., 'cattlebody', 'cattleface')
        project_root: Project root directory (defaults to current working directory)
        force: Whether to overwrite existing processed dataset

    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    if project_root is None:
        project_root = os.getcwd()

    project_path = Path(project_root)
    raw_dataset_dir = project_path / "dataset" / dataset_name
    processed_dataset_dir = project_path / "processed_data" / dataset_name

    logger.info(
        f"Converting dataset '{dataset_name}' from {raw_dataset_dir} to {processed_dataset_dir}")

    # Check if raw dataset exists
    if not raw_dataset_dir.exists():
        logger.error(f"Raw dataset directory not found: {raw_dataset_dir}")
        return False

    # Check if already processed
    if processed_dataset_dir.exists() and not force:
        logger.info(
            f"Dataset '{dataset_name}' already processed. Use --force to overwrite.")
        return True

    return preprocess_raw_dataset(
        input_dir=str(raw_dataset_dir),
        output_dir=str(processed_dataset_dir),
        force=force
    )


def preprocess_raw_dataset(
    input_dir: str,
    output_dir: str,
    split_ratio: float = 0.8,
    force: bool = False
) -> bool:
    """
    Preprocess raw dataset for training.

    Args:
        input_dir: Path to raw dataset directory
        output_dir: Path to output processed dataset directory
        split_ratio: Ratio for train/validation split (default: 0.8)
        force: Whether to overwrite existing output directory

    Returns:
        bool: True if preprocessing succeeded, False otherwise
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    logger.info(f"Processing dataset from {input_path} to {output_path}")

    # Remove existing output if force is True
    if output_path.exists() and force:
        logger.info(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # Detect dataset format and process accordingly
        if _is_yolo_format(input_path):
            logger.info("Detected YOLO format dataset")
            return _process_yolo_dataset(input_path, output_path, split_ratio)
        elif _is_coco_format(input_path):
            logger.info("Detected COCO format dataset")
            return _process_coco_dataset(input_path, output_path, split_ratio)
        elif _is_voc_format(input_path):
            logger.info("Detected VOC format dataset")
            return _process_voc_dataset(input_path, output_path, split_ratio)
        else:
            logger.info("Attempting generic format processing")
            return _process_generic_dataset(input_path, output_path, split_ratio)

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return False


def _is_yolo_format(dataset_path: Path) -> bool:
    """Check if dataset is in YOLO format."""
    # Look for .txt annotation files and images in same directory structure
    txt_files = list(dataset_path.rglob("*.txt"))
    img_files = list(dataset_path.rglob("*.jpg")) + \
        list(dataset_path.rglob("*.png"))

    # YOLO format has txt files with same names as images
    if txt_files and img_files:
        # Check if there are matching txt and image files
        txt_names = {f.stem for f in txt_files}
        img_names = {f.stem for f in img_files}
        common_names = txt_names.intersection(img_names)
        return len(common_names) > 0
    return False


def _is_coco_format(dataset_path: Path) -> bool:
    """Check if dataset is in COCO format."""
    # Look for annotations.json or similar COCO annotation files
    json_files = list(dataset_path.rglob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # COCO format has specific keys
                if all(key in data for key in ['images', 'annotations', 'categories']):
                    return True
        except:
            continue
    return False


def _is_voc_format(dataset_path: Path) -> bool:
    """Check if dataset is in VOC format."""
    # Look for XML annotation files
    xml_files = list(dataset_path.rglob("*.xml"))
    return len(xml_files) > 0


def _process_yolo_dataset(input_path: Path, output_path: Path, split_ratio: float) -> bool:
    """Process YOLO format dataset."""
    logger.info("Processing YOLO format dataset...")

    # Check if dataset is already in split format (has train/val directories)
    if (input_path / "train").exists() and (input_path / "val").exists():
        logger.info(
            "Dataset already has train/val splits, copying existing structure...")
        return _copy_existing_splits(input_path, output_path)

    # Find all image and annotation pairs
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(input_path.rglob(ext)))

    # Filter images that have corresponding annotation files
    valid_pairs = []
    for img_file in image_files:
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            valid_pairs.append((img_file, txt_file))

    logger.info(f"Found {len(valid_pairs)} image-annotation pairs")

    if len(valid_pairs) == 0:
        logger.error("No valid image-annotation pairs found")
        return False

    # Split dataset
    random.shuffle(valid_pairs)
    train_count = int(len(valid_pairs) * split_ratio)
    val_count = int(len(valid_pairs) * 0.15)  # 15% for validation

    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    # Copy files to appropriate directories
    for pairs, split_name in [(train_pairs, 'train'), (val_pairs, 'val'), (test_pairs, 'test')]:
        split_dir = output_path / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_file, txt_file in pairs:
            # Copy image to images subdirectory
            dst_img = images_dir / img_file.name
            shutil.copy2(img_file, dst_img)

            # Copy annotation to labels subdirectory
            dst_txt = labels_dir / txt_file.name
            shutil.copy2(txt_file, dst_txt)

    logger.info(
        f"Split completed: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

    # Create dataset info file
    _create_dataset_info(output_path, len(train_pairs),
                         len(val_pairs), len(test_pairs), "YOLO")

    return True


def _copy_existing_splits(input_path: Path, output_path: Path) -> bool:
    """Copy existing train/val/test splits to output directory."""
    splits = ['train', 'val', 'test']
    counts = {'train': 0, 'val': 0, 'test': 0}

    for split in splits:
        split_input = input_path / split
        split_output = output_path / split

        if split_input.exists():
            logger.info(f"Copying {split} split...")
            if split_output.exists():
                shutil.rmtree(split_output)
            shutil.copytree(split_input, split_output)

            # Count files
            images_dir = split_output / "images"
            if images_dir.exists():
                counts[split] = len(list(images_dir.glob("*.jpg"))) + \
                    len(list(images_dir.glob("*.png"))) + \
                    len(list(images_dir.glob("*.jpeg")))
            else:
                # If no images subdirectory, count all images in split directory
                counts[split] = len(list(split_output.glob("*.jpg"))) + \
                    len(list(split_output.glob("*.png"))) + \
                    len(list(split_output.glob("*.jpeg")))
        else:
            logger.warning(
                f"Split directory {split_input} not found, skipping...")

    # Copy data.yaml if it exists
    data_yaml = input_path / "data.yaml"
    if data_yaml.exists():
        shutil.copy2(data_yaml, output_path / "data.yaml")
        logger.info("Copied data.yaml configuration file")

    # Copy README files if they exist
    for readme_file in ["README.dataset.txt", "README.roboflow.txt"]:
        readme_path = input_path / readme_file
        if readme_path.exists():
            shutil.copy2(readme_path, output_path / readme_file)

    logger.info(
        f"Existing splits copied: {counts['train']} train, {counts['val']} val, {counts['test']} test")

    # Create dataset info file
    _create_dataset_info(
        output_path, counts['train'], counts['val'], counts['test'], "YOLO")

    return True


def _process_voc_dataset(input_path: Path, output_path: Path, split_ratio: float) -> bool:
    """Process VOC format dataset."""
    logger.info("Processing VOC format dataset...")

    # Find all image and XML annotation pairs
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(input_path.rglob(ext)))

    valid_pairs = []
    for img_file in image_files:
        xml_file = img_file.with_suffix('.xml')
        if xml_file.exists():
            valid_pairs.append((img_file, xml_file))

    logger.info(f"Found {len(valid_pairs)} image-annotation pairs")

    if len(valid_pairs) == 0:
        logger.error("No valid image-annotation pairs found")
        return False

    # Split and copy files (similar to YOLO processing)
    random.shuffle(valid_pairs)
    train_count = int(len(valid_pairs) * split_ratio)
    val_count = int(len(valid_pairs) * 0.15)

    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    for pairs, split_name in [(train_pairs, 'train'), (val_pairs, 'val'), (test_pairs, 'test')]:
        split_dir = output_path / split_name
        for img_file, xml_file in pairs:
            shutil.copy2(img_file, split_dir / img_file.name)
            shutil.copy2(xml_file, split_dir / xml_file.name)

    logger.info(
        f"Split completed: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    _create_dataset_info(output_path, len(train_pairs),
                         len(val_pairs), len(test_pairs), "VOC")

    return True


def _process_coco_dataset(input_path: Path, output_path: Path, split_ratio: float) -> bool:
    """Process COCO format dataset."""
    logger.info("Processing COCO format dataset...")
    # Implementation for COCO format would go here
    # For now, fall back to generic processing
    return _process_generic_dataset(input_path, output_path, split_ratio)


def _process_generic_dataset(input_path: Path, output_path: Path, split_ratio: float) -> bool:
    """Process dataset with unknown/generic format."""
    logger.info("Processing generic format dataset...")

    # Check for cattleface-style dataset (Annotation/ and CowfaceImage/ directories)
    if (input_path / "Annotation").exists() and (input_path / "CowfaceImage").exists():
        logger.info("Detected cattleface dataset format")
        return _process_cattleface_dataset(input_path, output_path, split_ratio)

    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(input_path.rglob(ext)))

    logger.info(f"Found {len(image_files)} image files")

    if len(image_files) == 0:
        logger.error("No image files found")
        return False

    # Split images
    random.shuffle(image_files)
    train_count = int(len(image_files) * split_ratio)
    val_count = int(len(image_files) * 0.15)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    # Copy files to splits
    for files, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        split_dir = output_path / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_file in files:
            shutil.copy2(img_file, images_dir / img_file.name)

            # Look for any matching annotation files
            for ext in ['.txt', '.xml', '.json']:
                ann_file = img_file.with_suffix(ext)
                if ann_file.exists():
                    shutil.copy2(ann_file, labels_dir / ann_file.name)

    logger.info(
        f"Split completed: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    _create_dataset_info(output_path, len(train_files),
                         len(val_files), len(test_files), "Generic")

    return True


def _process_cattleface_dataset(input_path: Path, output_path: Path, split_ratio: float) -> bool:
    """Process cattleface dataset with Annotation/ and CowfaceImage/ structure."""
    logger.info("Processing cattleface dataset...")

    images_dir = input_path / "CowfaceImage"
    annotations_dir = input_path / "Annotation"

    if not images_dir.exists() or not annotations_dir.exists():
        logger.error(
            "Required directories (CowfaceImage/, Annotation/) not found")
        return False

    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(images_dir.glob(ext)))

    logger.info(f"Found {len(image_files)} images in CowfaceImage/")

    # Find matching annotation files
    valid_pairs = []
    for img_file in image_files:
        # Look for matching annotation file (could be .txt, .xml, or other formats)
        base_name = img_file.stem
        for ann_ext in ['.txt', '.xml', '.json']:
            ann_file = annotations_dir / f"{base_name}{ann_ext}"
            if ann_file.exists():
                valid_pairs.append((img_file, ann_file))
                break

    logger.info(f"Found {len(valid_pairs)} image-annotation pairs")

    if len(valid_pairs) == 0:
        logger.error("No valid image-annotation pairs found")
        return False

    # Split dataset
    random.shuffle(valid_pairs)
    train_count = int(len(valid_pairs) * split_ratio)
    val_count = int(len(valid_pairs) * 0.15)

    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]

    # Copy files to splits
    for pairs, split_name in [(train_pairs, 'train'), (val_pairs, 'val'), (test_pairs, 'test')]:
        split_dir = output_path / split_name
        images_dir_out = split_dir / "images"
        labels_dir_out = split_dir / "labels"
        images_dir_out.mkdir(parents=True, exist_ok=True)
        labels_dir_out.mkdir(parents=True, exist_ok=True)

        for img_file, ann_file in pairs:
            # Copy image
            shutil.copy2(img_file, images_dir_out / img_file.name)
            # Copy annotation
            shutil.copy2(ann_file, labels_dir_out / ann_file.name)

    logger.info(
        f"Cattleface dataset processed: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    _create_dataset_info(output_path, len(train_pairs), len(
        val_pairs), len(test_pairs), "CattleFace")

    return True


def _create_dataset_info(output_path: Path, train_count: int, val_count: int, test_count: int, format_type: str):
    """Create dataset information file."""
    info = {
        "dataset_name": output_path.name,
        "format": format_type,
        "splits": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": train_count + val_count + test_count
        },
        "created_by": "Cattle Detection System Preprocessor"
    }

    info_file = output_path / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Dataset info saved to {info_file}")
