"""
Dynamic Configuration Loader

Loads config.yaml and dynamically detects dataset properties at runtime:
- num_classes
- class_names
- image/label counts
- dataset format

This ensures we never hardcode dataset properties that should be auto-detected.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class DynamicConfigLoader:
    """Dynamic configuration loader that auto-detects dataset properties."""

    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Load base configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path.cwd()
        self.analysis_dir = self.project_root / "dataset_analysis_results"

    def get_full_config(self) -> Dict:
        """
        Get complete configuration with auto-detected dataset properties.

        Returns:
            Dict: Complete configuration including runtime-detected properties
        """
        config = self.config.copy()

        # Auto-detect dataset properties
        dataset_config = config.get('dataset', {})
        dataset_name = dataset_config.get('name')
        dataset_split = dataset_config.get('split', 'raw')

        if dataset_name:
            # Try to load from analysis first
            dataset_props = self._load_from_analysis(
                dataset_name, dataset_split)

            # If not in analysis, detect from filesystem
            if not dataset_props:
                dataset_root = self._resolve_dataset_path(dataset_config)
                dataset_props = self._detect_from_filesystem(dataset_root)

            # Merge detected properties
            if dataset_props:
                config['dataset'].update(dataset_props)
                logger.info(f"Auto-detected dataset properties:")
                logger.info(
                    f"  num_classes: {dataset_props.get('num_classes')}")
                logger.info(
                    f"  class_names: {dataset_props.get('class_names')}")
                logger.info(f"  format: {dataset_props.get('format')}")

        # Auto-configure loss based on dataset analysis
        config = self._auto_configure_loss(config, dataset_name, dataset_split)

        # Auto-configure resolution based on dataset analysis
        config = self._auto_configure_resolution(
            config, dataset_name, dataset_split)

        return config

    def _resolve_dataset_path(self, dataset_config: Dict) -> Path:
        """Resolve dataset path from config."""
        root = dataset_config.get('root', '')

        # Handle template variables like ${dataset.name}
        dataset_name = dataset_config.get('name', '')
        root = root.replace('${dataset.name}', dataset_name)

        # Make absolute
        if not Path(root).is_absolute():
            root = self.project_root / root

        return Path(root)

    def _load_from_analysis(
        self,
        dataset_name: str,
        split: str
    ) -> Optional[Dict]:
        """Load dataset properties from analysis results."""
        analysis_file = self.analysis_dir / \
            f"{dataset_name}_{split}_analysis.json"

        if not analysis_file.exists():
            logger.warning(f"No analysis file found: {analysis_file}")
            return None

        try:
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)

            # Extract properties
            labels = analysis.get('labels', {})

            # Check if labels exist
            if 'error' in labels:
                logger.error(f"No labels found in {dataset_name}_{split}!")
                return None

            classes = labels.get('classes', {})
            structure = analysis.get('structure', {})

            props = {
                'num_classes': classes.get('num_classes'),
                'format': structure.get('format', 'unknown'),
                'has_data_yaml': structure.get('has_data_yaml', False)
            }

            # Get class names from yaml_content if available
            yaml_content = structure.get('yaml_content', {})
            if 'names' in yaml_content:
                props['class_names'] = yaml_content['names']
            else:
                # Generate default names
                num_classes = props['num_classes']
                props['class_names'] = [
                    f'class_{i}' for i in range(num_classes)]

            # Get split information
            splits = structure.get('splits', {})
            props['splits'] = {
                name: {
                    'num_images': info.get('images', 0),
                    'num_labels': info.get('labels', 0)
                }
                for name, info in splits.items()
            }

            logger.info(
                f"Loaded properties from analysis: {analysis_file.name}")
            return props

        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            return None

    def _detect_from_filesystem(self, dataset_root: Path) -> Optional[Dict]:
        """Detect dataset properties from filesystem."""
        if not dataset_root.exists():
            logger.error(f"Dataset directory not found: {dataset_root}")
            return None

        logger.info(f"Detecting dataset properties from: {dataset_root}")

        # Detect format
        format_type = self._detect_format(dataset_root)

        # Detect classes
        num_classes, class_names = self._detect_classes(
            dataset_root, format_type)

        # Detect splits
        splits = self._detect_splits(dataset_root)

        props = {
            'num_classes': num_classes,
            'class_names': class_names,
            'format': format_type,
            'splits': splits
        }

        logger.info(f"Detected from filesystem:")
        logger.info(f"  Format: {format_type}")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Splits: {list(splits.keys())}")

        return props

    def _detect_format(self, dataset_root: Path) -> str:
        """Detect dataset format."""
        # Check for data.yaml (YOLO)
        if (dataset_root / 'data.yaml').exists():
            return 'yolo'

        # Check for YOLO structure (images/labels directories)
        has_yolo_structure = False
        for split in ['train', 'val', 'test']:
            split_dir = dataset_root / split
            if split_dir.exists():
                if (split_dir / 'images').exists() and (split_dir / 'labels').exists():
                    has_yolo_structure = True
                    break

        if has_yolo_structure:
            return 'yolo'

        # Check for COCO format (JSON annotations)
        json_files = list(dataset_root.glob('**/*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if all(k in data for k in ['images', 'annotations', 'categories']):
                    return 'coco'
            except:
                continue

        # Check for VOC format (XML annotations)
        xml_files = list(dataset_root.glob('**/*.xml'))
        if xml_files:
            return 'voc'

        return 'unknown'

    def _detect_classes(
        self,
        dataset_root: Path,
        format_type: str
    ) -> Tuple[int, List[str]]:
        """Detect number of classes and class names."""

        # Try to read from data.yaml
        data_yaml = dataset_root / 'data.yaml'
        if data_yaml.exists():
            try:
                with open(data_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                num_classes = data.get('nc', 0)
                class_names = data.get('names', [])
                if num_classes and class_names:
                    return num_classes, class_names
            except:
                pass

        # Detect from label files
        if format_type == 'yolo':
            return self._detect_classes_yolo(dataset_root)
        elif format_type == 'coco':
            return self._detect_classes_coco(dataset_root)
        elif format_type == 'voc':
            return self._detect_classes_voc(dataset_root)

        return 0, []

    def _detect_classes_yolo(self, dataset_root: Path) -> Tuple[int, List[str]]:
        """Detect classes from YOLO format labels."""
        class_ids = set()

        # Scan label files
        label_files = list(dataset_root.glob('**/*.txt'))

        # Limit scanning to avoid slowness
        max_files = min(1000, len(label_files))

        for label_file in label_files[:max_files]:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
            except:
                continue

        num_classes = len(class_ids)
        class_names = [f'class_{i}' for i in range(num_classes)]

        return num_classes, class_names

    def _detect_classes_coco(self, dataset_root: Path) -> Tuple[int, List[str]]:
        """Detect classes from COCO format."""
        json_files = list(dataset_root.glob('**/*.json'))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if 'categories' in data:
                    categories = data['categories']
                    num_classes = len(categories)
                    class_names = [
                        cat.get('name', f'class_{i}') for i, cat in enumerate(categories)]
                    return num_classes, class_names
            except:
                continue

        return 0, []

    def _detect_classes_voc(self, dataset_root: Path) -> Tuple[int, List[str]]:
        """Detect classes from VOC format."""
        import xml.etree.ElementTree as ET

        class_names_set = set()
        xml_files = list(dataset_root.glob('**/*.xml'))

        max_files = min(100, len(xml_files))

        for xml_file in xml_files[:max_files]:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name is not None:
                        class_names_set.add(name.text)
            except:
                continue

        class_names = sorted(list(class_names_set))
        return len(class_names), class_names

    def _detect_splits(self, dataset_root: Path) -> Dict:
        """Detect data splits."""
        splits = {}

        for split_name in ['train', 'val', 'test']:
            split_dir = dataset_root / split_name

            if not split_dir.exists():
                continue

            # Count images
            images_dir = split_dir / \
                'images' if (split_dir / 'images').exists() else split_dir
            num_images = sum(1 for _ in images_dir.glob('*.jpg')) + \
                sum(1 for _ in images_dir.glob('*.png')) + \
                sum(1 for _ in images_dir.glob('*.jpeg'))

            # Count labels
            labels_dir = split_dir / \
                'labels' if (split_dir / 'labels').exists() else split_dir
            num_labels = sum(1 for _ in labels_dir.glob(
                '*.txt')) if labels_dir.exists() else 0

            splits[split_name] = {
                'num_images': num_images,
                'num_labels': num_labels
            }

        return splits

    def _auto_configure_loss(
        self,
        config: Dict,
        dataset_name: str,
        split: str
    ) -> Dict:
        """Auto-configure loss based on dataset characteristics."""
        loss_config = config.get('loss', {})

        if loss_config.get('type') == 'auto':
            # Load analysis to check for class imbalance
            analysis_file = self.analysis_dir / \
                f"{dataset_name}_{split}_analysis.json"

            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)

                warnings = analysis.get('quality', {}).get('warnings', [])

                # Check for class imbalance
                has_imbalance = any('imbalance' in w.lower() for w in warnings)

                if has_imbalance:
                    config['loss']['type'] = 'focal'
                    logger.info(
                        "Auto-configured loss: focal (due to class imbalance)")
                else:
                    config['loss']['type'] = 'standard'
                    logger.info("Auto-configured loss: standard")

        return config

    def _auto_configure_resolution(
        self,
        config: Dict,
        dataset_name: str,
        split: str
    ) -> Dict:
        """Auto-configure resolution based on object sizes."""
        analysis_file = self.analysis_dir / \
            f"{dataset_name}_{split}_analysis.json"

        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)

            # Check recommendations for resolution
            arch_recommendations = analysis.get(
                'recommendations', {}).get('architecture', [])

            for rec in arch_recommendations:
                if 'higher resolution' in rec.lower():
                    # Suggest higher resolution for small objects
                    if 'preprocess' not in config:
                        config['preprocess'] = {}

                    # Only suggest if not already set to high resolution
                    current_size = config['preprocess'].get(
                        'target_size', [640, 640])
                    if current_size[0] < 1024:
                        logger.info(
                            f"📊 Dataset analysis suggests higher resolution (e.g., 1280x1280) for small objects")
                        logger.info(
                            f"   Current: {current_size}, consider updating in config.yaml")

        return config


def load_config(config_path: str = "src/config/config.yaml") -> Dict:
    """
    Load configuration with runtime auto-detection.

    Args:
        config_path: Path to src/config/config.yaml

    Returns:
        Dict: Complete configuration with auto-detected properties
    """
    loader = DynamicConfigLoader(config_path)
    return loader.get_full_config()


if __name__ == '__main__':
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    config = load_config()

    print("\n" + "="*80)
    print("LOADED CONFIGURATION")
    print("="*80)
    print(f"\nDataset: {config['dataset']['name']}")
    print(f"Classes: {config['dataset'].get('num_classes', 'N/A')}")
    print(f"Class names: {config['dataset'].get('class_names', 'N/A')}")
    print(f"Format: {config['dataset'].get('format', 'N/A')}")
    print(f"\nLoss type: {config['loss'].get('type', 'N/A')}")
    print("\nSplits:")
    for split_name, split_info in config['dataset'].get('splits', {}).items():
        print(
            f"  {split_name}: {split_info['num_images']} images, {split_info['num_labels']} labels")
