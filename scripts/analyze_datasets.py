#!/usr/bin/env python3
"""
Dataset Discovery & Analysis Script.

Automatically discovers, analyzes, and validates all datasets in the project.
Generates comprehensive reports and configuration suggestions.

Usage:
    python analyze_datasets.py
    python analyze_datasets.py --dataset cattlebody
    python analyze_datasets.py --output report.json
"""

from datetime import datetime
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict
import yaml
import json
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class DatasetAnalyzer:
    """Automatically discover and analyze datasets."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.datasets = {}

    def discover_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Discover all datasets in dataset/ and processed_data/ folders."""
        print("🔍 Discovering datasets...")

        # Check dataset/ folder
        dataset_dir = self.root_dir / 'dataset'
        if dataset_dir.exists():
            for ds_path in dataset_dir.iterdir():
                if ds_path.is_dir() and not ds_path.name.startswith('.'):
                    self._analyze_dataset(ds_path, 'raw')

        # Check processed_data/ folder
        processed_dir = self.root_dir / 'processed_data'
        if processed_dir.exists():
            for ds_path in processed_dir.iterdir():
                if ds_path.is_dir() and not ds_path.name.startswith('.'):
                    self._analyze_dataset(ds_path, 'processed')

        return self.datasets

    def _analyze_dataset(self, ds_path: Path, ds_type: str):
        """Analyze a single dataset."""
        ds_name = ds_path.name

        print(f"  📁 Analyzing: {ds_name} ({ds_type})")

        info = {
            'name': ds_name,
            'path': str(ds_path),
            'type': ds_type,
            'format': self._detect_format(ds_path),
            'splits': {},
            'num_classes': None,
            'class_names': [],
            'statistics': {},
            'issues': [],
            'config': {}
        }

        # Detect splits (train/val/test)
        for split in ['train', 'val', 'test']:
            split_info = self._analyze_split(ds_path, split)
            if split_info:
                info['splits'][split] = split_info

        # Read data.yaml if exists
        data_yaml = ds_path / 'data.yaml'
        if data_yaml.exists():
            info['config'] = self._read_yaml(data_yaml)
            info['num_classes'] = info['config'].get('nc')
            info['class_names'] = info['config'].get('names', [])

        # Auto-detect num_classes if not found
        if info['num_classes'] is None:
            info['num_classes'] = self._detect_num_classes(ds_path)

        # Validate dataset
        info['issues'] = self._validate_dataset(info)

        # Calculate statistics
        info['statistics'] = self._calculate_statistics(info)

        self.datasets[ds_name] = info

    def _detect_format(self, ds_path: Path) -> str:
        """Detect dataset format (YOLO, COCO, Pascal VOC, etc.)."""
        # Check for YOLO format (train/images + train/labels)
        if (ds_path / 'train' / 'images').exists() and (ds_path / 'train' / 'labels').exists():
            return 'yolo'

        # Check for processed format (Annotation/ + Images/)
        if (ds_path / 'Annotation').exists() and (ds_path / 'CowfaceImage').exists():
            return 'custom_annotation'

        # Check for COCO format
        if (ds_path / 'annotations').exists():
            return 'coco'

        return 'unknown'

    def _analyze_split(self, ds_path: Path, split: str) -> Optional[Dict]:
        """Analyze a data split (train/val/test)."""
        split_path = ds_path / split
        if not split_path.exists():
            return None

        images_path = split_path / 'images'
        labels_path = split_path / 'labels'

        info = {
            'path': str(split_path),
            'num_images': 0,
            'num_labels': 0,
            'image_formats': set(),
            'issues': []
        }

        # Count images
        if images_path.exists():
            image_files = list(images_path.glob('*.*'))
            info['num_images'] = len([f for f in image_files if f.suffix.lower() in [
                                     '.jpg', '.jpeg', '.png', '.bmp']])
            info['image_formats'] = set(f.suffix.lower()
                                        for f in image_files if f.is_file())

        # Count labels
        if labels_path.exists():
            label_files = list(labels_path.glob('*.txt'))
            info['num_labels'] = len(label_files)

        # Check for mismatches
        if info['num_images'] != info['num_labels']:
            info['issues'].append(
                f"Image-label mismatch: {info['num_images']} images vs {info['num_labels']} labels")

        return info

    def _detect_num_classes(self, ds_path: Path) -> int:
        """Auto-detect number of classes from label files."""
        max_class = -1

        for split in ['train', 'val', 'test']:
            labels_path = ds_path / split / 'labels'
            if labels_path.exists():
                for label_file in labels_path.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    max_class = max(max_class, class_id)
                    except:
                        pass

        return max_class + 1 if max_class >= 0 else 0

    def _validate_dataset(self, info: Dict) -> List[str]:
        """Validate dataset and return list of issues."""
        issues = []

        # Check if at least train split exists
        if 'train' not in info['splits']:
            issues.append("Missing training split")

        # Check if num_classes is valid
        if info['num_classes'] == 0:
            issues.append("Could not determine number of classes")

        # Check for empty splits
        for split, split_info in info['splits'].items():
            if split_info['num_images'] == 0:
                issues.append(f"Empty {split} split")

        return issues

    def _calculate_statistics(self, info: Dict) -> Dict:
        """Calculate dataset statistics."""
        stats = {
            'total_images': sum(s['num_images'] for s in info['splits'].values()),
            'total_labels': sum(s['num_labels'] for s in info['splits'].values()),
            'split_distribution': {}
        }

        # Calculate split percentages
        total = stats['total_images']
        if total > 0:
            for split, split_info in info['splits'].items():
                percentage = (split_info['num_images'] / total) * 100
                stats['split_distribution'][split] = f"{percentage:.1f}%"

        return stats

    def _read_yaml(self, yaml_path: Path) -> Dict:
        """Read YAML file safely."""
        try:
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except:
            return {}

    def generate_report(self, output_format: str = 'console') -> str:
        """Generate analysis report."""
        if output_format == 'console':
            return self._generate_console_report()
        elif output_format == 'json':
            return json.dumps(self.datasets, indent=2, default=str)
        elif output_format == 'markdown':
            return self._generate_markdown_report()
        else:
            raise ValueError(f"Unknown format: {output_format}")

    def _generate_console_report(self) -> str:
        """Generate console-friendly report."""
        lines = []
        lines.append("=" * 80)
        lines.append("📊 DATASET ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total datasets: {len(self.datasets)}")
        lines.append("")

        for ds_name, info in self.datasets.items():
            lines.append("─" * 80)
            lines.append(f"📁 Dataset: {ds_name}")
            lines.append("─" * 80)
            lines.append(f"  Path:         {info['path']}")
            lines.append(f"  Type:         {info['type']}")
            lines.append(f"  Format:       {info['format']}")
            lines.append(f"  Classes:      {info['num_classes']}")

            if info['class_names']:
                lines.append(
                    f"  Class names:  {', '.join(info['class_names'])}")

            lines.append("")
            lines.append("  📈 Statistics:")
            lines.append(
                f"    Total images: {info['statistics']['total_images']}")
            lines.append(
                f"    Total labels: {info['statistics']['total_labels']}")

            if info['statistics']['split_distribution']:
                lines.append("    Split distribution:")
                for split, pct in info['statistics']['split_distribution'].items():
                    split_info = info['splits'][split]
                    lines.append(
                        f"      {split:5s}: {split_info['num_images']:5d} images ({pct})")

            if info['issues']:
                lines.append("")
                lines.append("  ⚠️  Issues:")
                for issue in info['issues']:
                    lines.append(f"    - {issue}")
            else:
                lines.append("")
                lines.append("  ✅ No issues found")

            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Dataset Analysis Report")
        lines.append(
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Total Datasets:** {len(self.datasets)}\n")

        for ds_name, info in self.datasets.items():
            lines.append(f"\n## {ds_name}\n")
            lines.append(f"- **Path:** `{info['path']}`")
            lines.append(f"- **Type:** {info['type']}")
            lines.append(f"- **Format:** {info['format']}")
            lines.append(f"- **Classes:** {info['num_classes']}")

            if info['class_names']:
                lines.append(
                    f"- **Class Names:** {', '.join(info['class_names'])}")

            lines.append("\n### Statistics\n")
            lines.append(
                f"- Total images: {info['statistics']['total_images']}")
            lines.append(
                f"- Total labels: {info['statistics']['total_labels']}")

            if info['splits']:
                lines.append("\n### Data Splits\n")
                lines.append("| Split | Images | Percentage |")
                lines.append("|-------|--------|------------|")
                for split, split_info in info['splits'].items():
                    pct = info['statistics']['split_distribution'].get(
                        split, 'N/A')
                    lines.append(
                        f"| {split} | {split_info['num_images']} | {pct} |")

            if info['issues']:
                lines.append("\n### ⚠️ Issues\n")
                for issue in info['issues']:
                    lines.append(f"- {issue}")
            else:
                lines.append("\n### ✅ Validation\n\nNo issues found.")

        return "\n".join(lines)

    def generate_config_template(self, dataset_name: str) -> Dict:
        """Generate config template for a specific dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        info = self.datasets[dataset_name]

        template = {
            'dataset': {
                'name': dataset_name,
                'root': f"dataset/{dataset_name}" if info['type'] == 'raw' else f"processed_data/{dataset_name}",
                'format': info['format'],
                'num_classes': info['num_classes'],
                'class_names': info['class_names'],
            },
            'paths': {},
            'augmentation': {
                'enabled': True,
                'h_flip': 0.5,
                'v_flip': 0.0,
                'rotation': 10,
                'brightness': 0.2,
                'contrast': 0.2,
            },
            'training': {
                'epochs': 100,
                'batch_size': 8,
                'lr': 0.001,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
            }
        }

        # Add split paths
        for split in ['train', 'val', 'test']:
            if split in info['splits']:
                template['paths'][f'{split}_images'] = f"dataset/{dataset_name}/{split}/images"
                template['paths'][f'{split}_labels'] = f"dataset/{dataset_name}/{split}/labels"

        return template


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Dataset Discovery & Analysis')
    parser.add_argument('--dataset', '-d', help='Analyze specific dataset')
    parser.add_argument('--output', '-o', help='Output file (json/md/txt)')
    parser.add_argument('--format', '-f', choices=['console', 'json', 'markdown'],
                        default='console', help='Output format')
    parser.add_argument('--generate-config', '-c',
                        help='Generate config for dataset')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DatasetAnalyzer(Path.cwd())

    # Discover datasets
    datasets = analyzer.discover_datasets()

    # Generate report
    report = analyzer.generate_report(output_format=args.format)

    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"✅ Report saved to: {output_path}")
    else:
        print(report)

    # Generate config if requested
    if args.generate_config:
        config = analyzer.generate_config_template(args.generate_config)
        config_path = Path(f"configs/{args.generate_config}.yaml")
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n✅ Config template saved to: {config_path}")

    # Print summary
    print(f"\n📊 Summary: Found {len(datasets)} dataset(s)")
    for name, info in datasets.items():
        status = "✅" if not info['issues'] else "⚠️"
        print(
            f"  {status} {name}: {info['statistics']['total_images']} images, {info['num_classes']} classes")


if __name__ == '__main__':
    main()
