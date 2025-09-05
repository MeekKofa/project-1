#!/usr/bin/env python3
"""
Cleanup script for removing old individual epoch metric files.
Keeps only essential files (final evaluations and CSV tracking).
"""

from src.evaluation.metrics import DetectionMetrics
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


def cleanup_metrics_directory(metrics_dir: Path, dry_run: bool = False):
    """
    Clean up metrics directory by removing old epoch files.

    Args:
        metrics_dir: Path to metrics directory
        dry_run: If True, only show what would be deleted
    """
    if not metrics_dir.exists():
        logger.error(f"Metrics directory not found: {metrics_dir}")
        return

    files_to_remove = []
    files_to_keep = []

    for file_path in metrics_dir.iterdir():
        if file_path.is_file():
            filename = file_path.name

            # Keep these files
            if any(pattern in filename for pattern in ['.csv', 'final', 'training_curves', 'README']):
                files_to_keep.append(file_path)
            # Remove epoch-specific files
            elif 'epoch_' in filename and (filename.endswith('.json') or filename.endswith('.txt')):
                files_to_remove.append(file_path)
            else:
                files_to_keep.append(file_path)

    print(f"\n📁 Analyzing directory: {metrics_dir}")
    print(f"📊 Total files found: {len(files_to_keep) + len(files_to_remove)}")
    print(f"✅ Files to keep: {len(files_to_keep)}")
    print(f"🗑️  Files to remove: {len(files_to_remove)}")

    if files_to_keep:
        print(f"\n✅ Keeping these files:")
        for file_path in sorted(files_to_keep):
            print(f"   • {file_path.name}")

    if files_to_remove:
        print(f"\n🗑️  {'Would remove' if dry_run else 'Removing'} these files:")
        for file_path in sorted(files_to_remove):
            print(f"   • {file_path.name}")

        if not dry_run:
            removed_count = 0
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Could not remove {file_path}: {e}")

            print(f"\n✅ Successfully removed {removed_count} files")

            # Add info file about cleanup
            info_file = metrics_dir / "cleanup_info.txt"
            with open(info_file, 'w') as f:
                from datetime import datetime
                f.write(
                    f"Metrics cleanup performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Files removed: {removed_count}\n")
                f.write(f"Files kept: {len(files_to_keep)}\n\n")
                f.write("Individual epoch files have been consolidated into:\n")
                f.write("• training_metrics.csv - All epoch metrics in CSV format\n")
                f.write("• *_final.json - Final evaluation results in JSON\n")
                f.write("• *_final_report.txt - Enhanced final evaluation report\n")
                f.write("• training_curves.png - Training progress visualization\n")

            print(f"📝 Cleanup info saved to: {info_file}")
    else:
        print("\n✨ No files need to be removed!")


def cleanup_all_model_metrics(outputs_dir: Path, dry_run: bool = False):
    """
    Clean up metrics for all model/dataset combinations.

    Args:
        outputs_dir: Path to outputs directory
        dry_run: If True, only show what would be deleted
    """
    if not outputs_dir.exists():
        logger.error(f"Outputs directory not found: {outputs_dir}")
        return

    metrics_dirs = []

    # Find all metrics directories
    for dataset_dir in outputs_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name != 'legacy':
            for model_dir in dataset_dir.iterdir():
                if model_dir.is_dir():
                    metrics_dir = model_dir / 'metrics'
                    if metrics_dir.exists():
                        metrics_dirs.append(metrics_dir)

    if not metrics_dirs:
        print("No metrics directories found!")
        return

    print(f"🔍 Found {len(metrics_dirs)} metrics directories to clean:")
    for metrics_dir in metrics_dirs:
        rel_path = metrics_dir.relative_to(outputs_dir)
        print(f"   • {rel_path}")

    print("\n" + "="*60)

    for i, metrics_dir in enumerate(metrics_dirs, 1):
        print(
            f"\n[{i}/{len(metrics_dirs)}] Processing: {metrics_dir.relative_to(outputs_dir)}")
        cleanup_metrics_directory(metrics_dir, dry_run)

    print("\n" + "="*60)
    print("🎉 Metrics cleanup complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up old individual epoch metric files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - see what would be deleted
  python cleanup_metrics.py --dry-run
  
  # Clean specific metrics directory
  python cleanup_metrics.py --dir outputs/cattlebody/faster_rcnn/metrics
  
  # Clean all metrics directories
  python cleanup_metrics.py --all
  
  # Force clean without confirmation
  python cleanup_metrics.py --all --force
        """
    )

    parser.add_argument('--dir', type=str,
                        help='Specific metrics directory to clean')
    parser.add_argument('--all', action='store_true',
                        help='Clean all metrics directories in outputs/')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompts')
    parser.add_argument('--outputs-dir', type=str, default='outputs',
                        help='Path to outputs directory (default: outputs)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    if not args.dir and not args.all:
        parser.print_help()
        return

    # Confirmation for non-dry-run operations
    if not args.dry_run and not args.force:
        if args.all:
            response = input(
                "\n⚠️  This will remove individual epoch files from ALL metrics directories. Continue? [y/N]: ")
        else:
            response = input(
                f"\n⚠️  This will remove individual epoch files from {args.dir}. Continue? [y/N]: ")

        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return

    # Perform cleanup
    if args.dir:
        metrics_dir = Path(args.dir)
        cleanup_metrics_directory(metrics_dir, args.dry_run)
    elif args.all:
        outputs_dir = Path(args.outputs_dir)
        cleanup_all_model_metrics(outputs_dir, args.dry_run)


if __name__ == "__main__":
    main()
