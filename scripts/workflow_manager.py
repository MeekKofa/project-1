"""
Comprehensive Dataset Workflow Manager

Full pipeline: Analysis → Preprocessing → Training → Validation → Testing → Visualization
Argument-driven, robust, production-ready

Usage:
    # Full pipeline
    python workflow_manager.py --dataset cattlebody --stage all
    
    # Individual stages
    python workflow_manager.py --dataset cattle --stage analyze
    python workflow_manager.py --dataset cattle --stage preprocess
    python workfl        with open('src/config/config.yaml', 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

        logger.info("✅ Updated src/config/config.yaml with profile settings")anager.py --dataset cattle --stage train
    
    # With custom config
    pyt    parser.add_argument(
        '--config',
        type=str,
        default='src/config/config.yaml',
        help='Path to configuration YAML (contains dataset profiles)'
    )rkflow_manager.py --dataset cattlebody --stage preprocess --config config.yaml
"""

import argparse
import sys
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, List
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages the complete detection pipeline workflow."""

    def __init__(
        self,
        dataset_name: str,
        config_path: str = "src/config/config.yaml"
    ):
        """Initialize workflow manager."""
        self.dataset_name = dataset_name
        self.project_root = Path.cwd()

        # Load unified configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get dataset profile from unified config
        self.profile = self.config.get(
            'dataset_profiles', {}).get(dataset_name)
        if not self.profile:
            logger.warning(
                f"No profile found for {dataset_name}, using defaults")
            self.profile = self.profiles.get('defaults', {})

        # Setup paths
        self.analysis_dir = self.project_root / "dataset_analysis_results"
        self.results_dir = self.project_root / "workflow_results" / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Workflow state
        self.state_file = self.results_dir / "workflow_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load workflow state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'dataset': self.dataset_name,
            'stages_completed': [],
            'last_run': None,
            'status': 'pending'
        }

    def _save_state(self):
        """Save workflow state."""
        self.state['last_run'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _mark_stage_complete(self, stage: str, success: bool, details: Dict = None):
        """Mark a stage as complete."""
        stage_info = {
            'stage': stage,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }

        # Update or add stage
        existing = [s for s in self.state['stages_completed']
                    if s['stage'] == stage]
        if existing:
            self.state['stages_completed'].remove(existing[0])
        self.state['stages_completed'].append(stage_info)

        self._save_state()

    def run_stage(self, stage: str) -> bool:
        """Run a specific workflow stage."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STAGE: {stage.upper()}")
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"{'='*80}\n")

        stage_methods = {
            'check': self.stage_check,
            'analyze': self.stage_analyze,
            'preprocess': self.stage_preprocess,
            'train': self.stage_train,
            'validate': self.stage_validate,
            'test': self.stage_test,
            'visualize': self.stage_visualize,
            'all': self.stage_all
        }

        if stage not in stage_methods:
            logger.error(f"Unknown stage: {stage}")
            return False

        try:
            success = stage_methods[stage]()
            self._mark_stage_complete(stage, success)
            return success
        except Exception as e:
            logger.error(f"Stage {stage} failed: {e}", exc_info=True)
            self._mark_stage_complete(stage, False, {'error': str(e)})
            return False

    def stage_check(self) -> bool:
        """Stage 1: Check dataset health and requirements."""
        logger.info("📋 Checking dataset health...")

        # Check if dataset exists
        structure = self.profile.get('structure', {})
        raw_path = self.project_root / structure.get('raw_path', '')

        if not raw_path.exists():
            logger.error(f"Dataset not found: {raw_path}")
            return False

        logger.info(f"✅ Dataset found: {raw_path}")

        # Check for analysis results
        analysis_files = list(self.analysis_dir.glob(
            f"{self.dataset_name}_*.json"))
        if analysis_files:
            logger.info(f"✅ Found {len(analysis_files)} analysis file(s)")

            # Load and check quality
            for analysis_file in analysis_files:
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)

                issues = analysis.get('quality', {}).get('issues', [])
                warnings = analysis.get('quality', {}).get('warnings', [])

                if issues:
                    logger.warning(f"⚠️  {len(issues)} issue(s) found:")
                    for issue in issues:
                        logger.warning(f"   - {issue}")

                if warnings:
                    logger.warning(f"⚠️  {len(warnings)} warning(s):")
                    for warning in warnings:
                        logger.warning(f"   - {warning}")
        else:
            logger.warning(
                "⚠️  No analysis results found. Run 'analyze' stage first.")

        # Check dataset status from profile
        status = self.profile.get('status')
        if status == 'broken':
            logger.error("❌ Dataset is marked as BROKEN in profile!")
            quality_issues = self.profile.get(
                'analysis', {}).get('quality_issues', [])
            for issue in quality_issues:
                logger.error(f"   - {issue}")
            return False

        logger.info("✅ Dataset check complete")
        return True

    def stage_analyze(self) -> bool:
        """Stage 2: Run deep dataset analysis."""
        logger.info("🔬 Running deep dataset analysis...")

        # Check if analyze_datasets_deep.py exists
        analyze_script = self.project_root / "analyze_datasets_deep.py"
        if not analyze_script.exists():
            logger.error(f"Analysis script not found: {analyze_script}")
            return False

        # Run analysis
        try:
            result = subprocess.run(
                [sys.executable, str(analyze_script)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"Analysis failed:\n{result.stderr}")
                return False

            logger.info("✅ Analysis complete")
            logger.info(f"Results: {self.analysis_dir}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Analysis timed out (10 minutes)")
            return False
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return False

    def stage_preprocess(self) -> bool:
        """Stage 3: Preprocess dataset using profile configuration."""
        logger.info("🔧 Preprocessing dataset...")

        # Get preprocessing config
        preprocess_cfg = self.profile.get('preprocessing', {})

        if not preprocess_cfg.get('enabled', True):
            logger.warning("Preprocessing disabled in profile")
            return True

        # Check if preprocess script exists
        preprocess_script = self.project_root / "preprocess_dataset.py"
        if not preprocess_script.exists():
            logger.error(
                f"Preprocessing script not found: {preprocess_script}")
            return False

        # Determine split
        structure = self.profile.get('structure', {})
        split = 'raw' if 'raw' in structure.get(
            'raw_path', '') else 'processed'

        # Run preprocessing
        try:
            cmd = [
                sys.executable,
                str(preprocess_script),
                '--dataset', self.dataset_name,
                '--split', split,
                '--config', 'config.yaml',
                '--force'
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"Preprocessing failed:\n{result.stderr}")
                return False

            logger.info("✅ Preprocessing complete")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Preprocessing timed out (1 hour)")
            return False
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return False

    def stage_train(self) -> bool:
        """Stage 4: Train model."""
        logger.info("🚀 Training model...")

        # Get training config
        training_cfg = self.profile.get('training', {})

        if not training_cfg.get('enabled', True):
            logger.warning("Training disabled in profile")
            return True

        # Update config.yaml with dataset-specific settings
        self._update_config_for_training()

        # Check if training script exists
        train_script = self.project_root / "train.py"
        if not train_script.exists():
            logger.warning("train.py not found, using main.py")
            train_script = self.project_root / "main.py"
            if not train_script.exists():
                logger.error("No training script found (train.py or main.py)")
                return False

        # Run training
        try:
            cmd = [
                sys.executable,
                str(train_script),
                '--config', 'config.yaml',
                '--dataset', self.dataset_name
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                timeout=86400  # 24 hour timeout
            )

            if result.returncode != 0:
                logger.error("Training failed")
                return False

            logger.info("✅ Training complete")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Training timed out (24 hours)")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def stage_validate(self) -> bool:
        """Stage 5: Validate model."""
        logger.info("✓ Validating model...")

        # Validation is typically part of training
        # This stage is for explicit validation runs
        logger.info("Validation is performed during training")
        logger.info("Check training logs for validation metrics")
        return True

    def stage_test(self) -> bool:
        """Stage 6: Test model on test set."""
        logger.info("🧪 Testing model...")

        # Check for test script
        test_script = self.project_root / "test.py"
        if not test_script.exists():
            logger.warning("test.py not found, skipping")
            return True

        # Run testing
        try:
            cmd = [
                sys.executable,
                str(test_script),
                '--config', 'config.yaml',
                '--dataset', self.dataset_name
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600
            )

            if result.returncode != 0:
                logger.error(f"Testing failed:\n{result.stderr}")
                return False

            logger.info("✅ Testing complete")
            return True

        except Exception as e:
            logger.error(f"Testing failed: {e}")
            return False

    def stage_visualize(self) -> bool:
        """Stage 7: Generate visualizations."""
        logger.info("📊 Generating visualizations...")

        logger.info("Visualizations are generated during training")
        logger.info(f"Check outputs directory for results")
        return True

    def stage_all(self) -> bool:
        """Run all stages in sequence."""
        stages = ['check', 'analyze', 'preprocess',
                  'train', 'validate', 'test', 'visualize']

        for stage in stages:
            logger.info(f"\n{'='*80}")
            logger.info(f"RUNNING STAGE: {stage.upper()}")
            logger.info(f"{'='*80}\n")

            success = self.run_stage(stage)

            if not success:
                logger.error(f"Stage '{stage}' failed. Stopping pipeline.")
                return False

            logger.info(f"\n✅ Stage '{stage}' completed successfully\n")

        logger.info(f"\n{'='*80}")
        logger.info("🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*80}\n")
        return True

    def _update_config_for_training(self):
        """Update config.yaml with dataset-specific settings from profile."""
        training_cfg = self.profile.get('training', {})

        # Update main config with profile settings
        self.config['train']['epochs'] = training_cfg.get(
            'epochs', self.config['train']['epochs'])
        self.config['train']['batch_size'] = training_cfg.get(
            'batch_size', self.config['train']['batch_size'])
        self.config['train']['learning_rate'] = training_cfg.get(
            'learning_rate', self.config['train']['learning_rate'])

        # Update loss configuration
        loss_type = training_cfg.get('loss_type', 'auto')
        self.config['loss']['type'] = loss_type

        if loss_type == 'focal':
            self.config['loss']['focal_alpha'] = training_cfg.get(
                'focal_alpha', 0.25)
            self.config['loss']['focal_gamma'] = training_cfg.get(
                'focal_gamma', 2.0)

        # Save updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info("✅ Updated config.yaml with profile settings")

    def print_summary(self):
        """Print workflow summary."""
        print(f"\n{'='*80}")
        print(f"WORKFLOW SUMMARY: {self.dataset_name}")
        print(f"{'='*80}")

        print(f"\nDataset Profile:")
        print(f"  Task: {self.profile.get('task', 'N/A')}")
        print(f"  Modality: {self.profile.get('modality', 'N/A')}")
        print(f"  Status: {self.profile.get('status', 'active')}")

        analysis = self.profile.get('analysis', {})
        print(f"\nDataset Info:")
        print(f"  Classes: {analysis.get('num_classes', 'N/A')}")
        print(f"  Class Names: {analysis.get('class_names', 'N/A')}")
        print(f"  Format: {analysis.get('format', 'N/A')}")

        print(f"\nCompleted Stages:")
        for stage_info in self.state['stages_completed']:
            status = "✅" if stage_info['success'] else "❌"
            print(
                f"  {status} {stage_info['stage']} - {stage_info['timestamp']}")

        print(f"\nResults Directory:")
        print(f"  {self.results_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Dataset Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python workflow_manager.py --dataset cattlebody --stage all
  
  # Run specific stage
  python workflow_manager.py --dataset cattle --stage preprocess
  
  # With custom profile
  python workflow_manager.py --dataset cattle --stage train --profile custom_profiles.yaml
  
Stages:
  check      - Check dataset health and requirements
  analyze    - Run deep dataset analysis
  preprocess - Preprocess dataset with quality filters
  train      - Train detection model
  validate   - Validate model performance
  test       - Test model on test set
  visualize  - Generate visualizations
  all        - Run all stages in sequence
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['cattle', 'cattlebody', 'cattleface'],
        help='Dataset name'
    )

    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=['check', 'analyze', 'preprocess', 'train',
                 'validate', 'test', 'visualize', 'all'],
        help='Workflow stage to run'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--profile',
        type=str,
        default='dataset_profiles.yaml',
        help='Path to dataset profiles file'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print workflow summary and exit'
    )

    args = parser.parse_args()

    # Create workflow manager
    manager = WorkflowManager(
        dataset_name=args.dataset,
        config_path=args.config
    )

    # Print summary if requested
    if args.summary:
        manager.print_summary()
        sys.exit(0)

    # Run stage
    success = manager.run_stage(args.stage)

    # Print summary
    manager.print_summary()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
