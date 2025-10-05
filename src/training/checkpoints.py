"""
Checkpoint Manager - Handles saving and loading model checkpoints.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


class CheckpointManager:
    """
    Manages model checkpoints with support for best/latest/periodic saves.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best: bool = True,
        save_latest: bool = True,
        checkpoint_freq: int = 10,
        max_keep: Optional[int] = None
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_latest: Whether to save latest model
            checkpoint_freq: Save checkpoint every N epochs
            max_keep: Maximum number of periodic checkpoints to keep (None = keep all)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_latest = save_latest
        self.checkpoint_freq = checkpoint_freq
        self.max_keep = max_keep

        self.best_metric = None
        self.best_epoch = None

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False
    ):
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            metrics: Current metrics
            config: Training configuration
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
        }

        # Save best checkpoint
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint: {best_path}")
            self.best_metric = metrics.get(
                'val_loss', metrics.get('loss', None))
            self.best_epoch = epoch

        # Save latest checkpoint
        if self.save_latest:
            latest_path = self.checkpoint_dir / 'latest.pth'
            torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if epoch % self.checkpoint_freq == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"✓ Saved checkpoint: {epoch_path}")

            # Clean old checkpoints if max_keep is set
            if self.max_keep:
                self._clean_old_checkpoints()

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint on

        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore best metric tracking
        self.best_metric = checkpoint.get('best_metric')
        self.best_epoch = checkpoint.get('best_epoch')

        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        if self.best_metric:
            print(
                f"  Best metric: {self.best_metric:.4f} (epoch {self.best_epoch})")

        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
        }

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None if not found
        """
        latest_path = self.checkpoint_dir / 'latest.pth'
        return latest_path if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if not found
        """
        best_path = self.checkpoint_dir / 'best.pth'
        return best_path if best_path.exists() else None

    def _clean_old_checkpoints(self):
        """Remove old periodic checkpoints, keeping only the most recent max_keep."""
        if not self.max_keep:
            return

        # Get all periodic checkpoints (epoch_*.pth)
        checkpoints = sorted(
            self.checkpoint_dir.glob('epoch_*.pth'),
            key=lambda p: int(p.stem.split('_')[1])
        )

        # Remove oldest checkpoints if we have too many
        while len(checkpoints) > self.max_keep:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            print(f"✓ Removed old checkpoint: {old_checkpoint.name}")

    def list_checkpoints(self) -> Dict[str, Path]:
        """
        List all available checkpoints.

        Returns:
            Dictionary mapping checkpoint types to paths
        """
        checkpoints = {}

        # Best checkpoint
        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            checkpoints['best'] = best_path

        # Latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        if latest_path.exists():
            checkpoints['latest'] = latest_path

        # Periodic checkpoints
        for epoch_path in sorted(self.checkpoint_dir.glob('epoch_*.pth')):
            epoch = int(epoch_path.stem.split('_')[1])
            checkpoints[f'epoch_{epoch}'] = epoch_path

        return checkpoints

    def get_resume_checkpoint(self) -> Optional[Path]:
        """
        Get the checkpoint to resume from (latest if available).

        Returns:
            Path to checkpoint or None
        """
        return self.get_latest_checkpoint()


# Convenience functions
def save_model(
    model: torch.nn.Module,
    save_path: Path,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save just the model weights (for inference).

    Args:
        model: Model to save
        save_path: Path to save to
        additional_info: Optional additional information to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
    }

    if additional_info:
        save_dict.update(additional_info)

    torch.save(save_dict, save_path)
    print(f"✓ Saved model: {save_path}")


def load_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load model weights (for inference).

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"✓ Loaded model from: {checkpoint_path}")
    return model
