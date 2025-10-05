"""Utility helpers for rendering training and validation metric plots."""

from pathlib import Path
from typing import List, Dict, Any, Iterable
from importlib import import_module


class MetricsPlotter:
    """Generate common metric visualizations for training workflows."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        train_history: List[Dict[str, Any]],
        val_history: List[Dict[str, Any]],
        epoch_history: List[Dict[str, Any]]
    ) -> None:
        """Refresh all plots using the latest metric history."""
        try:
            plt = import_module('matplotlib.pyplot')
        except ModuleNotFoundError:
            return

        plt.style.use('ggplot')

        self._plot_loss_curves(train_history, val_history, plt)
        self._plot_loss_components(train_history, prefix='train', plt=plt)
        self._plot_loss_components(val_history, prefix='val', plt=plt)
        self._plot_detection_metrics(val_history, plt)
        self._plot_learning_rate(epoch_history, plt)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_loss_curves(
        self,
        train_history: List[Dict[str, Any]],
        val_history: List[Dict[str, Any]],
        plt
    ) -> None:
        train_epochs, train_losses = self._extract_metric(train_history, 'loss')
        val_epochs, val_losses = self._extract_metric(val_history, 'loss')

        if not train_losses and not val_losses:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        if train_losses:
            ax.plot(train_epochs, train_losses, marker='o', label='Train Loss')
        if val_losses:
            ax.plot(val_epochs, val_losses, marker='o', label='Validation Loss')

        ax.set_title('Loss Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        fig.savefig(self.output_dir / 'loss_curves.png')
        plt.close(fig)

    def _plot_loss_components(
        self,
        history: List[Dict[str, Any]],
        prefix: str,
        plt
    ) -> None:
        if not history:
            return

        components = [
            key for key in ('loss_box', 'loss_cls', 'loss_obj')
            if any(key in row for row in history)
        ]
        if not components:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        for component in components:
            comp_epochs, values = self._extract_metric(history, component)
            if not values:
                continue
            ax.plot(comp_epochs, values, marker='o', label=component)

        if not ax.lines:
            plt.close(fig)
            return

        title = f"{prefix.capitalize()} Loss Components"
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        fig.savefig(self.output_dir / f'{prefix}_loss_components.png')
        plt.close(fig)

    def _plot_detection_metrics(self, val_history: List[Dict[str, Any]], plt) -> None:
        if not val_history:
            return

        metrics = ['map_50', 'precision_50', 'recall_50', 'f1_50']
        available = [m for m in metrics if any(m in row for row in val_history)]
        if not available:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in available:
            metric_epochs, values = self._extract_metric(val_history, metric)
            if not values:
                continue
            ax.plot(metric_epochs, values, marker='o', label=metric)

        if not ax.lines:
            plt.close(fig)
            return

        ax.set_title('Validation Detection Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        fig.savefig(self.output_dir / 'val_detection_metrics.png')
        plt.close(fig)

    def _plot_learning_rate(self, epoch_history: List[Dict[str, Any]], plt) -> None:
        if not epoch_history:
            return

        epochs, lr_values = self._extract_metric(epoch_history, 'lr')
        if not lr_values:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, lr_values, marker='o', color='tab:purple')
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        fig.savefig(self.output_dir / 'learning_rate.png')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metric(history: Iterable[Dict[str, Any]], key: str):
        epochs: List[int] = []
        values: List[float] = []
        for row in history:
            if key not in row or 'epoch' not in row:
                continue
            value = row[key]
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            epochs.append(int(row['epoch']))
            values.append(value)
        return epochs, values
