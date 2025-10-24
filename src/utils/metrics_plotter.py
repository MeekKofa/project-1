"""Utility helpers for rendering training and validation metric plots."""

import math
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

        train_history = self._normalize_history(train_history)
        val_history = self._normalize_history(val_history)
        epoch_history = self._normalize_history(epoch_history)

        max_epoch = self._max_epoch(train_history, val_history, epoch_history)
        if max_epoch == 0:
            max_epoch = None

        self._plot_loss_curves(train_history, val_history, plt, max_epoch)
        self._plot_loss_components(train_history, prefix='train', plt=plt, max_epoch=max_epoch)
        self._plot_loss_components(val_history, prefix='val', plt=plt, max_epoch=max_epoch)
        self._plot_detection_metrics(val_history, plt, max_epoch=max_epoch)
        self._plot_scalar_metrics(epoch_history, plt, max_epoch=max_epoch)
        self._plot_learning_rate(epoch_history, plt, max_epoch=max_epoch)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_loss_curves(
        self,
        train_history: List[Dict[str, Any]],
        val_history: List[Dict[str, Any]],
        plt,
        max_epoch: Optional[int]
    ) -> None:
        train_epochs, train_losses = self._extract_metric(train_history, 'loss', max_epoch)
        val_epochs, val_losses = self._extract_metric(val_history, 'loss', max_epoch)

        if not self._has_values(train_losses) and not self._has_values(val_losses):
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        if self._has_values(train_losses):
            ax.plot(train_epochs, train_losses, marker='o', label='Train Loss')
        if self._has_values(val_losses):
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
        plt,
        max_epoch: Optional[int]
    ) -> None:
        if not history:
            return

        components = sorted({
            key
            for row in history
            for key, value in row.items()
            if key != 'epoch' and key.startswith('loss') and isinstance(value, (int, float))
        })
        if not components:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        for component in components:
            comp_epochs, values = self._extract_metric(history, component, max_epoch)
            if not self._has_values(values):
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

    def _plot_detection_metrics(
        self,
        val_history: List[Dict[str, Any]],
        plt,
        max_epoch: Optional[int]
    ) -> None:
        if not val_history:
            return

        metrics = ['map_50', 'precision_50', 'recall_50', 'f1_50']
        available = [m for m in metrics if any(m in row for row in val_history)]
        if not available:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in available:
            metric_epochs, values = self._extract_metric(val_history, metric, max_epoch)
            if not self._has_values(values):
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

    def _plot_scalar_metrics(
        self,
        epoch_history: List[Dict[str, Any]],
        plt,
        max_epoch: Optional[int]
    ) -> None:
        if not epoch_history:
            return

        scalar_keys = sorted({
            key
            for row in epoch_history
            for key, value in row.items()
            if key not in {'epoch', 'lr'} and isinstance(value, (int, float))
        })

        if not scalar_keys:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        plotted = False

        for key in scalar_keys:
            epochs, values = self._extract_metric(epoch_history, key, max_epoch)
            if not self._has_values(values):
                continue
            ax.plot(epochs, values, marker='o', label=key)
            plotted = True

        if not plotted:
            plt.close(fig)
            return

        ax.set_title('Training Metrics Overview')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        fig.savefig(self.output_dir / 'training_metrics_overview.png')
        plt.close(fig)

    def _plot_learning_rate(
        self,
        epoch_history: List[Dict[str, Any]],
        plt,
        max_epoch: Optional[int]
    ) -> None:
        if not epoch_history:
            return

        epochs, lr_values = self._extract_metric(epoch_history, 'lr', max_epoch)
        if not self._has_values(lr_values):
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
    def _extract_metric(
        history: Iterable[Dict[str, Any]],
        key: str,
        max_epoch: Optional[int]
    ) -> Tuple[List[int], List[float]]:
        epoch_to_value: Dict[int, float] = {}
        for row in history:
            if key not in row:
                continue
            epoch_val = row.get('epoch')
            if epoch_val is None:
                continue
            try:
                epoch_int = int(epoch_val)
            except (TypeError, ValueError):
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            epoch_to_value[epoch_int] = value

        if not epoch_to_value:
            return [], []

        if max_epoch is None:
            epochs = sorted(epoch_to_value.keys())
        else:
            epochs = list(range(1, max_epoch + 1))

        values: List[float] = []
        last_value: Optional[float] = None
        for epoch in epochs:
            value = epoch_to_value.get(epoch)
            if value is None and last_value is not None:
                values.append(last_value)
            elif value is None:
                values.append(float('nan'))
            else:
                values.append(value)
                last_value = value

        if not any(isinstance(val, (int, float)) and not math.isnan(float(val)) for val in values):
            return [], []

        return epochs, values

    @staticmethod
    def _normalize_history(history: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: Dict[int, Dict[str, Any]] = {}
        for row in history or []:
            epoch_val = row.get('epoch')
            if epoch_val is None:
                continue
            try:
                epoch_int = int(epoch_val)
            except (TypeError, ValueError):
                continue
            normalized_row = dict(row)
            normalized_row['epoch'] = epoch_int
            normalized[epoch_int] = normalized_row
        return [normalized[idx] for idx in sorted(normalized.keys())]

    @staticmethod
    def _max_epoch(*histories: Iterable[Dict[str, Any]]) -> int:
        max_epoch = 0
        for history in histories:
            for row in history:
                epoch_val = row.get('epoch')
                if epoch_val is None:
                    continue
                try:
                    epoch_int = int(epoch_val)
                except (TypeError, ValueError):
                    continue
                if epoch_int > max_epoch:
                    max_epoch = epoch_int
        return max_epoch

    @staticmethod
    def _has_values(values: List[float]) -> bool:
        if not values:
            return False
        for val in values:
            try:
                numeric = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric):
                continue
            return True
        return False
