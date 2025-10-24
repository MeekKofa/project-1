"""Helpers for resolving output directories for dataset/model artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class ModelArtifactPaths:
    """Describes the standard artifact layout under ``outputs/<dataset>/<model>``."""

    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    predictions: Path
    visualizations: Path

    def ensure(self, *names: str) -> "ModelArtifactPaths":
        """Ensure one or more subdirectories exist.

        Args:
            *names: attribute names to create (e.g. ``"logs"``, ``"metrics"``).

        Returns:
            Self, to allow fluent usage.
        """
        for name in names:
            path = getattr(self, name, None)
            if path is None:
                raise AttributeError(f"Unknown artifact segment '{name}'")
            path.mkdir(parents=True, exist_ok=True)
        return self

    def iter_subdirs(self) -> Iterable[Path]:
        """Yield all registered subdirectory paths."""
        yield self.checkpoints
        yield self.logs
        yield self.metrics
        yield self.predictions
        yield self.visualizations


def resolve_model_artifact_paths(
    dataset: str,
    model: str,
    base_dir: Optional[str] = None,
) -> ModelArtifactPaths:
    """Resolve the canonical output layout for a dataset/model combination.

    Directories are not created automatically. Use :meth:`ModelArtifactPaths.ensure`
    to create whichever segments you actually need at runtime.

    Args:
        dataset: Dataset identifier (e.g. ``"cattle"``).
        model: Model identifier (e.g. ``"faster_rcnn"``).
        base_dir: Optional override for the root outputs directory. Defaults to
            ``"outputs"`` if ``None``.

    Returns:
        A :class:`ModelArtifactPaths` instance describing the layout.
    """
    if not dataset:
        raise ValueError("dataset must be provided to resolve artifact paths")
    if not model:
        raise ValueError("model must be provided to resolve artifact paths")

    root = Path(base_dir or "outputs").expanduser().resolve() / dataset / model

    return ModelArtifactPaths(
        root=root,
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        metrics=root / "metrics",
        predictions=root / "predictions",
        visualizations=root / "visualizations",
    )
