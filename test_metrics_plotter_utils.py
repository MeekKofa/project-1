"""Unit tests for MetricsPlotter utility helpers."""

import math
from pathlib import Path

from src.utils.metrics_plotter import MetricsPlotter


def test_extract_metric_includes_missing_epochs(tmp_path: Path) -> None:
    plotter = MetricsPlotter(tmp_path)
    history = [
        {'epoch': 1, 'loss': 1.0},
        {'epoch': 3, 'loss': 0.8},
    ]

    epochs, values = plotter._extract_metric(history, 'loss', max_epoch=3)

    assert epochs == [1, 2, 3]
    assert math.isclose(values[0], 1.0)
    assert math.isnan(values[1])
    assert math.isclose(values[2], 0.8)


def test_normalize_history_prefers_latest(tmp_path: Path) -> None:
    history = [
        {'epoch': 1, 'loss': 1.0},
        {'epoch': '1', 'loss': 0.9},
        {'epoch': 2, 'loss': 0.7},
    ]

    normalized = MetricsPlotter._normalize_history(history)

    assert len(normalized) == 2
    assert normalized[0]['epoch'] == 1
    assert normalized[0]['loss'] == 0.9
    assert normalized[1]['epoch'] == 2


def test_has_values_filters_nan() -> None:
    assert not MetricsPlotter._has_values([float('nan'), float('nan')])
    assert MetricsPlotter._has_values([float('nan'), 0.0, 1.0])
