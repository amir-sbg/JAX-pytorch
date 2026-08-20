from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    actual, estimated = _matching_arrays(targets, predictions)
    residuals = estimated - actual
    mse = float(np.mean(residuals**2))
    variance = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 if variance == 0.0 and mse == 0.0 else 0.0
    if variance > 0.0:
        r2 = 1.0 - float(np.sum(residuals**2) / variance)
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }


def residual_summary(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    actual, estimated = _matching_arrays(targets, predictions)
    residuals = estimated - actual
    return {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "std": float(np.std(residuals)),
        "p10": float(np.percentile(residuals, 10)),
        "p90": float(np.percentile(residuals, 90)),
        "max_abs": float(np.max(np.abs(residuals))),
    }


def metric_delta(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(first) & set(second))
    return {key: float(second[key] - first[key]) for key in keys}


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _matching_arrays(targets: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(targets, dtype=np.float64)
    estimated = np.asarray(predictions, dtype=np.float64)
    if actual.shape != estimated.shape:
        raise ValueError("targets and predictions must have the same shape")
    if actual.size == 0:
        raise ValueError("targets and predictions must not be empty")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(estimated)):
        raise ValueError("targets and predictions must contain only finite values")
    return actual, estimated
