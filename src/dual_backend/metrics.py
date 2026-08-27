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


def prediction_agreement_summary(
    targets: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
) -> dict[str, float]:
    actual, first = _matching_arrays(targets, first_predictions)
    _, second = _matching_arrays(targets, second_predictions)
    if first.shape != second.shape:
        raise ValueError("prediction arrays must have the same shape")

    first_residuals = first - actual
    second_residuals = second - actual
    abs_gap = np.abs(second - first)
    first_abs_error = np.abs(first_residuals)
    second_abs_error = np.abs(second_residuals)
    return {
        "mean_abs_prediction_gap": float(np.mean(abs_gap)),
        "max_abs_prediction_gap": float(np.max(abs_gap)),
        "prediction_correlation": _safe_correlation(first, second),
        "residual_correlation": _safe_correlation(first_residuals, second_residuals),
        "residual_sign_disagreement_rate": float(
            np.mean(np.sign(first_residuals) != np.sign(second_residuals))
        ),
        "first_backend_lower_error_rate": float(np.mean(first_abs_error < second_abs_error)),
        "second_backend_lower_error_rate": float(np.mean(second_abs_error < first_abs_error)),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return 1.0 if np.allclose(first, second) else 0.0
    first_std = float(np.std(first))
    second_std = float(np.std(second))
    if first_std < 1e-12 or second_std < 1e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.corrcoef(first, second)[0, 1])


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
