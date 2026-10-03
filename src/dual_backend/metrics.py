from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def mlp_capacity_summary(input_dim: int, hidden_dims: tuple[int, ...]) -> dict[str, int]:
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if not hidden_dims or any(width < 1 for width in hidden_dims):
        raise ValueError("hidden_dims must contain positive widths")

    layer_sizes = (input_dim, *hidden_dims, 1)
    linear_parameters = 0
    multiply_adds = 0
    for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
        linear_parameters += in_features * out_features + out_features
        multiply_adds += in_features * out_features
    return {
        "input_dim": input_dim,
        "hidden_layers": len(hidden_dims),
        "trainable_parameters": linear_parameters,
        "linear_multiply_adds_per_sample": multiply_adds,
    }


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


def regression_calibration_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    actual, estimated = _matching_arrays(targets, predictions)
    design = np.column_stack([np.ones_like(estimated), estimated])
    coefficients = np.linalg.lstsq(design, actual, rcond=None)[0]
    calibrated = design @ coefficients
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "mean_prediction_bias": float(np.mean(estimated - actual)),
        "calibrated_rmse": float(np.sqrt(np.mean((calibrated - actual) ** 2))),
    }


def error_by_target_quantile(
    targets: np.ndarray,
    predictions: np.ndarray,
    bins: int = 4,
) -> list[dict[str, float | int]]:
    actual, estimated = _matching_arrays(targets, predictions)
    if bins < 2:
        raise ValueError("bins must be at least 2")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(actual, quantiles)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (actual >= lower) & (actual <= upper)
        else:
            mask = (actual >= lower) & (actual < upper)
        if not np.any(mask):
            continue
        residuals = estimated[mask] - actual[mask]
        rows.append(
            {
                "bin": index + 1,
                "rows": int(np.sum(mask)),
                "target_min": float(np.min(actual[mask])),
                "target_max": float(np.max(actual[mask])),
                "mae": float(np.mean(np.abs(residuals))),
                "mean_error": float(np.mean(residuals)),
            }
        )
    return rows


def error_by_prediction_quantile(
    targets: np.ndarray,
    predictions: np.ndarray,
    bins: int = 4,
) -> list[dict[str, float | int]]:
    actual, estimated = _matching_arrays(targets, predictions)
    if bins < 2:
        raise ValueError("bins must be at least 2")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(estimated, quantiles)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (estimated >= lower) & (estimated <= upper)
        else:
            mask = (estimated >= lower) & (estimated < upper)
        if not np.any(mask):
            continue
        residuals = estimated[mask] - actual[mask]
        rows.append(
            {
                "bin": index + 1,
                "rows": int(np.sum(mask)),
                "prediction_min": float(np.min(estimated[mask])),
                "prediction_max": float(np.max(estimated[mask])),
                "mae": float(np.mean(np.abs(residuals))),
                "mean_error": float(np.mean(residuals)),
            }
        )
    return rows


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


def paired_rmse_bootstrap(
    targets: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    actual, first = _matching_arrays(targets, first_predictions)
    _, second = _matching_arrays(targets, second_predictions)
    if first.shape != second.shape:
        raise ValueError("prediction arrays must have the same shape")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, actual.size, size=(samples, actual.size))
    actual_samples = actual[indices]
    first_rmse = np.sqrt(np.mean((first[indices] - actual_samples) ** 2, axis=1))
    second_rmse = np.sqrt(np.mean((second[indices] - actual_samples) ** 2, axis=1))
    deltas = second_rmse - first_rmse
    tail = (1.0 - confidence) / 2.0
    observed_delta = regression_metrics(actual, second)["rmse"] - regression_metrics(actual, first)["rmse"]
    return {
        "observed_rmse_delta_second_minus_first": float(observed_delta),
        "rmse_delta_ci_low": float(np.quantile(deltas, tail)),
        "rmse_delta_ci_high": float(np.quantile(deltas, 1.0 - tail)),
        "probability_second_lower_rmse": float(np.mean(deltas < 0.0)),
    }


def learning_curve_summary(history: list[dict[str, float]]) -> dict[str, float | int]:
    if not history:
        return {
            "epochs_ran": 0,
            "best_epoch": 0,
            "initial_train_loss": 0.0,
            "final_train_loss": 0.0,
            "initial_validation_loss": 0.0,
            "final_validation_loss": 0.0,
            "best_validation_loss": 0.0,
            "validation_loss_improvement": 0.0,
            "final_generalization_gap": 0.0,
        }
    for row in history:
        if "train_loss" not in row or "validation_loss" not in row:
            raise ValueError("history rows must include train_loss and validation_loss")
    history_values = np.asarray(
        [[row["train_loss"], row["validation_loss"]] for row in history],
        dtype=float,
    )
    if not np.all(np.isfinite(history_values)):
        raise ValueError("history losses must contain only finite values")

    first = history[0]
    last = history[-1]
    best_index, best = min(
        enumerate(history),
        key=lambda item: float(item[1]["validation_loss"]),
    )
    return {
        "epochs_ran": len(history),
        "best_epoch": int(best.get("epoch", best_index + 1)),
        "initial_train_loss": float(first["train_loss"]),
        "final_train_loss": float(last["train_loss"]),
        "initial_validation_loss": float(first["validation_loss"]),
        "final_validation_loss": float(last["validation_loss"]),
        "best_validation_loss": float(best["validation_loss"]),
        "validation_loss_improvement": float(
            first["validation_loss"] - best["validation_loss"]
        ),
        "final_generalization_gap": float(last["validation_loss"] - last["train_loss"]),
    }


def training_diagnostics(
    history: list[dict[str, float]],
    overfit_tolerance: float = 0.10,
) -> dict[str, float | int | bool]:
    if overfit_tolerance < 0:
        raise ValueError("overfit_tolerance must not be negative")
    if not history:
        return {
            "epochs_after_best": 0,
            "best_epoch_is_final": False,
            "validation_rebound": 0.0,
            "relative_validation_rebound": 0.0,
            "train_loss_drop_fraction": 0.0,
            "overfit_flag": False,
        }
    for row in history:
        if "train_loss" not in row or "validation_loss" not in row:
            raise ValueError("history rows must include train_loss and validation_loss")
    history_values = np.asarray(
        [[row["train_loss"], row["validation_loss"]] for row in history],
        dtype=float,
    )
    if not np.all(np.isfinite(history_values)):
        raise ValueError("history losses must contain only finite values")

    best_index, best = min(
        enumerate(history),
        key=lambda item: float(item[1]["validation_loss"]),
    )
    first = history[0]
    last = history[-1]
    best_validation = float(best["validation_loss"])
    final_validation = float(last["validation_loss"])
    validation_rebound = final_validation - best_validation
    initial_train = float(first["train_loss"])
    final_train = float(last["train_loss"])
    train_drop = initial_train - final_train
    relative_rebound = validation_rebound / max(abs(best_validation), 1e-12)
    return {
        "epochs_after_best": len(history) - best_index - 1,
        "best_epoch_is_final": best_index == len(history) - 1,
        "validation_rebound": validation_rebound,
        "relative_validation_rebound": relative_rebound,
        "train_loss_drop_fraction": train_drop / max(abs(initial_train), 1e-12),
        "overfit_flag": bool(
            relative_rebound > overfit_tolerance
            and final_train <= float(best["train_loss"])
        ),
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
