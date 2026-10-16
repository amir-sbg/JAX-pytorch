from __future__ import annotations

import csv
from pathlib import Path

from dual_backend.config import ExperimentConfig
from dual_backend.data import TabularDataset, inverse_targets, load_tabular_dataset
from dual_backend.metrics import (
    error_by_target_quantile,
    gradient_norm_summary,
    error_by_prediction_quantile,
    learning_curve_summary,
    metric_delta,
    mlp_capacity_summary,
    paired_rmse_bootstrap,
    paired_absolute_error_sign_test,
    prediction_agreement_summary,
    regression_calibration_summary,
    regression_metrics,
    residual_summary,
    save_json,
    training_diagnostics,
)
from dual_backend.results import BackendResult


def run_experiment(config: ExperimentConfig) -> dict:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.report_dir / "run_summary.json"
    predictions_path = config.report_dir / "predictions.csv"

    dataset = load_tabular_dataset(
        dataset=config.dataset,
        csv_path=config.csv_path,
        target_column=config.target_column,
        validation_size=config.validation_size,
        test_size=config.test_size,
        seed=config.training.seed,
    )
    results: list[BackendResult] = []
    if config.backend in {"torch", "both"}:
        from dual_backend.torch_backend import train_torch

        results.append(
            train_torch(
                dataset,
                config.model,
                config.training,
                config.output_dir,
            )
        )
    if config.backend in {"jax", "both"}:
        from dual_backend.jax_backend import train_jax

        results.append(
            train_jax(
                dataset,
                config.model,
                config.training,
                config.output_dir,
            )
        )

    backend_reports = {
        result.backend: _result_report(
            dataset,
            result,
            config.report_dir,
            config.training.gradient_clip,
        )
        for result in results
    }
    predictions_by_backend = {
        result.backend: inverse_targets(dataset, result.predictions_scaled)
        for result in results
    }
    comparison = _comparison_report(
        dataset,
        backend_reports,
        predictions_by_backend,
    )
    summary = {
        "config": config.to_dict(),
        "dataset": _dataset_summary(dataset),
        "model_capacity": mlp_capacity_summary(
            dataset.input_dim,
            config.model.hidden_dims,
        ),
        "backends": backend_reports,
        "comparison": comparison,
        "selection": _selection_summary(backend_reports),
        "report_path": str(summary_path),
        "predictions_path": str(predictions_path),
    }
    save_json(summary, summary_path)
    _write_predictions(dataset, results, predictions_path)
    return summary


def _result_report(
    dataset: TabularDataset,
    result: BackendResult,
    report_dir: Path,
    gradient_clip: float,
) -> dict:
    actual = inverse_targets(dataset, dataset.y_test)
    predictions = inverse_targets(dataset, result.predictions_scaled)
    metrics = regression_metrics(actual, predictions)
    residuals = residual_summary(actual, predictions)
    history_path = report_dir / f"{result.backend}_history.csv"
    _write_history(result.history, history_path)
    return {
        "metrics": metrics,
        "calibration": regression_calibration_summary(actual, predictions),
        "residuals": residuals,
        "target_quantile_errors": error_by_target_quantile(actual, predictions),
        "prediction_quantile_errors": error_by_prediction_quantile(
            actual,
            predictions,
        ),
        "best_epoch": result.best_epoch,
        "best_validation_loss_scaled": result.best_validation_loss,
        "train_seconds": result.train_seconds,
        "checkpoint_path": str(result.checkpoint_path),
        "history_path": str(history_path),
        "learning_curve": learning_curve_summary(result.history),
        "training_diagnostics": training_diagnostics(result.history),
        "gradient_diagnostics": gradient_norm_summary(result.history, gradient_clip),
        "extra": result.extra,
    }


def _comparison_report(
    dataset: TabularDataset,
    reports: dict[str, dict],
    predictions_by_backend: dict[str, object],
) -> dict:
    if "torch" not in reports or "jax" not in reports:
        return {}
    actual = inverse_targets(dataset, dataset.y_test)
    return {
        "metric_delta_jax_minus_torch": metric_delta(
            reports["torch"]["metrics"],
            reports["jax"]["metrics"],
        ),
        "train_seconds_delta_jax_minus_torch": float(
            reports["jax"]["train_seconds"] - reports["torch"]["train_seconds"]
        ),
        "prediction_agreement": prediction_agreement_summary(
            actual,
            predictions_by_backend["torch"],
            predictions_by_backend["jax"],
        ),
        "paired_rmse_bootstrap": paired_rmse_bootstrap(
            actual,
            predictions_by_backend["torch"],
            predictions_by_backend["jax"],
        ),
        "paired_absolute_error_sign_test": paired_absolute_error_sign_test(
            actual,
            predictions_by_backend["torch"],
            predictions_by_backend["jax"],
        ),
    }


def _selection_summary(reports: dict[str, dict]) -> dict[str, str | float]:
    if not reports:
        return {}
    metric = "rmse"
    best_backend, best_report = min(
        reports.items(),
        key=lambda item: float(item[1]["metrics"][metric]),
    )
    return {
        "best_backend_by_rmse": best_backend,
        "best_rmse": float(best_report["metrics"][metric]),
    }


def _dataset_summary(dataset: TabularDataset) -> dict:
    return {
        "target": dataset.target_name,
        "features": list(dataset.feature_names),
        "rows": {
            "train": int(len(dataset.x_train)),
            "validation": int(len(dataset.x_validation)),
            "test": int(len(dataset.x_test)),
        },
        "input_dim": dataset.input_dim,
    }


def _write_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(
    dataset: TabularDataset,
    results: list[BackendResult],
    path: Path,
) -> None:
    actual = inverse_targets(dataset, dataset.y_test)
    rows = []
    converted = {
        result.backend: inverse_targets(dataset, result.predictions_scaled)
        for result in results
    }
    for index, target in enumerate(actual):
        row = {"row": index, "target": float(target)}
        for backend, predictions in converted.items():
            row[f"{backend}_prediction"] = float(predictions[index])
            row[f"{backend}_residual"] = float(predictions[index] - target)
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["row", "target"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
