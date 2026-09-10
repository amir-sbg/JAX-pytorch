import numpy as np
import pytest

from dual_backend.data import fit_standardizer, inverse_targets, make_numpy_batches
from dual_backend.metrics import (
    error_by_prediction_quantile,
    error_by_target_quantile,
    learning_curve_summary,
    metric_delta,
    mlp_capacity_summary,
    prediction_agreement_summary,
    regression_metrics,
    residual_summary,
    training_diagnostics,
)


def test_standardizer_round_trip() -> None:
    values = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)
    standardizer = fit_standardizer(values)

    restored = standardizer.inverse(standardizer.transform(values))

    np.testing.assert_allclose(restored, values, atol=1e-6)


def test_mlp_capacity_summary_counts_parameters_and_ops() -> None:
    summary = mlp_capacity_summary(input_dim=4, hidden_dims=(8, 2))

    assert summary["hidden_layers"] == 2
    assert summary["trainable_parameters"] == 4 * 8 + 8 + 8 * 2 + 2 + 2 * 1 + 1
    assert summary["linear_multiply_adds_per_sample"] == 4 * 8 + 8 * 2 + 2 * 1


def test_make_numpy_batches_covers_all_rows() -> None:
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    targets = np.arange(10, dtype=np.float32)
    rng = np.random.default_rng(4)

    batches = make_numpy_batches(features, targets, batch_size=4, rng=rng)

    assert len(batches) == 3
    assert sum(len(batch_targets) for _, batch_targets in batches) == 10


def test_make_numpy_batches_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="matching rows"):
        make_numpy_batches(np.zeros((2, 2)), np.zeros(3), 2, np.random.default_rng(0))


def test_regression_metrics_and_residuals() -> None:
    targets = np.array([1.0, 2.0, 3.0])
    predictions = np.array([1.0, 2.5, 2.0])

    metrics = regression_metrics(targets, predictions)
    residuals = residual_summary(targets, predictions)

    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.0 + 0.25 + 1.0) / 3.0))
    assert residuals["max_abs"] == 1.0


def test_error_by_target_quantile_tracks_bias_by_range() -> None:
    targets = np.array([0.0, 1.0, 2.0, 3.0])
    predictions = np.array([0.0, 1.5, 1.5, 5.0])

    rows = error_by_target_quantile(targets, predictions, bins=2)

    assert [row["rows"] for row in rows] == [2, 2]
    assert rows[0]["mae"] == pytest.approx(0.25)
    assert rows[1]["mean_error"] == pytest.approx(0.75)


def test_error_by_prediction_quantile_tracks_forecast_ranges() -> None:
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.array([1.0, 2.5, 2.5, 5.0])

    rows = error_by_prediction_quantile(targets, predictions, bins=2)

    assert [row["rows"] for row in rows] == [1, 3]
    assert rows[0]["prediction_min"] == pytest.approx(1.0)
    assert rows[1]["prediction_max"] == pytest.approx(5.0)


def test_error_by_target_quantile_rejects_single_bin() -> None:
    with pytest.raises(ValueError, match="bins"):
        error_by_target_quantile(np.array([1.0, 2.0]), np.array([1.0, 2.0]), bins=1)
    with pytest.raises(ValueError, match="bins"):
        error_by_prediction_quantile(np.array([1.0, 2.0]), np.array([1.0, 2.0]), bins=1)


def test_metrics_reject_nan_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([1.0]), np.array([np.nan]))


def test_metric_delta_uses_second_minus_first() -> None:
    assert metric_delta({"rmse": 2.0}, {"rmse": 1.5}) == {"rmse": -0.5}


def test_prediction_agreement_summarizes_backend_differences() -> None:
    targets = np.array([1.0, 2.0, 3.0])
    torch_predictions = np.array([1.2, 1.8, 2.9])
    jax_predictions = np.array([1.1, 2.2, 3.4])

    report = prediction_agreement_summary(targets, torch_predictions, jax_predictions)

    assert report["mean_abs_prediction_gap"] == pytest.approx((0.1 + 0.4 + 0.5) / 3)
    assert -1.0 <= report["prediction_correlation"] <= 1.0
    assert report["residual_sign_disagreement_rate"] == pytest.approx(2 / 3)
    assert report["first_backend_lower_error_rate"] > 0


def test_learning_curve_summary_tracks_best_epoch_and_gap() -> None:
    history = [
        {"epoch": 1.0, "train_loss": 0.8, "validation_loss": 0.9},
        {"epoch": 2.0, "train_loss": 0.5, "validation_loss": 0.55},
        {"epoch": 3.0, "train_loss": 0.4, "validation_loss": 0.62},
    ]

    summary = learning_curve_summary(history)

    assert summary["epochs_ran"] == 3
    assert summary["best_epoch"] == 2
    assert summary["validation_loss_improvement"] == pytest.approx(0.35)
    assert summary["final_generalization_gap"] == pytest.approx(0.22)


def test_learning_curve_summary_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="train_loss"):
        learning_curve_summary([{"epoch": 1.0, "validation_loss": 0.2}])


def test_training_diagnostics_flags_validation_rebound() -> None:
    history = [
        {"epoch": 1.0, "train_loss": 1.0, "validation_loss": 1.2},
        {"epoch": 2.0, "train_loss": 0.7, "validation_loss": 0.8},
        {"epoch": 3.0, "train_loss": 0.5, "validation_loss": 1.0},
    ]

    diagnostics = training_diagnostics(history)

    assert diagnostics["epochs_after_best"] == 1
    assert diagnostics["best_epoch_is_final"] is False
    assert diagnostics["validation_rebound"] == pytest.approx(0.2)
    assert diagnostics["train_loss_drop_fraction"] == pytest.approx(0.5)
    assert diagnostics["overfit_flag"] is True
