import numpy as np
import pytest

from dual_backend.data import fit_standardizer, inverse_targets, make_numpy_batches
from dual_backend.metrics import metric_delta, regression_metrics, residual_summary


def test_standardizer_round_trip() -> None:
    values = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)
    standardizer = fit_standardizer(values)

    restored = standardizer.inverse(standardizer.transform(values))

    np.testing.assert_allclose(restored, values, atol=1e-6)


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


def test_metrics_reject_nan_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([1.0]), np.array([np.nan]))


def test_metric_delta_uses_second_minus_first() -> None:
    assert metric_delta({"rmse": 2.0}, {"rmse": 1.5}) == {"rmse": -0.5}
