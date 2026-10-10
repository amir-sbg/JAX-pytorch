import numpy as np
import pytest
import torch

from dual_backend.torch_backend import _mean_gradient_norm, _resolve_device, predict_torch


def test_torch_prediction_rejects_invalid_features() -> None:
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="non-empty"):
        predict_torch(model, np.zeros((0, 2), dtype=np.float32), torch.device("cpu"))

    values = np.zeros((2, 2), dtype=np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predict_torch(model, values, torch.device("cpu"))


def test_torch_device_check_matches_runtime() -> None:
    if torch.cuda.is_available():
        assert _resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="not available"):
            _resolve_device("cuda")


def test_gradient_norm_summary_uses_all_batches() -> None:
    assert _mean_gradient_norm([1.0, 3.0, 5.0]) == pytest.approx(3.0)
    assert _mean_gradient_norm([]) == 0.0
