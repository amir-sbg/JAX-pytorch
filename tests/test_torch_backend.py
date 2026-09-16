import numpy as np
import pytest
import torch

from dual_backend.torch_backend import _resolve_device, predict_torch


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
