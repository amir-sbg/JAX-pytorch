from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BackendResult:
    backend: str
    predictions_scaled: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    train_seconds: float
    checkpoint_path: Path
    extra: dict[str, float | int | str]
