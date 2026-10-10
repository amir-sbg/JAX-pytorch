from __future__ import annotations

from math import cos, pi


def learning_rate_for_epoch(
    epoch: int,
    total_epochs: int,
    base_learning_rate: float,
    schedule: str = "constant",
    warmup_epochs: int = 0,
    min_learning_rate_ratio: float = 0.05,
) -> float:
    if total_epochs < 1 or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be within the configured training range")
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("schedule must be constant or cosine")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be between 0 and total_epochs - 1")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be between 0 and 1")

    if warmup_epochs and epoch <= warmup_epochs:
        return base_learning_rate * epoch / warmup_epochs
    if schedule == "constant":
        return base_learning_rate

    decay_epochs = total_epochs - warmup_epochs
    decay_index = epoch - warmup_epochs - 1
    progress = decay_index / max(decay_epochs - 1, 1)
    cosine = 0.5 * (1.0 + cos(pi * progress))
    multiplier = min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine
    return base_learning_rate * multiplier
