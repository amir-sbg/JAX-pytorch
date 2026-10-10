import pytest

from dual_backend.schedules import learning_rate_for_epoch


def test_cosine_schedule_warms_up_and_reaches_minimum() -> None:
    values = [
        learning_rate_for_epoch(
            epoch,
            total_epochs=6,
            base_learning_rate=0.01,
            schedule="cosine",
            warmup_epochs=2,
            min_learning_rate_ratio=0.1,
        )
        for epoch in range(1, 7)
    ]

    assert values[0] == pytest.approx(0.005)
    assert values[1] == pytest.approx(0.01)
    assert values[2] == pytest.approx(0.01)
    assert values[-1] == pytest.approx(0.001)
    assert values[2:] == sorted(values[2:], reverse=True)


def test_constant_schedule_only_applies_warmup() -> None:
    assert learning_rate_for_epoch(1, 4, 0.02, warmup_epochs=2) == pytest.approx(0.01)
    assert learning_rate_for_epoch(3, 4, 0.02, warmup_epochs=2) == pytest.approx(0.02)
