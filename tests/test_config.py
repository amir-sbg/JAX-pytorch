from pathlib import Path

import pytest

from dual_backend.config import ExperimentConfig, ModelConfig, TrainingConfig


def test_experiment_config_serializes_paths_and_hidden_dims() -> None:
    config = ExperimentConfig(
        output_dir=Path("out"),
        report_dir=Path("rep"),
        model=ModelConfig(hidden_dims=(16, 8)),
    )

    payload = config.to_dict()

    assert payload["output_dir"] == "out"
    assert payload["report_dir"] == "rep"
    assert payload["model"]["hidden_dims"] == [16, 8]


def test_config_requires_csv_target_details() -> None:
    with pytest.raises(ValueError, match="csv_path"):
        ExperimentConfig(dataset="csv").validate()


def test_training_config_rejects_bad_optimizer_values() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0).validate()
    with pytest.raises(ValueError, match="gradient_clip"):
        TrainingConfig(gradient_clip=0).validate()


def test_model_config_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        ModelConfig(activation="swish").validate()
