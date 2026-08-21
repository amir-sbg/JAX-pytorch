from pathlib import Path

import pytest

from dual_backend.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    experiment_config_from_dict,
    load_experiment_config,
)


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


def test_experiment_config_from_dict_builds_nested_config() -> None:
    config = experiment_config_from_dict(
        {
            "backend": "jax",
            "model": {"hidden_dims": [32, 16], "activation": "relu"},
            "training": {"epochs": 5, "learning_rate": 0.001},
            "report_dir": "tmp_reports",
        }
    )

    assert config.backend == "jax"
    assert config.model.hidden_dims == (32, 16)
    assert config.model.activation == "relu"
    assert config.training.epochs == 5
    assert config.training.learning_rate == pytest.approx(0.001)
    assert config.report_dir == Path("tmp_reports")


def test_load_experiment_config_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text('{"backend": "torch", "training": {"batch_size": 8}}', encoding="utf-8")

    config = load_experiment_config(path)

    assert config.backend == "torch"
    assert config.training.batch_size == 8
