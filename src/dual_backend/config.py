from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    activation: str = "gelu"

    def validate(self) -> None:
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer")
        if any(width < 2 for width in self.hidden_dims):
            raise ValueError("hidden_dims values must be at least 2")
        if self.activation not in {"relu", "gelu", "tanh"}:
            raise ValueError("activation must be one of: relu, gelu, tanh")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 20
    gradient_clip: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str = "both"
    dataset: str = "diabetes"
    csv_path: Path | None = None
    target_column: str | None = None
    validation_size: float = 0.20
    test_size: float = 0.20
    output_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    def validate(self) -> None:
        if self.backend not in {"torch", "jax", "both"}:
            raise ValueError("backend must be torch, jax, or both")
        if self.dataset not in {"diabetes", "csv"}:
            raise ValueError("dataset must be diabetes or csv")
        if self.dataset == "csv" and (self.csv_path is None or self.target_column is None):
            raise ValueError("csv_path and target_column are required for csv datasets")
        if not 0 < self.validation_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("validation_size and test_size must be between 0 and 1")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size and test_size must sum to less than 1")
        self.model.validate()
        self.training.validate()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["csv_path"] = str(self.csv_path) if self.csv_path else None
        payload["output_dir"] = str(self.output_dir)
        payload["report_dir"] = str(self.report_dir)
        payload["model"]["hidden_dims"] = list(self.model.hidden_dims)
        return payload
