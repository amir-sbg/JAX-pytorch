from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)


@dataclass(frozen=True)
class TabularDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    feature_standardizer: Standardizer
    target_standardizer: Standardizer

    @property
    def input_dim(self) -> int:
        return int(self.x_train.shape[1])


def load_tabular_dataset(
    dataset: str = "diabetes",
    csv_path: Path | None = None,
    target_column: str | None = None,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    seed: int = 42,
) -> TabularDataset:
    _validate_split_sizes(validation_size, test_size)
    if dataset == "diabetes":
        features, targets, feature_names, target_name = _load_diabetes()
    elif dataset == "csv":
        if csv_path is None or target_column is None:
            raise ValueError("csv_path and target_column are required for csv datasets")
        features, targets, feature_names, target_name = _load_csv(csv_path, target_column)
    else:
        raise ValueError("dataset must be diabetes or csv")

    split = _split_arrays(features, targets, validation_size, test_size, seed)
    x_train, y_train = split["train"]
    x_validation, y_validation = split["validation"]
    x_test, y_test = split["test"]

    feature_standardizer = fit_standardizer(x_train)
    target_standardizer = fit_standardizer(y_train.reshape(-1, 1))

    return TabularDataset(
        x_train=feature_standardizer.transform(x_train),
        y_train=target_standardizer.transform(y_train.reshape(-1, 1))[:, 0],
        x_validation=feature_standardizer.transform(x_validation),
        y_validation=target_standardizer.transform(y_validation.reshape(-1, 1))[:, 0],
        x_test=feature_standardizer.transform(x_test),
        y_test=target_standardizer.transform(y_test.reshape(-1, 1))[:, 0],
        feature_names=feature_names,
        target_name=target_name,
        feature_standardizer=feature_standardizer,
        target_standardizer=target_standardizer,
    )


def fit_standardizer(values: np.ndarray) -> Standardizer:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("standardizer expects a 2-D array")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return Standardizer(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def make_numpy_batches(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(features) != len(targets):
        raise ValueError("features and targets must have matching rows")
    indices = np.arange(len(features))
    if shuffle:
        rng.shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batches.append((features[batch_idx], targets[batch_idx]))
    return batches


def inverse_targets(dataset: TabularDataset, values: np.ndarray) -> np.ndarray:
    return dataset.target_standardizer.inverse(np.asarray(values).reshape(-1, 1))[:, 0]


def _load_diabetes() -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes()
    return (
        bunch.data.astype(np.float32),
        bunch.target.astype(np.float32),
        tuple(str(name) for name in bunch.feature_names),
        "disease_progression",
    )


def _load_csv(
    csv_path: Path,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if target_column not in frame.columns:
        raise ValueError(f"target_column {target_column!r} is not in the CSV")
    numeric = frame.select_dtypes(include=["number"]).copy()
    if target_column not in numeric.columns:
        raise ValueError("target_column must be numeric")
    numeric = numeric.dropna(axis=0).reset_index(drop=True)
    if len(numeric) < 10:
        raise ValueError("dataset must have at least 10 complete numeric rows")
    targets = numeric[target_column].to_numpy(dtype=np.float32)
    features = numeric.drop(columns=[target_column])
    return (
        features.to_numpy(dtype=np.float32),
        targets,
        tuple(str(name) for name in features.columns),
        target_column,
    )


def _split_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    validation_size: float,
    test_size: float,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if len(features) != len(targets):
        raise ValueError("features and targets must have matching rows")
    if len(features) < 10:
        raise ValueError("dataset must contain at least 10 rows")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    test_count = max(1, int(round(len(indices) * test_size)))
    validation_count = max(1, int(round(len(indices) * validation_size)))
    train_count = len(indices) - test_count - validation_count
    if train_count < 1:
        raise ValueError("split sizes leave no training rows")

    train_idx = indices[:train_count]
    validation_idx = indices[train_count : train_count + validation_count]
    test_idx = indices[train_count + validation_count :]
    return {
        "train": (features[train_idx], targets[train_idx]),
        "validation": (features[validation_idx], targets[validation_idx]),
        "test": (features[test_idx], targets[test_idx]),
    }


def _validate_split_sizes(validation_size: float, test_size: float) -> None:
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must sum to less than 1")
