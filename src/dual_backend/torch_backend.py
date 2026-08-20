from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from dual_backend.config import ModelConfig, TrainingConfig
from dual_backend.data import TabularDataset
from dual_backend.results import BackendResult


def _build_torch_model(input_dim: int, config: ModelConfig):
    import torch
    from torch import nn

    activation = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }[config.activation]

    layers: list[nn.Module] = []
    previous = input_dim
    for width in config.hidden_dims:
        layers.append(nn.Linear(previous, width))
        layers.append(activation())
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def train_torch(
    dataset: TabularDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    device: str = "auto",
) -> BackendResult:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    training_config.validate()
    model_config.validate()
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    resolved_device = _resolve_device(device)
    model = _build_torch_model(dataset.input_dim, model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    train_loader = _make_loader(dataset, training_config, torch)
    x_validation = torch.as_tensor(dataset.x_validation, dtype=torch.float32, device=resolved_device)
    y_validation = torch.as_tensor(dataset.y_validation, dtype=torch.float32, device=resolved_device)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        seen = 0
        for features, targets in train_loader:
            features = features.to(resolved_device)
            targets = targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features).squeeze(-1)
            loss = loss_fn(predictions, targets)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(features)
            seen += len(features)

        validation_loss = _torch_loss(model, x_validation, y_validation, loss_fn)
        train_loss = train_loss_sum / max(seen, 1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "gradient_norm": float(grad_norm),
            }
        )

        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= training_config.patience:
            break

    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "torch_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "hidden_dims": list(model_config.hidden_dims),
                "activation": model_config.activation,
                "input_dim": dataset.input_dim,
            },
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )
    predictions = predict_torch(model, dataset.x_test, resolved_device)
    return BackendResult(
        backend="torch",
        predictions_scaled=predictions,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation_loss),
        train_seconds=float(train_seconds),
        checkpoint_path=checkpoint_path,
        extra={"device": str(resolved_device), "epochs_ran": len(history)},
    )


def predict_torch(model, features: np.ndarray, device) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        predictions = model(tensor).squeeze(-1)
    return predictions.detach().cpu().numpy().astype(np.float32)


def load_torch_model(checkpoint_path: Path, device: str = "auto"):
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload["model_config"]
    model_config = ModelConfig(
        hidden_dims=tuple(config["hidden_dims"]),
        activation=config["activation"],
    )
    resolved_device = _resolve_device(device)
    model = _build_torch_model(config["input_dim"], model_config)
    model.load_state_dict(payload["state_dict"])
    return model.to(resolved_device), resolved_device, model_config


def _make_loader(dataset: TabularDataset, config: TrainingConfig, torch):
    from torch.utils.data import DataLoader, TensorDataset

    features = torch.as_tensor(dataset.x_train, dtype=torch.float32)
    targets = torch.as_tensor(dataset.y_train, dtype=torch.float32)
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )


def _torch_loss(model, features, targets, loss_fn) -> float:
    import torch

    model.eval()
    with torch.inference_mode():
        predictions = model(features).squeeze(-1)
        return float(loss_fn(predictions, targets))


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
