from __future__ import annotations

from pathlib import Path

from dual_backend.config import ExperimentConfig
from dual_backend.data import load_tabular_dataset
from dual_backend.interop import (
    numpy_mlp_forward,
    parameter_tree_parity_report,
    parity_report,
    torch_state_to_jax_params,
)
from dual_backend.jax_backend import apply_jax
from dual_backend.metrics import save_json
from dual_backend.torch_backend import load_torch_model, predict_torch


def check_torch_to_jax_layout(
    checkpoint_path: Path,
    config: ExperimentConfig,
    output_path: Path,
    tolerance: float = 1e-5,
    device: str = "auto",
    gradient_batch_size: int = 32,
) -> dict:
    config.validate()
    if gradient_batch_size < 1:
        raise ValueError("gradient_batch_size must be positive")
    dataset = load_tabular_dataset(
        dataset=config.dataset,
        csv_path=config.csv_path,
        target_column=config.target_column,
        validation_size=config.validation_size,
        test_size=config.test_size,
        seed=config.training.seed,
    )
    model, resolved_device, model_config = load_torch_model(checkpoint_path, device=device)
    checkpoint_training = _checkpoint_training_settings(checkpoint_path)
    torch_predictions = predict_torch(model, dataset.x_test, resolved_device)
    jax_layout_params = torch_state_to_jax_params(model.state_dict())
    converted_predictions = numpy_mlp_forward(
        jax_layout_params,
        dataset.x_test,
        activation=model_config.activation,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "converted_layers": len(jax_layout_params),
        "examples": int(len(dataset.x_test)),
        "parity": parity_report(torch_predictions, converted_predictions, tolerance=tolerance),
        "gradient_parity": _gradient_parity(
            model,
            jax_layout_params,
            dataset.x_train[:gradient_batch_size],
            dataset.y_train[:gradient_batch_size],
            activation=model_config.activation,
            loss_name=str(checkpoint_training.get("loss", config.training.loss)),
            huber_delta=float(checkpoint_training.get("huber_delta", config.training.huber_delta)),
            tolerance=tolerance,
        ),
        "input_gradient_parity": _input_gradient_parity(
            model,
            jax_layout_params,
            dataset.x_train[:gradient_batch_size],
            dataset.y_train[:gradient_batch_size],
            activation=model_config.activation,
            loss_name=str(checkpoint_training.get("loss", config.training.loss)),
            huber_delta=float(checkpoint_training.get("huber_delta", config.training.huber_delta)),
            tolerance=tolerance,
        ),
    }
    save_json(report, output_path)
    return report


def _checkpoint_training_settings(checkpoint_path: Path) -> dict:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    training = payload.get("training", {})
    return training if isinstance(training, dict) else {}


def _gradient_parity(
    model,
    jax_layout_params,
    features,
    targets,
    *,
    activation: str,
    loss_name: str,
    huber_delta: float,
    tolerance: float,
) -> dict[str, float | bool]:
    import jax
    import jax.numpy as jnp
    import torch

    device = next(model.parameters()).device
    torch_features = torch.as_tensor(features, dtype=torch.float32, device=device)
    torch_targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    model.train()
    model.zero_grad(set_to_none=True)
    torch_predictions = model(torch_features).squeeze(-1)
    residuals = torch_predictions - torch_targets
    if loss_name == "mse":
        torch_loss = torch.mean(residuals**2)
    elif loss_name == "huber":
        absolute = torch.abs(residuals)
        quadratic = torch.minimum(absolute, torch.tensor(huber_delta, device=device))
        linear = absolute - quadratic
        torch_loss = torch.mean(0.5 * quadratic**2 + huber_delta * linear)
    else:
        raise ValueError("loss_name must be mse or huber")
    torch_loss.backward()
    torch_gradients = torch_state_to_jax_params(
        {name: parameter.grad for name, parameter in model.named_parameters()}
    )

    jax_params = tuple(
        {
            "weight": jnp.asarray(layer["weight"], dtype=jnp.float32),
            "bias": jnp.asarray(layer["bias"], dtype=jnp.float32),
        }
        for layer in jax_layout_params
    )
    jax_features = jnp.asarray(features, dtype=jnp.float32)
    jax_targets = jnp.asarray(targets, dtype=jnp.float32)

    def loss_function(params):
        predictions = apply_jax(params, jax_features, activation)
        errors = predictions - jax_targets
        if loss_name == "mse":
            return jnp.mean(errors**2)
        absolute = jnp.abs(errors)
        quadratic = jnp.minimum(absolute, huber_delta)
        linear = absolute - quadratic
        return jnp.mean(0.5 * quadratic**2 + huber_delta * linear)

    jax_gradients = jax.grad(loss_function)(jax_params)
    return parameter_tree_parity_report(
        torch_gradients,
        jax_gradients,
        tolerance=tolerance,
    )


def _input_gradient_parity(
    model,
    jax_layout_params,
    features,
    targets,
    *,
    activation: str,
    loss_name: str,
    huber_delta: float,
    tolerance: float,
) -> dict[str, float | bool | int]:
    """Compare dLoss/dInput after converting a PyTorch checkpoint to JAX layout."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    device = next(model.parameters()).device
    torch_features = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)
    torch_targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    torch_predictions = model(torch_features).squeeze(-1)
    torch_errors = torch_predictions - torch_targets
    if loss_name == "mse":
        torch_loss = torch.mean(torch_errors**2)
    elif loss_name == "huber":
        absolute = torch.abs(torch_errors)
        quadratic = torch.minimum(
            absolute,
            torch.tensor(huber_delta, device=device),
        )
        torch_loss = torch.mean(
            0.5 * quadratic**2 + huber_delta * (absolute - quadratic)
        )
    else:
        raise ValueError("loss_name must be mse or huber")
    torch_loss.backward()
    torch_gradient = torch_features.grad.detach().cpu().numpy().reshape(-1)

    jax_params = tuple(
        {
            "weight": jnp.asarray(layer["weight"], dtype=jnp.float32),
            "bias": jnp.asarray(layer["bias"], dtype=jnp.float32),
        }
        for layer in jax_layout_params
    )
    jax_targets = jnp.asarray(targets, dtype=jnp.float32)

    def loss_from_features(input_features):
        predictions = apply_jax(jax_params, input_features, activation)
        errors = predictions - jax_targets
        if loss_name == "mse":
            return jnp.mean(errors**2)
        absolute = jnp.abs(errors)
        quadratic = jnp.minimum(absolute, huber_delta)
        return jnp.mean(0.5 * quadratic**2 + huber_delta * (absolute - quadratic))

    jax_gradient = np.asarray(
        jax.grad(loss_from_features)(jnp.asarray(features, dtype=jnp.float32))
    ).reshape(-1)
    difference = jax_gradient - torch_gradient
    denominator = max(float(np.linalg.norm(torch_gradient)), 1e-12)
    norm_product = float(np.linalg.norm(torch_gradient) * np.linalg.norm(jax_gradient))
    cosine = (
        float(np.dot(torch_gradient, jax_gradient) / norm_product)
        if norm_product > 0.0
        else 1.0
    )
    maximum_error = float(np.max(np.abs(difference)))
    return {
        "values_compared": int(torch_gradient.size),
        "max_absolute_error": maximum_error,
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "relative_l2_error": float(np.linalg.norm(difference) / denominator),
        "cosine_similarity": cosine,
        "within_tolerance": maximum_error <= tolerance,
    }
