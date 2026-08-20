from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from dual_backend.config import ModelConfig, TrainingConfig
from dual_backend.data import TabularDataset, make_numpy_batches
from dual_backend.results import BackendResult


def init_jax_params(input_dim: int, config: ModelConfig, key):
    import jax
    import jax.numpy as jnp

    config.validate()
    layer_sizes = (input_dim, *config.hidden_dims, 1)
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for in_features, out_features, layer_key in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        scale = jnp.sqrt(jnp.asarray(2.0 / max(in_features, 1), dtype=jnp.float32))
        params.append(
            {
                "weight": jax.random.normal(layer_key, (in_features, out_features)) * scale,
                "bias": jnp.zeros((out_features,), dtype=jnp.float32),
            }
        )
    return tuple(params)


def apply_jax(params, features, activation: str = "gelu"):
    import jax.nn as jnn
    import jax.numpy as jnp

    activations = features
    for layer in params[:-1]:
        activations = activations @ layer["weight"] + layer["bias"]
        if activation == "relu":
            activations = jnn.relu(activations)
        elif activation == "gelu":
            activations = jnn.gelu(activations)
        elif activation == "tanh":
            activations = jnp.tanh(activations)
        else:
            raise ValueError("unsupported activation")
    output_layer = params[-1]
    outputs = activations @ output_layer["weight"] + output_layer["bias"]
    return outputs[:, 0]


def train_jax(
    dataset: TabularDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: Path,
) -> BackendResult:
    import jax
    import jax.numpy as jnp

    model_config.validate()
    training_config.validate()

    key = jax.random.PRNGKey(training_config.seed)
    params = init_jax_params(dataset.input_dim, model_config, key)
    optimizer_state = _init_adam_state(params)
    rng = np.random.default_rng(training_config.seed)

    x_train = dataset.x_train.astype(np.float32)
    y_train = dataset.y_train.astype(np.float32)
    x_validation = jnp.asarray(dataset.x_validation, dtype=jnp.float32)
    y_validation = jnp.asarray(dataset.y_validation, dtype=jnp.float32)

    best_params = params
    best_validation_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        batches = make_numpy_batches(x_train, y_train, training_config.batch_size, rng, shuffle=True)
        batch_losses = []
        grad_norms = []
        for features, targets in batches:
            params, optimizer_state, loss, grad_norm = _train_step(
                params,
                optimizer_state,
                jnp.asarray(features, dtype=jnp.float32),
                jnp.asarray(targets, dtype=jnp.float32),
                training_config.learning_rate,
                training_config.weight_decay,
                training_config.gradient_clip,
                model_config.activation,
            )
            batch_losses.append(float(loss))
            grad_norms.append(float(grad_norm))

        validation_loss = float(
            _mse_loss(params, x_validation, y_validation, model_config.activation)
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(batch_losses)),
                "validation_loss": validation_loss,
                "gradient_norm": float(np.mean(grad_norms)),
            }
        )

        if validation_loss < best_validation_loss - 1e-7:
            best_params = params
            best_validation_loss = validation_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= training_config.patience:
            break

    train_seconds = time.perf_counter() - start_time
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "jax_params.npz"
    save_jax_params(best_params, checkpoint_path)
    predictions = predict_jax(best_params, dataset.x_test, model_config.activation)
    return BackendResult(
        backend="jax",
        predictions_scaled=predictions,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation_loss),
        train_seconds=float(train_seconds),
        checkpoint_path=checkpoint_path,
        extra={"backend": jax.default_backend(), "epochs_ran": len(history)},
    )


def predict_jax(params, features: np.ndarray, activation: str = "gelu") -> np.ndarray:
    import jax.numpy as jnp

    predictions = apply_jax(params, jnp.asarray(features, dtype=jnp.float32), activation)
    return np.asarray(predictions, dtype=np.float32)


def save_jax_params(params, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, layer in enumerate(params):
        arrays[f"layer_{index}_weight"] = np.asarray(layer["weight"], dtype=np.float32)
        arrays[f"layer_{index}_bias"] = np.asarray(layer["bias"], dtype=np.float32)
    np.savez(path, **arrays)


def load_jax_params(path: Path):
    import jax.numpy as jnp

    with np.load(path, allow_pickle=False) as arrays:
        indices = sorted(
            int(name.split("_")[1])
            for name in arrays.files
            if name.startswith("layer_") and name.endswith("_weight")
        )
        if indices != list(range(len(indices))):
            raise ValueError("checkpoint layers must be consecutive")
        params = []
        for index in indices:
            weight_key = f"layer_{index}_weight"
            bias_key = f"layer_{index}_bias"
            if bias_key not in arrays.files:
                raise ValueError(f"checkpoint is missing {bias_key}")
            params.append(
                {
                    "weight": jnp.asarray(arrays[weight_key], dtype=jnp.float32),
                    "bias": jnp.asarray(arrays[bias_key], dtype=jnp.float32),
                }
            )
    return tuple(params)


def _mse_loss(params, features, targets, activation: str):
    import jax.numpy as jnp

    predictions = apply_jax(params, features, activation)
    residuals = predictions - targets
    return jnp.mean(residuals**2)


def _loss_with_decay(params, features, targets, activation: str, weight_decay: float):
    import jax.numpy as jnp

    penalty = sum(jnp.sum(layer["weight"] ** 2) for layer in params)
    return _mse_loss(params, features, targets, activation) + weight_decay * penalty


def _init_adam_state(params):
    import jax
    import jax.numpy as jnp

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "step": jnp.array(0, dtype=jnp.int32)}


def _tree_l2_norm(tree):
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum((jnp.sum(leaf**2) for leaf in leaves), jnp.array(0.0)))


def _clip_gradients(grads, max_norm: float):
    import jax
    import jax.numpy as jnp

    norm = _tree_l2_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda grad: grad * scale, grads), norm


def _adamw_update(params, grads, state, learning_rate: float):
    import jax
    import jax.numpy as jnp

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    step = state["step"] + 1
    m = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state["m"], grads)
    v = jax.tree_util.tree_map(lambda old, grad: beta2 * old + (1.0 - beta2) * (grad**2), state["v"], grads)
    m_correction = 1.0 - jnp.power(beta1, step)
    v_correction = 1.0 - jnp.power(beta2, step)
    m_hat = jax.tree_util.tree_map(lambda value: value / m_correction, m)
    v_hat = jax.tree_util.tree_map(lambda value: value / v_correction, v)
    params = jax.tree_util.tree_map(
        lambda param, mh, vh: param - learning_rate * mh / (jnp.sqrt(vh) + epsilon),
        params,
        m_hat,
        v_hat,
    )
    return params, {"m": m, "v": v, "step": step}


def _train_step_impl(
    params,
    optimizer_state,
    features,
    targets,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    activation: str,
):
    import jax

    loss, grads = jax.value_and_grad(_loss_with_decay)(
        params,
        features,
        targets,
        activation,
        weight_decay,
    )
    grads, grad_norm = _clip_gradients(grads, gradient_clip)
    params, optimizer_state = _adamw_update(params, grads, optimizer_state, learning_rate)
    return params, optimizer_state, loss, grad_norm


_TRAIN_STEP = None


def _train_step(*args):
    global _TRAIN_STEP
    if _TRAIN_STEP is None:
        import jax

        _TRAIN_STEP = jax.jit(_train_step_impl, static_argnames=("activation",))
    return _TRAIN_STEP(*args)
