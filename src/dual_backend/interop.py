from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def torch_state_to_jax_params(state_dict: Mapping[str, object]) -> tuple[dict[str, np.ndarray], ...]:
    weight_keys = sorted(
        (key for key in state_dict if key.endswith("weight")),
        key=_layer_sort_key,
    )
    params = []
    for weight_key in weight_keys:
        prefix = weight_key[: -len("weight")]
        bias_key = prefix + "bias"
        if bias_key not in state_dict:
            raise ValueError(f"missing bias tensor for {weight_key}")
        params.append(
            {
                "weight": _to_numpy(state_dict[weight_key]).T.astype(np.float32),
                "bias": _to_numpy(state_dict[bias_key]).astype(np.float32),
            }
        )
    if not params:
        raise ValueError("state_dict does not contain linear layer weights")
    return tuple(params)


def jax_params_to_torch_state(params: tuple[dict[str, np.ndarray], ...]) -> dict[str, np.ndarray]:
    state = {}
    for index, layer in enumerate(params):
        linear_position = index * 2
        state[f"{linear_position}.weight"] = np.asarray(layer["weight"], dtype=np.float32).T
        state[f"{linear_position}.bias"] = np.asarray(layer["bias"], dtype=np.float32)
    return state


def numpy_mlp_forward(
    params: tuple[dict[str, np.ndarray], ...],
    features: np.ndarray,
    activation: str = "gelu",
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    for layer in params[:-1]:
        values = values @ np.asarray(layer["weight"], dtype=np.float32) + np.asarray(
            layer["bias"],
            dtype=np.float32,
        )
        values = _activation(values, activation)
    output = values @ np.asarray(params[-1]["weight"], dtype=np.float32) + np.asarray(
        params[-1]["bias"],
        dtype=np.float32,
    )
    return output[:, 0].astype(np.float32)


def parity_report(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float = 1e-5,
) -> dict[str, float | bool]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("arrays must have matching shapes for parity")
    absolute = np.abs(left - right)
    return {
        "max_abs_diff": float(np.max(absolute)),
        "mean_abs_diff": float(np.mean(absolute)),
        "within_tolerance": bool(np.max(absolute) <= tolerance),
        "tolerance": float(tolerance),
    }


def _activation(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "gelu":
        return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)))
    if activation == "tanh":
        return np.tanh(values)
    raise ValueError("unsupported activation")


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _layer_sort_key(key: str) -> tuple[int, str]:
    for part in key.split("."):
        if part.isdigit():
            return int(part), key
    return 0, key
