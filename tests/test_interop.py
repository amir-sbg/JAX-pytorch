import numpy as np
import pytest

from dual_backend.interop import (
    activation_health_report,
    directional_derivative_report,
    jax_params_to_torch_state,
    numpy_mlp_forward,
    parameter_tree_parity_report,
    parity_report,
    torch_state_to_jax_params,
)
from dual_backend.parity import _hessian_vector_parity, _input_gradient_parity


def test_torch_state_to_jax_params_transposes_linear_weights() -> None:
    state = {
        "0.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "0.bias": np.array([0.5, -0.5], dtype=np.float32),
        "2.weight": np.array([[1.0, -1.0]], dtype=np.float32),
        "2.bias": np.array([0.25], dtype=np.float32),
    }

    params = torch_state_to_jax_params(state)

    assert params[0]["weight"].shape == (2, 2)
    np.testing.assert_allclose(params[0]["weight"], state["0.weight"].T)
    np.testing.assert_allclose(params[1]["bias"], state["2.bias"])


def test_jax_params_to_torch_state_round_trip() -> None:
    params = (
        {"weight": np.ones((3, 4), dtype=np.float32), "bias": np.zeros(4, dtype=np.float32)},
        {"weight": np.ones((4, 1), dtype=np.float32), "bias": np.zeros(1, dtype=np.float32)},
    )

    state = jax_params_to_torch_state(params)
    restored = torch_state_to_jax_params(state)

    for left, right in zip(params, restored):
        np.testing.assert_allclose(left["weight"], right["weight"])
        np.testing.assert_allclose(left["bias"], right["bias"])


def test_numpy_mlp_forward_and_parity_report() -> None:
    params = (
        {
            "weight": np.array([[1.0, -1.0]], dtype=np.float32),
            "bias": np.array([0.0, 0.0], dtype=np.float32),
        },
        {
            "weight": np.array([[1.0], [1.0]], dtype=np.float32),
            "bias": np.array([0.0], dtype=np.float32),
        },
    )
    features = np.array([[2.0]], dtype=np.float32)

    predictions = numpy_mlp_forward(params, features, activation="relu")
    report = parity_report(predictions, np.array([2.0], dtype=np.float32))

    assert predictions.tolist() == [2.0]
    assert report["within_tolerance"]


def test_parity_report_rejects_empty_non_finite_or_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        parity_report(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        parity_report(np.array([np.nan]), np.array([0.0]))
    with pytest.raises(ValueError, match="tolerance"):
        parity_report(np.array([0.0]), np.array([0.0]), tolerance=-1.0)


def test_parameter_tree_parity_reports_direction_and_scale() -> None:
    left = (
        {"weight": np.array([[1.0], [2.0]]), "bias": np.array([0.5])},
    )
    right = (
        {"weight": np.array([[1.0], [2.01]]), "bias": np.array([0.5])},
    )

    report = parameter_tree_parity_report(left, right, tolerance=0.02)

    assert report["within_tolerance"]
    assert report["parameters_compared"] == 3
    assert report["cosine_similarity"] > 0.999
    assert report["relative_l2_error"] > 0.0


def test_input_gradient_parity_matches_torch_and_jax() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 1),
    )
    params = torch_state_to_jax_params(model.state_dict())
    features = np.array([[0.2, -0.4], [0.7, 0.1]], dtype=np.float32)
    targets = np.array([0.3, -0.2], dtype=np.float32)

    report = _input_gradient_parity(
        model,
        params,
        features,
        targets,
        activation="tanh",
        loss_name="mse",
        huber_delta=1.0,
        tolerance=1e-5,
    )

    assert report["within_tolerance"]
    assert report["cosine_similarity"] > 0.99999
    assert report["relative_l2_error"] < 1e-5


def test_hessian_vector_product_matches_torch_and_jax() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 1),
    )
    params = torch_state_to_jax_params(model.state_dict())
    report = _hessian_vector_parity(
        model,
        params,
        np.array([[0.2, -0.4], [0.7, 0.1]], dtype=np.float32),
        np.array([0.3, -0.2], dtype=np.float32),
        activation="tanh",
        loss_name="mse",
        huber_delta=1.0,
        tolerance=1e-5,
    )

    assert report["within_tolerance"]
    assert report["relative_l2_error"] < 1e-5


def test_activation_health_detects_relu_sparsity_and_tanh_saturation() -> None:
    params = (
        {"weight": np.array([[1.0, -1.0]]), "bias": np.array([0.0, 0.0])},
        {"weight": np.ones((2, 1)), "bias": np.zeros(1)},
    )
    features = np.array([[2.0], [-2.0]], dtype=np.float32)

    relu = activation_health_report(params, features, activation="relu")[0]
    tanh = activation_health_report(params, features * 5.0, activation="tanh")[0]

    assert relu["near_zero_fraction"] == pytest.approx(0.5)
    assert relu["saturated_fraction"] == 0.0
    assert tanh["saturated_fraction"] == 1.0


def test_directional_derivative_matches_quadratic_gradient() -> None:
    point = np.array([1.5, -2.0])
    report = directional_derivative_report(
        lambda value: float(np.sum(value**2)),
        point,
        2.0 * point,
        direction=np.array([0.3, -0.7]),
        epsilon=1e-5,
    )

    assert report["relative_error"] < 1e-8
