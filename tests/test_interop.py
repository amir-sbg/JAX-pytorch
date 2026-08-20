import numpy as np

from dual_backend.interop import (
    jax_params_to_torch_state,
    numpy_mlp_forward,
    parity_report,
    torch_state_to_jax_params,
)


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
