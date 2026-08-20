from __future__ import annotations

from pathlib import Path

from dual_backend.config import ExperimentConfig
from dual_backend.data import load_tabular_dataset
from dual_backend.interop import numpy_mlp_forward, parity_report, torch_state_to_jax_params
from dual_backend.metrics import save_json
from dual_backend.torch_backend import load_torch_model, predict_torch


def check_torch_to_jax_layout(
    checkpoint_path: Path,
    config: ExperimentConfig,
    output_path: Path,
    tolerance: float = 1e-5,
    device: str = "auto",
) -> dict:
    config.validate()
    dataset = load_tabular_dataset(
        dataset=config.dataset,
        csv_path=config.csv_path,
        target_column=config.target_column,
        validation_size=config.validation_size,
        test_size=config.test_size,
        seed=config.training.seed,
    )
    model, resolved_device, model_config = load_torch_model(checkpoint_path, device=device)
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
    }
    save_json(report, output_path)
    return report
