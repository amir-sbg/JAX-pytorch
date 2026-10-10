from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_backend.config import ExperimentConfig, ModelConfig, TrainingConfig, load_experiment_config
from dual_backend.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="dual-backend-regression")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="Train PyTorch, JAX, or both backends")
    train.add_argument("--backend", choices=["torch", "jax", "both"], default="both")
    train.add_argument("--dataset", choices=["diabetes", "csv"], default="diabetes")
    train.add_argument("--csv-path", type=Path)
    train.add_argument("--target-column")
    train.add_argument("--validation-size", type=float, default=0.20)
    train.add_argument("--test-size", type=float, default=0.20)
    train.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32])
    train.add_argument("--activation", choices=["relu", "gelu", "tanh"], default="gelu")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-3)
    train.add_argument("--learning-rate-schedule", choices=["constant", "cosine"], default="constant")
    train.add_argument("--warmup-epochs", type=int, default=0)
    train.add_argument("--min-learning-rate-ratio", type=float, default=0.05)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--loss", choices=["mse", "huber"], default="mse")
    train.add_argument("--huber-delta", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--report-dir", type=Path, default=Path("reports"))

    parity = subcommands.add_parser("parity", help="Check PyTorch checkpoint conversion into JAX layout")
    parity.add_argument("--torch-checkpoint", type=Path, default=Path("artifacts/torch_model.pt"))
    parity.add_argument("--dataset", choices=["diabetes", "csv"], default="diabetes")
    parity.add_argument("--csv-path", type=Path)
    parity.add_argument("--target-column")
    parity.add_argument("--validation-size", type=float, default=0.20)
    parity.add_argument("--test-size", type=float, default=0.20)
    parity.add_argument("--seed", type=int, default=42)
    parity.add_argument("--tolerance", type=float, default=1e-5)
    parity.add_argument("--output", type=Path, default=Path("reports/parity.json"))
    parity.add_argument("--device", default="auto")

    run_config = subcommands.add_parser("run-config", help="Run an experiment from a JSON config")
    run_config.add_argument("config_path", type=Path)

    args = parser.parse_args()
    if args.command == "train":
        summary = run_experiment(_config_from_train_args(args))
        print(json.dumps(_compact_summary(summary), indent=2))
    elif args.command == "run-config":
        summary = run_experiment(load_experiment_config(args.config_path))
        print(json.dumps(_compact_summary(summary), indent=2))
    elif args.command == "parity":
        from dual_backend.parity import check_torch_to_jax_layout

        report = check_torch_to_jax_layout(
            args.torch_checkpoint,
            _config_from_parity_args(args),
            args.output,
            tolerance=args.tolerance,
            device=args.device,
        )
        print(json.dumps(report, indent=2))


def _config_from_train_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        backend=args.backend,
        dataset=args.dataset,
        csv_path=args.csv_path,
        target_column=args.target_column,
        validation_size=args.validation_size,
        test_size=args.test_size,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        model=ModelConfig(
            hidden_dims=tuple(args.hidden_dims),
            activation=args.activation,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            learning_rate_schedule=args.learning_rate_schedule,
            warmup_epochs=args.warmup_epochs,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
            weight_decay=args.weight_decay,
            loss=args.loss,
            huber_delta=args.huber_delta,
            patience=args.patience,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
        ),
    )


def _config_from_parity_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        backend="torch",
        dataset=args.dataset,
        csv_path=args.csv_path,
        target_column=args.target_column,
        validation_size=args.validation_size,
        test_size=args.test_size,
        training=TrainingConfig(seed=args.seed),
    )


def _compact_summary(summary: dict) -> dict:
    return {
        "dataset": summary["dataset"],
        "metrics": {
            name: report["metrics"]
            for name, report in summary["backends"].items()
        },
        "selection": summary.get("selection", {}),
        "comparison": summary["comparison"],
        "report": summary["report_path"],
    }


if __name__ == "__main__":
    main()
