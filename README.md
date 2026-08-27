# JAX + PyTorch Dual-Backend Regression Pipeline

This repository is a compact, usable ML engineering project for training the same tabular regression model in both PyTorch and JAX. The goal is not just to compare libraries, but to keep the data pipeline, architecture, metrics, reports, and checkpoints aligned so backend differences are easy to inspect.

The default experiment uses the scikit-learn diabetes regression dataset because it is small and reproducible, but the same CLI also accepts a numeric CSV dataset with a target column.

## What is implemented

- Shared preprocessing: train/validation/test split, feature scaling, target scaling, batch construction
- PyTorch backend: `nn.Sequential` MLP, AdamW, gradient clipping, early stopping, checkpointing
- JAX backend: explicit parameter PyTrees, JIT-compiled training step, hand-written Adam update, checkpointing to `.npz`
- Backend comparison: consistent test metrics, residual summaries, prediction CSVs, training-history CSVs
- Cross-backend agreement diagnostics for prediction gaps, correlations, and residual sign disagreements
- PyTorch → JAX layout check: converts a PyTorch checkpoint into the JAX parameter layout and verifies numerical parity
- CLI workflow for running experiments and writing reproducible reports

## Project layout

```text
src/dual_backend/
  config.py          experiment, model, and training config dataclasses
  data.py            dataset loading, splitting, scaling, batching
  metrics.py         regression metrics and residual summaries
  torch_backend.py   PyTorch model/training/checkpoint path
  jax_backend.py     JAX model/training/checkpoint path
  interop.py         parameter conversion and parity utilities
  parity.py          PyTorch checkpoint → JAX-layout verification
  pipeline.py        end-to-end experiment runner
  cli.py             command-line interface
tests/
  test_data_metrics.py
  test_interop.py
  test_config.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For Apple Silicon or CPU-only machines, the standard `jax`/`jaxlib` wheels are enough. For CUDA JAX, install the JAX wheel that matches the local CUDA stack before running the project.

## Run the default experiment

```bash
python -m dual_backend.cli train \
  --backend both \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.003
```

Outputs are written to:

```text
artifacts/
  torch_model.pt
  jax_params.npz

reports/
  run_summary.json
  predictions.csv
  torch_history.csv
  jax_history.csv
```

`run_summary.json` contains metrics for each backend plus the JAX-minus-PyTorch metric deltas and prediction-agreement diagnostics in the original target scale.

## Run only one backend

```bash
python -m dual_backend.cli train --backend torch
python -m dual_backend.cli train --backend jax
```

This is useful when checking backend-specific performance or debugging a training issue.

## Run from a saved config

For repeatable experiments, the same settings can be stored in JSON:

```bash
python -m dual_backend.cli run-config examples/diabetes_both.json
```

This is the easiest way to keep model width, optimizer settings, split sizes, and output paths fixed across PyTorch/JAX comparisons.

## Use a custom CSV

The CSV path should contain numeric feature columns and one numeric target column.

```bash
python -m dual_backend.cli train \
  --backend both \
  --dataset csv \
  --csv-path data/my_regression_data.csv \
  --target-column target
```

Rows with missing numeric values are dropped before splitting. Feature and target scaling are fitted only on the training split.

## Check PyTorch → JAX parameter layout

After training the PyTorch backend:

```bash
python -m dual_backend.cli parity \
  --torch-checkpoint artifacts/torch_model.pt \
  --output reports/parity.json
```

The parity command loads the PyTorch model, converts each linear layer into the JAX weight layout, runs the converted NumPy/JAX-style forward pass, and reports max/mean absolute prediction differences. This catches the common transpose bug when moving between PyTorch `(out, in)` linear weights and JAX `(in, out)` matrix multiplication.

## Tests

```bash
python -m pytest
```

The lightweight tests cover preprocessing, metrics, config validation, and parameter-layout conversion. Full backend training requires PyTorch and JAX installed.

## Why this project is useful

This project shows the same ML workflow implemented across both ecosystems:

- PyTorch for conventional model engineering and checkpointing
- JAX for explicit functional parameters, JIT-compiled updates, and transparent optimizer math
- A shared reporting layer so results are comparable instead of being two disconnected scripts

The structure is intentionally small, but it is a real pipeline: it trains, evaluates, saves artifacts, writes reports, and checks cross-framework numerical consistency.
