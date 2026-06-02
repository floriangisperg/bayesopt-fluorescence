# Bayesian Optimization for Protein Refolding

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD6DD00035E-blue)](https://doi.org/10.1039/D6DD00035E)

Multi-objective Bayesian optimization workflow for fluorescence-guided protein refolding experiments. Uses Gaussian process surrogate models and qNEHVI batch acquisition to efficiently explore high-dimensional refolding conditions.

## Associated Publication

This repository contains the improved, open-source implementation of the workflow described in:

> **Spectroscopy-Assisted Bayesian Optimization for Efficient Refolding of Inclusion Body Proteins**
> F. Gisperg, R. Klausser, M. Kierein, M. Elshazly, J. Kopp, E. Prada Brichtova and O. Spadiut
> *Digital Discovery*, 2026
> [DOI: 10.1039/D6DD00035E](https://doi.org/10.1039/D6DD00035E) (Open Access)

The paper demonstrates that this workflow achieves ~3.5x higher product concentration at comparable yield, while requiring fewer than half the experiments of a conventional DoE approach.

## How It Works

The optimization loop cycles through four steps:

1. **Design** — generate an initial experimental plan using constraint-aware Latin hypercube sampling
2. **Experiment** — run the refolding experiments and record spectroscopy-derived objectives
3. **Train** — fit independent single-task Gaussian process models to each objective
4. **Suggest** — use qNEHVI acquisition to propose the next batch of experiments

Each iteration refines the surrogate models and steers sampling toward the Pareto front.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/). Install it with:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then set up the project:

### Option A: Double-click setup

| Windows            | Mac/Linux           |
| ------------------ | ------------------- |
| `workshop/setup.bat` | `workshop/setup.sh` |

The script installs uv (if needed), creates the virtual environment, and registers the Jupyter kernel.

### Option B: Use the terminal

```bash
uv sync
```

### GPU support

By default, PyTorch installs CPU-only wheels. To use CUDA, edit the `[tool.uv.sources]` section in `pyproject.toml` — instructions are in the comments there.

## Documentation

| Document | Description |
| --- | --- |
| [workshop/README.md](workshop/README.md) | Notebook-first quickstart for workshop use |
| [workshop/CLI_GUIDE.md](workshop/CLI_GUIDE.md) | Command-focused walkthrough for terminal use |

## Workflows

### Notebook workflow

```bash
uv run jupyter notebook
```

Open `workshop/workshop_notebook.ipynb` and follow the cells in order.

### CLI workflow

Use the three main scripts:

- `uv run python generate_initial_design.py`
- `uv run python train_models.py`
- `uv run python run_optimization.py`

See [workshop/CLI_GUIDE.md](workshop/CLI_GUIDE.md) for the full command sequence.

### Demo with synthetic data

```bash
uv run python workshop/demo_workflow.py --n_iterations 5 --n_initial 12 --n_candidates 4
```

## CLI Reference

### Generate the initial design

```bash
uv run python generate_initial_design.py \
    --n_samples 20 \
    --output_dir results \
    --project_name iteration_0 \
    --seed 42 \
    --n_candidates 100
```

Output: `results/iteration_0_experimental_plan.xlsx`

### Train the GP models

After filling in `Delta AEW` and `p_proxy` in the spreadsheet:

```bash
uv run python train_models.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models \
    --project_name iteration_0_models
```

Outputs: trained `.pth` model files and `.pkl` scaler files in `models/iteration_0_models/`.

### Generate the next candidates

```bash
uv run python run_optimization.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models/iteration_0_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 1
```

Outputs: `results/Iteration_1/Iteration_1_experimental_plan.xlsx` and a running `results/experimental_database.xlsx`.

## Configuration

All project settings live in `config.py`.

| Config class | Controls |
| --- | --- |
| `ExperimentConfig` | Parameter bounds, parameter names, objective names |
| `ConstraintConfig` | Urea constraint toggle and parameters |
| `ModelConfig` | GP training and validation settings |
| `OptimizationConfig` | qNEHVI and acquisition optimization settings |

Default experimental parameters: **DTT** (0–25 mM), **GSSG** (0–2.5 mM), **Dilution Factor** (2–40), **pH** (8–11), **Final Urea** (0–6 M).

## Constraint Handling

The physical urea constraint is controlled by `ConstraintConfig.ENABLE_UREA_CONSTRAINT`.

- Initial designs use a constrained Latin hypercube strategy specialized for the urea constraint.
- Bayesian optimization passes the urea condition as a nonlinear inequality constraint to the acquisition optimizer.
- A post-processing repair step exists as a fallback for numerical edge cases.

Feasibility condition: `final_urea * dilution_factor > solubilization_urea` (default: `> 8.0`).

## Repository Structure

```text
bayesopt-fluorescence/
├── config.py                  # Centralized configuration
├── generate_initial_design.py # LHS initial design
├── train_models.py            # GP model training
├── run_optimization.py        # qNEHVI candidate generation
├── pyproject.toml             # Dependencies and project metadata
├── acquisition/               # qNEHVI acquisition function
├── constraints/               # Urea-dilution physical constraint
├── data/                      # Preprocessing and scaling
├── models/                    # GP model, fitting, and validation
└── workshop/                  # Notebook, demo, and workshop materials
    ├── README.md
    ├── CLI_GUIDE.md
    ├── setup.bat / setup.sh
    ├── workshop_notebook.ipynb
    └── demo_workflow.py
```

## Troubleshooting

**Missing imports** — run `uv sync` to install dependencies.

**Missing required columns** — training expects all parameter columns and both objective columns in the Excel file.

**Model directory errors** — make sure `--model_dir` points to the exact subdirectory created by `train_models.py`.
