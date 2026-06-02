# Workshop Quickstart

This folder contains the interactive workshop materials for the Bayesian optimization workflow.

For the full project documentation, setup details, and configuration, see the [main README](../README.md).

## Contents

- **`workshop_notebook.ipynb`** — guided step-by-step notebook for designing experiments with BO
- **`demo_workflow.py`** — end-to-end demo that runs the full pipeline with synthetic data

## Installation

Requires [uv](https://docs.astral.sh/uv/).

### Option A: Double-click setup

| Windows     | Mac/Linux  |
| ----------- | ---------- |
| `setup.bat` | `setup.sh` |

### Option B: Terminal

```bash
uv sync
```

## Start the Notebook

```bash
uv run jupyter notebook
```

Navigate into `workshop/` and open `workshop_notebook.ipynb`.

## Workshop Loop

1. Define the campaign settings in the notebook.
2. Generate the initial experimental plan.
3. Run the experiments and fill in the Excel file.
4. Load the measured results.
5. Train the models.
6. Generate the next AI-suggested batch.
7. Repeat.

## Quick Demo

To exercise the full pipeline with synthetic data:

```bash
uv run python demo_workflow.py --n_iterations 5 --n_initial 12 --n_candidates 4
```

## If You Prefer the Terminal

Use the [CLI Guide](CLI_GUIDE.md) for the script-based workflow.
