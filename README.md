# Bayesian Optimization for Protein Refolding

Bayesian optimization workflow for fluorescence-guided protein refolding experiments.

This repository optimizes refolding conditions for two objectives, `Delta AEW` and `p_proxy`, using Gaussian process surrogate models and qNEHVI batch acquisition. The same core workflow is available through either a Jupyter notebook or command-line scripts.

## Documentation Map

Use this file as the main reference.

- [WORKSHOP_README.md](WORKSHOP_README.md): notebook-first quickstart for workshop use
- [CLI_GUIDE.md](CLI_GUIDE.md): command-focused walkthrough for terminal use

## Features

- Multi-objective Bayesian optimization with qNEHVI
- Separate single-task GP models for each objective
- Constraint-aware initial design generation
- Nonlinear urea constraint support during acquisition optimization
- Excel-based handoff between experiment planning and wet-lab execution
- Cross-validation and validation plots for trained models

## Installation

Requires **Anaconda** or **Miniconda**.

### Option A: Double-click setup

Just double-click the setup file in this folder:

| Windows     | Mac/Linux  |
| ----------- | ---------- |
| `setup.bat` | `setup.sh` |

The setup script creates the conda environment, installs dependencies, and registers the Jupyter kernel.

**Note:** The script activates the environment only temporarily during installation. After it finishes, open a new terminal and run:

```bash
conda activate bayesopt-fluorescence
```

Then you're ready to start Jupyter.

### Option B: Use the terminal

```bash
conda env create -f environment.yml
conda activate bayesopt-fluorescence

# Optional: register the kernel for Jupyter
python -m ipykernel install --user --name bayesopt-fluorescence --display-name "Python (BayesOpt)"
```

### GPU support

If you want CUDA-enabled PyTorch, remove `cpuonly` from `environment.yml` before creating the environment.

## Choose a Workflow

### Notebook workflow

```bash
conda activate bayesopt-fluorescence
jupyter notebook
```

Then open `workshop_notebook.ipynb`.

This is the easiest path for workshops and first-time users.

### CLI workflow

Use the three scripts directly:

- `python generate_initial_design.py`
- `python train_models.py`
- `python run_optimization.py`

See [CLI_GUIDE.md](CLI_GUIDE.md) for the full command sequence.

## Core Workflow

Regardless of whether you use the notebook or CLI, the optimization loop is the same:

1. Generate an initial experimental design.
2. Run the experiments and fill in the objective columns in the Excel file.
3. Train GP models on the completed data.
4. Generate the next batch of candidate experiments.
5. Append completed results to your running dataset and repeat.

## CLI Commands

### 1. Generate the initial design

```bash
python generate_initial_design.py \
    --n_samples 20 \
    --output_dir results \
    --project_name iteration_0 \
    --seed 42 \
    --n_candidates 100
```

Output:

- `results/iteration_0_experimental_plan.xlsx`

### 2. Train the GP models

After you have filled in `Delta AEW` and `p_proxy` in the spreadsheet:

```bash
python train_models.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models \
    --project_name iteration_0_models
```

Typical outputs:

- `models/iteration_0_models/model_1_delta_aew.pth`
- `models/iteration_0_models/model_2_p_proxy.pth`
- `models/iteration_0_models/scaler_1_delta_aew.pkl`
- `models/iteration_0_models/scaler_2_p_proxy.pkl`

### 3. Generate the next candidates

```bash
python run_optimization.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models/iteration_0_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 1
```

Outputs:

- `results/Iteration_1/Iteration_1_experimental_plan.xlsx`
- `results/experimental_database.xlsx`

`experimental_database.xlsx` is a running log of generated candidate batches. It is not a replacement for your fully completed training dataset, because new candidate plans are created with empty objective columns until experiments are performed.

## Configuration

Most project settings live in `config.py`.

- `ExperimentConfig`: parameter bounds, parameter names, objective names
- `ConstraintConfig`: urea constraint toggle and parameters
- `ModelConfig`: GP training and validation settings
- `OptimizationConfig`: qNEHVI and acquisition optimization settings

The default experimental parameters are:

- `DTT [mM]`: 0 to 25
- `GSSG [mM]`: 0 to 2.5
- `Dilution Factor`: 2 to 40
- `pH`: 8 to 11
- `Final Urea [M]`: 0 to 6

## Constraint Handling

The physical urea constraint is controlled by `ConstraintConfig.ENABLE_UREA_CONSTRAINT`.

Current implementation:

- Initial designs use a constrained Latin hypercube strategy specialized for the urea constraint.
- Bayesian optimization passes the urea condition as a nonlinear inequality constraint to the acquisition optimizer.
- A post-processing repair step still exists as a fallback for numerical edge cases.

The feasibility condition is:

```text
final_urea * dilution_factor > solubilization_urea
```

With the default settings in `config.py`, this becomes:

```text
final_urea * dilution_factor > 8.0
```

The corresponding refolding-buffer urea concentration is:

```text
urea_refolding = (final_urea * dilution_factor - solubilization_urea) / (dilution_factor - 1)
```

Feasible points have `urea_refolding > 0`.

## Repository Structure

```text
bayesopt-fluorescence/
|-- config.py
|-- generate_initial_design.py
|-- train_models.py
|-- run_optimization.py
|-- workshop_notebook.ipynb
|-- README.md
|-- WORKSHOP_README.md
|-- CLI_GUIDE.md
|-- acquisition/
|-- constraints/
|-- data/
`-- models/
```

## Troubleshooting

### Missing imports or module errors

Activate the environment first:

```bash
conda activate bayesopt-fluorescence
```

### Missing required columns

Training expects all parameter columns and both objective columns to be present in the Excel file.

### Model directory errors

Make sure `--model_dir` points to the exact subdirectory created by `train_models.py`.
