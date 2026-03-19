# Bayesian Optimization for Protein Refolding

An implementation of multi-objective Bayesian optimization for protein refolding optimization using fluorescence-based stability metrics.

## Overview

This codebase implements a Bayesian optimization framework for optimizing scFv antibody refolding conditions. It uses qNEHVI (Noisy Expected Hypervolume Improvement) acquisition function with Gaussian Process models to simultaneously maximize protein yield and concentration while respecting physical constraints of the urea dilution process.

## Features

- **Multi-objective optimization** using qNEHVI
- **Single-task Gaussian Process models** with Matérn kernels
- **Physical constraint handling** for urea dilution chemistry
- **Latin Hypercube Sampling** with maximin criterion for initial experimental design
- **Cross-validation** and model uncertainty quantification
- ## Installation

Requires **Anaconda** or **Miniconda**.

### Option A: Double-click setup (Easiest)

Just double-click the setup file in this folder:

| Windows | Mac/Linux |
|---------|-----------|
| `setup.bat` | `setup.sh` |

The script will create the conda environment, install all packages, and register the Jupyter kernel.

### Option B: Use the terminal

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate it
conda activate bayesopt-fluorescence

# 3. (Optional) Register Jupyter kernel
python -m ipykernel install --user --name bayesopt-fluorescence --display-name "Python (BayesOpt)"
```

### GPU Support (Optional)

If you have a CUDA-capable GPU, edit `environment.yml` and remove the `cpuonly` line before creating the environment.

## How to Use

You have two ways to run the optimization:

### Option 1: Jupyter Notebook (Recommended for beginners)

```bash
conda activate bayesopt-fluorescence
jupyter notebook
```

Then open `workshop_notebook.ipynb` for a step-by-step interactive guide.

### Option 2: Command Line

See [CLI_GUIDE.md](CLI_GUIDE.md) for the command-line workflow.

## Usage

The workflow consists of three main steps:

### 1. Generate Initial Experimental Design

```bash
python generate_initial_design.py \
    --n_samples 20 \
    --output_dir results \
    --project_name initial_design \
    --seed 42 \
    --n_candidates 100
```

This creates an initial set of 20 experiments using Latin Hypercube Sampling with maximin criterion optimization (evaluates 100 candidate designs) and physical constraints applied. Use `--no_maximin` to disable maximin optimization for faster generation.

### 2. Train Gaussian Process Models

After conducting the initial experiments and measuring outcomes:

```bash
python train_models.py \
    --data_file results/initial_design_experimental_results.xlsx \
    --model_dir models \
    --project_name iteration_1_models
```

This trains separate GP models for each objective (Delta AEW and p_proxy) with cross-validation.

### 3. Run Bayesian Optimization

```bash
python run_optimization.py \
    --data_file results/initial_design_experimental_results.xlsx \
    --model_dir models/iteration_1_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 2
```

This generates 4 new experimental conditions optimized for both objectives.

## Configuration

All parameters are centralized in `config.py`:

- **Experimental bounds** for parameters (DTT, GSSG, dilution factor, pH, final urea)
- **Optimization hyperparameters** (batch size, MC samples, reference point)
- **Model training parameters** (learning rate, iterations, kernel settings)
- **Constraint parameters** for urea dilution physics

## Code Structure

```
bayesopt-fluorescence/
├── config.py                 # Centralized configuration
├── generate_initial_design.py # Initial experimental design
├── train_models.py          # GP model training
├── run_optimization.py      # Main BO loop
├── README.md                # This file
├── models/                  # GP modeling utilities
│   ├── __init__.py
│   ├── gp_model.py         # Single-task GP model definition
│   ├── gp_fitting.py       # Training and loading functions
│   └── gp_validation.py    # Cross-validation utilities
├── acquisition/            # Acquisition functions
│   ├── __init__.py
│   ├── qnehvi.py          # qNEHVI implementation
│   └── utils.py           # Experimental planning utilities
├── constraints/            # Physical constraints
│   ├── __init__.py
│   └── urea_dilution.py   # Urea dilution constraints
└── data/                  # Data preprocessing
    ├── __init__.py
    └── preprocessing.py   # Normalization utilities
```

## Parameters

### Experimental Parameters

- **DTT**: 0-25 mM (solubilization buffer)
- **GSSG**: 0-2.5 mM
- **Dilution Factor**: 2-40
- **pH**: 8-11
- **Final Urea**: 0-6 M

### Optimization Objectives

1. **Delta AEW** (fluorescence-based yield metric)
2. **p_proxy** (protein concentration proxy)

### Physical Constraints

The urea dilution constraint ensures physically feasible refolding conditions:

```
urea_refolding = (final_urea * dilution_factor - 8) / (dilution_factor - 1) > 0
```

## Methodology

### Gaussian Process Models

- Single-task GPs with Matérn kernels (ν=2.5)
- Automatic Relevance Determination (ARD) for feature selection
- Standardized objectives with proper uncertainty quantification

### Acquisition Function

- qNEHVI for noisy multi-objective optimization
- Reference point at [0.0, 0.0] in standardized space
- Sequential optimization for batch generation

### Constraint Handling

- Iterative adjustment strategy for urea dilution constraints
- Automatic parameter correction to ensure feasibility

## Citation

If you use this code in your research, please cite:

```
[Add your citation information here]
```
