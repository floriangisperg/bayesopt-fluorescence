# CLI Guide: Bayesian Optimization from the Command Line

This guide shows how to run the Bayesian optimization workflow using command-line scripts instead of the Jupyter notebook. Use this if you prefer working in a terminal or want to automate your workflow.

## Prerequisites

Make sure you've installed the environment first:

```bash
conda activate bayesopt-fluorescence
```

---

## Quick Reference

| Step | Command |
|------|---------|
| 1. Generate initial design | `python generate_initial_design.py` |
| 2. Fill in results | *(manual - edit Excel file)* |
| 3. Train models | `python train_models.py` |
| 4. Get suggestions | `python run_optimization.py` |
| 5. Repeat from step 2 | |

---

## Step 1: Generate Initial Experimental Design

Create your first set of experiments using Latin Hypercube Sampling:

```bash
python generate_initial_design.py \
    --n_samples 12 \
    --output_dir results \
    --project_name iteration_0 \
    --seed 42
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_samples` | 20 | Number of initial experiments |
| `--output_dir` | `results` | Output directory |
| `--project_name` | `initial_design` | Name for the output file |
| `--seed` | 42 | Random seed for reproducibility |
| `--n_candidates` | 100 | Candidates for maximin optimization |
| `--no_maximin` | False | Disable maximin criterion |

### Output

- `results/iteration_0_experimental_plan.xlsx` - Your initial experiments

---

## Step 2: Perform Experiments & Enter Results

1. Open `results/iteration_0_experimental_plan.xlsx`
2. Perform your experiments
3. Fill in the objective columns (`Delta AEW` and `p_proxy`)
4. Save the file

**Important:** The file must contain both parameter columns AND objective columns with values filled in.

---

## Step 3: Train GP Models

Train Gaussian Process models on your experimental data:

```bash
python train_models.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models \
    --project_name iteration_0_models
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_file` | *(required)* | Excel file with experimental data |
| `--model_dir` | `models` | Directory to save models |
| `--project_name` | `gpytorch_models` | Subdirectory name for this training run |

### Output

- `models/iteration_0_models/model_1_delta_aew.pth` - Trained model for objective 1
- `models/iteration_0_models/model_2_p_proxy.pth` - Trained model for objective 2
- `models/iteration_0_models/scaler_1_*.pkl` - Scaler for objective 1
- `models/iteration_0_models/scaler_2_*.pkl` - Scaler for objective 2
- Validation plots (if cross-validation enabled)

---

## Step 4: Get BO Suggestions

Generate new experimental suggestions using Bayesian optimization:

```bash
python run_optimization.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models/iteration_0_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 1
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_file` | *(required)* | Excel file with all experimental data so far |
| `--model_dir` | *(required)* | Directory with trained models |
| `--output_dir` | `results` | Output directory |
| `--n_candidates` | 4 | Number of new candidates to generate |
| `--iteration` | *(required)* | Current iteration number |
| `--smoke_test` | False | Run with reduced computation (for testing) |

### Output

- `results/Iteration_1/Iteration_1_experimental_plan.xlsx` - New experiments
- `results/experimental_database.xlsx` - Combined database (updated)

---

## Step 5: Continue the Loop

### 5a. Perform new experiments

1. Open `results/Iteration_1/Iteration_1_experimental_plan.xlsx`
2. Perform experiments and fill in results
3. Save

### 5b. Combine data

Create a combined file with all experiments so far:

```bash
# On Windows (PowerShell)
python -c "import pandas as pd; df1 = pd.read_excel('results/iteration_0_experimental_plan.xlsx'); df2 = pd.read_excel('results/Iteration_1/Iteration_1_experimental_plan.xlsx'); pd.concat([df1, df2]).to_excel('results/combined_iteration_1.xlsx', index=False)"

# On Mac/Linux
python -c "import pandas as pd; df1 = pd.read_excel('results/iteration_0_experimental_plan.xlsx'); df2 = pd.read_excel('results/Iteration_1/Iteration_1_experimental_plan.xlsx'); pd.concat([df1, df2]).to_excel('results/combined_iteration_1.xlsx', index=False)"
```

### 5c. Retrain models

```bash
python train_models.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models \
    --project_name iteration_1_models
```

### 5d. Get new suggestions

```bash
python run_optimization.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models/iteration_1_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 2
```

---

## Full Workflow Example

Here's a complete example from start to finish:

```bash
# Activate environment
conda activate bayesopt-fluorescence

# === ITERATION 0: Initial Design ===
python generate_initial_design.py \
    --n_samples 12 \
    --output_dir results \
    --project_name iteration_0 \
    --seed 42

# -> Now do your experiments and fill in results in results/iteration_0_experimental_plan.xlsx

# === ITERATION 1: First BO Round ===
# Train models
python train_models.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models \
    --project_name iteration_0_models

# Get suggestions
python run_optimization.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models/iteration_0_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 1

# -> Do experiments for Iteration 1, fill in results

# === ITERATION 2: Second BO Round ===
# Combine data
python -c "import pandas as pd; df1 = pd.read_excel('results/iteration_0_experimental_plan.xlsx'); df2 = pd.read_excel('results/Iteration_1/Iteration_1_experimental_plan.xlsx'); pd.concat([df1, df2]).to_excel('results/combined_iteration_1.xlsx', index=False)"

# Retrain
python train_models.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models \
    --project_name iteration_1_models

# Get suggestions
python run_optimization.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models/iteration_1_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 2

# -> Continue the loop...
```

---

## Configuration

All parameters (bounds, objectives, constraints) are defined in `config.py`. Edit this file to customize:

- **Parameter bounds** (`ExperimentConfig.PARAMETER_BOUNDS`)
- **Objective names** (`ExperimentConfig.OBJECTIVE_NAMES`)
- **Constraint settings** (`ConstraintConfig`)
- **Model hyperparameters** (`ModelConfig`)
- **Optimization settings** (`OptimizationConfig`)

---

## Demo Mode (Testing)

To test the workflow without real experiments, use the demo script:

```bash
python demo_workflow.py --n_iterations 5 --n_initial 12 --n_candidates 4
```

This runs the complete workflow with synthetic data.

---

## Troubleshooting

### "Module not found" errors
Make sure you activated the conda environment:
```bash
conda activate bayesopt-fluorescence
```

### "Missing required columns"
Ensure your Excel file has all parameter and objective columns filled in.

### "Model file not found"
Check that the `--model_dir` path matches where you saved models in the training step.

---

## Notebook vs CLI Comparison

| Feature | Notebook | CLI |
|---------|----------|-----|
| Ease of use | Visual, step-by-step | Scriptable, automatable |
| Visualization | Inline plots | Saved to files |
| Best for | Learning, exploration | Production, batch jobs |
| Demo mode | Built-in (`DEMO_MODE=True`) | Use `demo_workflow.py` |
