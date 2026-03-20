# CLI Guide

This file covers the terminal workflow only.

For installation, project overview, configuration, and the canonical explanation of the urea constraint, see [README.md](README.md).

## Prerequisite

```bash
conda activate bayesopt-fluorescence
```

## Quick Reference

| Step                       | Command                             |
| -------------------------- | ----------------------------------- |
| 1. Generate initial design | `python generate_initial_design.py` |
| 2. Fill in results         | manual Excel editing                |
| 3. Train models            | `python train_models.py`            |
| 4. Get suggestions         | `python run_optimization.py`        |
| 5. Repeat                  | continue from step 2                |

## Step 1: Generate the Initial Design

```bash
python generate_initial_design.py \
    --n_samples 12 \
    --output_dir results \
    --project_name iteration_0 \
    --seed 42
```

Arguments:

| Argument         | Default          | Description                   |
| ---------------- | ---------------- | ----------------------------- |
| `--n_samples`    | 20               | Number of initial experiments |
| `--output_dir`   | `results`        | Output directory              |
| `--project_name` | `initial_design` | Base name for the output file |
| `--seed`         | 42               | Random seed                   |
| `--n_candidates` | 100              | Candidate designs for maximin |
| `--no_maximin`   | False            | Disable maximin selection     |

Output:

- `results/iteration_0_experimental_plan.xlsx`

Notes:

- If the urea constraint is enabled, this step uses a constraint-aware Latin hypercube design.
- The generated spreadsheet contains parameter columns and empty objective columns to be filled after experiments.

## Step 2: Run Experiments and Enter Results

1. Open `results/iteration_0_experimental_plan.xlsx`.
2. Run the experiments.
3. Fill in `Delta AEW` and `p_proxy`.
4. Save the file.

Training requires both objective columns to contain values.

## Step 3: Train GP Models

```bash
python train_models.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models \
    --project_name iteration_0_models
```

Arguments:

| Argument         | Default           | Description                             |
| ---------------- | ----------------- | --------------------------------------- |
| `--data_file`    | required          | Excel file with completed measurements  |
| `--model_dir`    | `models`          | Directory to save trained models        |
| `--project_name` | `gpytorch_models` | Subdirectory name for this training run |

Typical outputs:

- `models/iteration_0_models/model_1_delta_aew.pth`
- `models/iteration_0_models/model_2_p_proxy.pth`
- `models/iteration_0_models/scaler_1_delta_aew.pkl`
- `models/iteration_0_models/scaler_2_p_proxy.pkl`

## Step 4: Generate the Next Batch

```bash
python run_optimization.py \
    --data_file results/iteration_0_experimental_plan.xlsx \
    --model_dir models/iteration_0_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 1
```

Arguments:

| Argument         | Default   | Description                               |
| ---------------- | --------- | ----------------------------------------- |
| `--data_file`    | required  | Excel file with all completed data so far |
| `--model_dir`    | required  | Directory with trained models             |
| `--output_dir`   | `results` | Output directory                          |
| `--n_candidates` | 4         | Number of new candidates                  |
| `--iteration`    | required  | Iteration number for the new batch        |
| `--smoke_test`   | False     | Use reduced computation for quick testing |

Outputs:

- `results/Iteration_1/Iteration_1_experimental_plan.xlsx`
- `results/experimental_database.xlsx`

Notes:

- When enabled, the urea condition is passed into the acquisition optimizer as a nonlinear inequality constraint.
- The script still performs a repair pass afterward as a safety fallback.
- `experimental_database.xlsx` logs generated batches, but it does not automatically become your next training file because newly proposed experiments do not yet have measured objectives.

## Step 5: Continue the Loop

After you finish the next batch of experiments, append the completed rows to a master dataset and retrain from that combined file.

Example:

```bash
python -c "import pandas as pd; df0 = pd.read_excel('results/iteration_0_experimental_plan.xlsx'); df1 = pd.read_excel('results/Iteration_1/Iteration_1_experimental_plan.xlsx'); pd.concat([df0, df1], ignore_index=True).to_excel('results/combined_iteration_1.xlsx', index=False)"
```

Then retrain and generate the next suggestions:

```bash
python train_models.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models \
    --project_name iteration_1_models
```

```bash
python run_optimization.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models/iteration_1_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 2
```

## Demo Mode

To exercise the full workflow with synthetic data:

```bash
python demo_workflow.py --n_iterations 5 --n_initial 12 --n_candidates 4
```

## Troubleshooting

### Module import errors

```bash
conda activate bayesopt-fluorescence
```

### Missing required columns

Check that the Excel file contains all parameter columns plus measured values for `Delta AEW` and `p_proxy`.

### Model file not found

Check that `--model_dir` points to the subdirectory created by `train_models.py`.

## Notebook vs CLI

| Feature       | Notebook                  | CLI                       |
| ------------- | ------------------------- | ------------------------- |
| Best for      | Guided exploration        | Automation and scripting  |
| Interaction   | Step-by-step notebook     | Repeatable shell commands |
| Visualization | Inline                    | Saved files and logs      |
| Demo path     | `workshop_notebook.ipynb` | `demo_workflow.py`        |
