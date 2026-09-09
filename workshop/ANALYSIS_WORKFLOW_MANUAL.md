# Analysis and Reporting Workflow Manual

The repository includes reusable campaign-analysis utilities and richer CLI
outputs for Bayesian optimization iterations. This manual describes what each
step writes and how to read it.

## The analysis package

The `analysis/` package contains reusable functions for:

- Pareto-front detection and two-objective hypervolume calculation
- Campaign progress summaries by iteration
- Candidate reports with model predictions, uncertainty, acquisition values,
  feasibility margins, and nearest-neighbor distances
- Design-space, objective-trade-off, hypervolume, calibration, and residual plots
- Experiment database validation and bookkeeping helpers
- Run metadata JSON files with package versions, inputs, hashes, and config

Important modules:

- `analysis/pareto.py` — Pareto masks, auto reference points, 2-D hypervolume,
  and progress tables
- `analysis/plots.py` — reusable plotting functions
- `analysis/reporting.py` — candidate and JSON/Excel report helpers
- `analysis/database.py` — experiment ID and duplicate-flag helpers
- `analysis/metadata.py` — reproducibility metadata helpers

## Training outputs

Run training as before (`--data_file` is required):

```bash
uv run python train_models.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir trained_models \
    --project_name iteration_1_models
```

Files written to the model directory:

- `model_N_<name>.pth` / `scaler_N_<name>.pkl` — model and scaler checkpoints
  (with a fingerprint of their training data and configuration)
- `training_metadata.json` — software versions, configuration snapshot, and
  the data file's content hash
- `objective_N_<name>_lengthscales.xlsx` — ARD feature-relevance table per
  objective. Smaller lengthscales (larger `Relative Relevance`) indicate
  stronger model sensitivity to that parameter.
- `objective_N_validation_validation_table.xlsx` — per-sample LOOCV results
  in standardized and original units
- `objective_N_validation_calibration.png` — predicted vs actual with
  uncertainty bands
- `objective_N_validation_residuals_by_parameter.png` — LOOCV residuals
  against each process parameter
- `objective_N_validation_loocv_objective_N.png` and
  `objective_N_validation_parity_objective_N.png` — existing LOOCV views

## Optimization outputs

Run optimization as before (`--data_file` and `--model_dir` are required):

```bash
uv run python run_optimization.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir trained_models/iteration_1_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 2 \
    --seed 42
```

The same `--data_file` plus the same `--seed` always yield the same candidate
batch.

Files written to `results/Iteration_N/`:

- `Iteration_N_experimental_plan.xlsx` — the new experiments, with
  `Experiment ID`, `Iteration`, and `Status` bookkeeping columns
- `Iteration_N_candidate_report.xlsx` / `.csv` — candidate parameters,
  predicted objective means and standard deviations (original units),
  per-candidate acquisition values, predicted non-dominated status, distance
  to the nearest previous experiment (unit model space), urea constraint
  margin, and the urea concentration the refolding buffer must have
- `Iteration_N_acquisition_metadata.json` — optimizer settings, runtime, and
  final acquisition value
- `Iteration_N_campaign_progress.xlsx` — cumulative experiment count, Pareto
  points, hypervolume, and best values per iteration
- `Iteration_N_hypervolume_progress.png`,
  `Iteration_N_objective_tradeoff.png`,
  `Iteration_N_design_space_pairplot.png` — campaign diagnostics
- `Iteration_N_optimization_metadata.json` — run metadata (versions, config,
  data file hash, seed, acquisition metadata)

Rows without completed objective values are ignored for GP model input, so a
master database may contain suggested but unmeasured experiments.

## Configuration additions

`config.py` includes:

- `ExperimentConfig.OBJECTIVE_DIRECTIONS` — `"maximize"` or `"minimize"` per
  objective; used for reporting and Pareto analysis
- `OptimizationConfig.AUTO_REFERENCE_POINT` — derive the qNEHVI reference
  point from the observed standardized objectives (worst value minus a margin
  fraction of the observed span) instead of `REFERENCE_POINT`
- `OptimizationConfig.REFERENCE_POINT_MARGIN_FRACTION` — the margin fraction

The qNEHVI model targets remain maximization-oriented; use
`OBJECTIVE_DIRECTIONS` for consistent reporting and analysis.

## Tests

Run the test suite with:

```bash
uv run pytest -q
```

Current tests cover the constraint math (linear vs physical equivalence),
parameter-transform round-trips, initial-design strategies and feasibility,
checkpoint fingerprinting, data validation, Pareto and hypervolume
calculations, campaign progress summaries, and an end-to-end qNEHVI smoke
test.
