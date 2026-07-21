# Analysis and Reporting Workflow Manual

This package now includes reusable campaign-analysis utilities and richer CLI
outputs for Bayesian optimization iterations.

## New Analysis Package

The `analysis/` package contains reusable functions for:

- Pareto-front detection and two-objective hypervolume calculation
- campaign progress summaries by iteration
- candidate reports with model predictions, uncertainty, acquisition values,
  feasibility margins, and nearest-neighbor distances
- design-space, objective-tradeoff, hypervolume, calibration, and residual plots
- experiment database validation and completed-result merging
- run metadata JSON files with package versions, inputs, hashes, and config

Important modules:

- `analysis/pareto.py` - Pareto masks, Pareto fronts, auto reference points,
  hypervolume, and progress tables
- `analysis/plots.py` - reusable plotting functions
- `analysis/reporting.py` - candidate and JSON/Excel report helpers
- `analysis/database.py` - experiment database validation and merge helpers
- `analysis/metadata.py` - reproducibility metadata helpers

## Training Outputs

Run training as before, but `--data_file` is now required:

```bash
uv run python train_models.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models \
    --project_name iteration_1_models
```

New files written to the model directory:

- `validation_summary.xlsx` and `validation_summary.json`
- `training_metadata.json`
- `objective_N_<name>_training_loss.png`
- `objective_N_<name>_lengthscales.xlsx`
- `objective_N_validation_details.xlsx`
- `objective_N_validation_calibration.png`
- `objective_N_validation_residuals_by_parameter.png`
- existing LOOCV and parity plots

The lengthscale table is an ARD feature-relevance diagnostic. Smaller
lengthscales, or larger `Relative Relevance`, indicate stronger model
sensitivity to that parameter.

## Optimization Outputs

Run optimization as before, but both `--data_file` and `--model_dir` are now
required:

```bash
uv run python run_optimization.py \
    --data_file results/combined_iteration_1.xlsx \
    --model_dir models/iteration_1_models \
    --output_dir results \
    --n_candidates 4 \
    --iteration 2
```

New files written to `results/Iteration_N/`:

- `Iteration_N_experimental_plan.xlsx`
- `Iteration_N_candidate_report.xlsx`
- `Iteration_N_candidate_report.csv`
- `Iteration_N_acquisition_metadata.json`
- `Iteration_N_optimization_metadata.json`
- `Iteration_N_campaign_progress.xlsx`
- `Iteration_N_hypervolume_progress.png`
- `Iteration_N_objective_tradeoff.png`
- `Iteration_N_design_space_pairplot.png`

The candidate report includes:

- candidate parameters
- predicted objective means in original units
- predicted objective standard deviations in original units
- per-candidate acquisition values
- predicted non-dominated status
- distance to the nearest previous experiment in unit model space
- urea constraint margin
- urea refolding concentration

Rows without completed objective values are ignored for GP model input, so a
master database may contain suggested but unmeasured experiments.

## Configuration Additions

`config.py` now includes:

- `ExperimentConfig.OBJECTIVE_DIRECTIONS`
- `OptimizationConfig.AUTO_REFERENCE_POINT`
- `OptimizationConfig.REFERENCE_POINT_MARGIN_FRACTION`
- `OptimizationConfig.MIN_CANDIDATE_DISTANCE`
- `OptimizationConfig.ENABLE_REPLICATE_SUGGESTIONS`

The current qNEHVI path still assumes maximization-oriented model targets. Use
`OBJECTIVE_DIRECTIONS` for reporting and analysis consistency.

## Tests

Run the test suite with:

```bash
uv run pytest -q
```

Current tests cover:

- Pareto and hypervolume calculations
- campaign progress summaries
- parameter transformation round trips, including reciprocal dilution
- urea constraint repair
- constrained initial-design feasibility

## Notes

The workshop notebook was updated to convert optimized candidates through
`ParameterTransformer.unit_to_physical_model(...)` instead of BoTorch's linear
`unnormalize(...)`. This matters for non-linear parameter transforms such as the
reciprocal dilution-factor transform.
