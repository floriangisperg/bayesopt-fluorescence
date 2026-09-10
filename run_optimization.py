#!/usr/bin/env python3
"""
Bayesian Optimization loop for protein refolding.

This script implements the main BO loop using qNEHVI for multi-objective
optimization of Delta AEW and p_proxy.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.sampling import SobolQMCNormalSampler

from acquisition import create_qnehvi_acquisition, optimize_qnehvi
from acquisition.utils import update_experimental_database
from analysis import (
    add_experiment_ids,
    auto_reference_point,
    build_candidate_report,
    plot_design_space_pairplot,
    plot_hypervolume_progress,
    plot_objective_tradeoff,
    save_dataframe_report,
    save_json_report,
    save_run_metadata,
    summarize_campaign_progress,
    validate_experimental_dataframe,
)
from analysis.database import mark_duplicate_parameters
from analysis.metadata import file_sha256
from config import (
    ConstraintConfig,
    ExperimentConfig,
    LoggingConfig,
    OptimizationConfig,
    get_normalized_bounds,
    get_optimization_params,
)
from constraints import assert_urea_feasible, get_urea_linear_constraint
from constraints.urea_dilution import calculate_urea_refolding_concentration
from data.preprocessing import (
    inverse_transform_objectives,
    load_scalers,
    prepare_data,
    standardize_reference_point,
)
from data.transformation import build_transformer
from models import GPModel, load_gp_model, sort_objective_files

# Set up logging
logging.basicConfig(
    level=LoggingConfig.LOG_LEVEL,
    format=LoggingConfig.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def load_trained_models(model_dir: str, train_x: torch.Tensor, train_y: torch.Tensor,
                        expected_scalers=None):
    """Load previously trained GP models.

    Args:
        model_dir: Directory containing saved models.
        train_x: Training inputs (for model reconstruction).
        train_y: Training outputs (for model reconstruction).
        expected_scalers: Scalers fitted to the current raw measurements, used
            to detect shifts or rescaling hidden by standardized-target hashes.

    Returns:
        Tuple of (ModelListGP, list of scalers).
    """
    models = []
    scalers = []

    # Load models and scalers (ordered by embedded objective index)
    model_files = sort_objective_files(
        [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    )
    scaler_files = sort_objective_files(
        [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    )

    if len(model_files) != train_y.shape[1] or len(scaler_files) != train_y.shape[1]:
        raise ValueError("Expected exactly one model and scaler per objective; retrain in a clean directory.")

    for i, model_file in enumerate(model_files):
        model_path = os.path.join(model_dir, model_file)
        model, _ = load_gp_model(model_path, GPModel, train_x, train_y, i)
        models.append(model)

    for i, scaler_file in enumerate(scaler_files):
        scaler_path = os.path.join(model_dir, scaler_file)
        scaler = load_scalers(scaler_path)[0]
        scalers.append(scaler)

    # Standardization erases affine changes to raw measurements. Check its
    # fitted state too, including for checkpoints created before this guard.
    if expected_scalers is not None:
        if len(expected_scalers) != len(scalers):
            raise ValueError("Objective scaler count differs; retrain the models.")
        for i, (saved, current) in enumerate(zip(scalers, expected_scalers)):
            if any(not np.array_equal(getattr(saved, attr), getattr(current, attr))
                   for attr in ("mean_", "scale_", "var_", "n_samples_seen_")):
                raise ValueError(
                    f"Objective {i + 1} scaler does not match the current measurements. Retrain the models."
                )

    # Create ModelListGP for multi-objective optimization
    multi_model = ModelListGP(*models)

    return multi_model, scalers


def main():
    """Main optimization loop."""
    parser = argparse.ArgumentParser(description='Run Bayesian optimization loop')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Excel file with existing experimental data')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for new experiments')
    parser.add_argument('--n_candidates', type=int, default=4,
                       help='Number of new candidates to generate')
    parser.add_argument('--iteration', type=int, required=True,
                       help='Current iteration number')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for MC sampling and acquisition '
                            'optimization (same data + seed = same candidates)')
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run in smoke test mode (reduced computation)')

    args = parser.parse_args()

    # Seed torch so the MC sampler and acquisition optimizer are reproducible:
    # identical data and seed always yield the same candidate batch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set smoke test mode if requested
    if args.smoke_test:
        os.environ['SMOKE_TEST'] = '1'
        logger.info("Running in smoke test mode")

    logger.info(f"Starting optimization iteration {args.iteration}")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Model directory: {args.model_dir}")

    # Create output directory
    output_dir = Path(args.output_dir) / f"Iteration_{args.iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing data. The file may contain suggested-but-not-yet-run rows
    # (e.g. the experimental database), so incomplete rows are skipped with a
    # warning instead of failing the run; BO trains only on completed data.
    logger.info("Loading experimental data...")
    df = pd.read_excel(args.data_file)
    validate_experimental_dataframe(
        df,
        ExperimentConfig.PARAMETER_NAMES,
        ExperimentConfig.OBJECTIVE_NAMES,
        require_objectives=True,
    )
    if "Iteration" not in df.columns:
        df["Iteration"] = 0
    complete_mask = df[ExperimentConfig.OBJECTIVE_NAMES].notna().all(axis=1)
    if not complete_mask.all():
        dropped = int((~complete_mask).sum())
        logger.warning(f"Ignoring {dropped} rows without complete objective values")
    completed_df = df.loc[complete_mask].copy()
    if completed_df.empty:
        raise ValueError("At least one completed experiment is required for optimization")
    X_raw = completed_df[ExperimentConfig.PARAMETER_NAMES].to_numpy()
    y_raw = completed_df[ExperimentConfig.OBJECTIVE_NAMES].to_numpy()

    logger.info(f"Loaded {len(completed_df)} completed experiments")

    # Prepare data
    transformer = build_transformer(ExperimentConfig)
    train_x_normalized, train_y_standardized, scalers = prepare_data(X_raw, y_raw, transformer)

    # Load trained models
    logger.info("Loading trained models...")
    multi_model, model_scalers = load_trained_models(
        args.model_dir, train_x_normalized, train_y_standardized, expected_scalers=scalers
    )

    # Reference point for the hypervolume: either the configured real-space
    # point mapped into standardized space, or derived from the observed data
    # (worst value minus a fraction of the span), which is by construction
    # dominated by every observation.
    if OptimizationConfig.AUTO_REFERENCE_POINT:
        reference_point = torch.tensor(
            auto_reference_point(
                train_y_standardized.detach().cpu().numpy(),
                ["maximize"] * len(ExperimentConfig.OBJECTIVE_NAMES),
                OptimizationConfig.REFERENCE_POINT_MARGIN_FRACTION,
            ),
            dtype=torch.float64,
        )
        logger.info(f"Using auto reference point (standardized space): {reference_point.tolist()}")
    else:
        ref_point_real = OptimizationConfig.REFERENCE_POINT
        reference_point = standardize_reference_point(ref_point_real, model_scalers)
        for j, (value, obj_name) in enumerate(zip(ref_point_real, ExperimentConfig.OBJECTIVE_NAMES)):
            if y_raw[:, j].min() < value:
                logger.warning(
                    f"Reference point {value} for '{obj_name}' is above the worst "
                    f"observed value ({y_raw[:, j].min():.4f}); observations below it "
                    "cannot contribute to the hypervolume."
                )

    # Get optimization parameters
    opt_params = get_optimization_params()
    normalized_bounds = get_normalized_bounds(num_features=train_x_normalized.shape[1])

    # Initialize SobolQMCNormalSampler for better MC sampling
    logger.info("Initializing SobolQMCNormalSampler...")
    qnehvi_sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([opt_params["mc_samples"]])
    )

    # Create acquisition function
    logger.info("Creating qNEHVI acquisition function...")
    acq_function = create_qnehvi_acquisition(
        model=multi_model,
        reference_point=reference_point,
        X_baseline=train_x_normalized,
        sampler=qnehvi_sampler
    )

    # Set up the linear urea constraint if enabled
    inequality_constraints = None
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info(f"Urea constraint enabled (solubilization_urea={ConstraintConfig.SOLUBILIZATION_UREA} M)")
        inequality_constraints = [get_urea_linear_constraint()]

    # Optimize acquisition function
    logger.info(f"Optimizing acquisition function for {args.n_candidates} candidates...")
    candidates_normalized, acq_metadata = optimize_qnehvi(
        acq_function=acq_function,
        bounds=normalized_bounds,
        batch_size=args.n_candidates,
        sequential=OptimizationConfig.SEQUENTIAL_OPTIMIZATION,
        inequality_constraints=inequality_constraints,
        return_metadata=True,
        **opt_params
    )

    # Convert model-space candidates back to physical units
    final_candidates = transformer.unit_to_physical_model(candidates_normalized, as_tensor=True).double()

    # Validate constraint satisfaction in physical units before export. The
    # optimizer already enforces the constraint, so a violation here signals
    # an upstream numerical failure and must stop the run.
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info("Validating constraint satisfaction for generated candidates...")
        assert_urea_feasible(final_candidates)

    # Create DataFrame for new experiments, with bookkeeping columns so the
    # plan and database can be traced back to the proposal that made them
    new_experiments_df = pd.DataFrame(
        final_candidates.numpy(),
        columns=ExperimentConfig.PARAMETER_NAMES
    )

    # Add placeholder columns for objectives (to be filled after experiments)
    for obj_name in ExperimentConfig.OBJECTIVE_NAMES:
        new_experiments_df[obj_name] = np.nan

    new_experiments_df.insert(0, "Iteration", args.iteration)
    new_experiments_df.insert(1, "Status", "suggested")
    new_experiments_df = add_experiment_ids(new_experiments_df)
    new_experiments_df = mark_duplicate_parameters(
        new_experiments_df, ExperimentConfig.PARAMETER_NAMES
    )

    # Save new experimental plan
    plan_path = output_dir / f"Iteration_{args.iteration}_experimental_plan.xlsx"
    new_experiments_df.to_excel(plan_path, index=False)
    logger.info(f"Saved experimental plan to {plan_path}")

    # Update experimental database
    db_path = output_dir.parent / "experimental_database.xlsx"
    update_experimental_database(new_experiments_df.copy(), args.iteration, str(db_path))

    # Print summary
    print("\nOptimization Results:")
    print(f"Iteration: {args.iteration}")
    print(f"New candidates: {args.n_candidates}")
    print(f"Plan saved to: {plan_path}")
    print(f"Database updated: {db_path}")

    print("\nCandidate Summary:")
    for i, candidate in enumerate(final_candidates):
        print(f"  Candidate {i+1}:")
        for j, param_name in enumerate(ExperimentConfig.PARAMETER_NAMES):
            print(f"    {param_name}: {candidate[j]:.3f}")

    # Calculate predicted performance for the new candidates. Note: a
    # ModelListGP must not be called with a single shared input tensor
    # (gpytorch zips inputs with sub-models); predict per objective instead.
    candidate_report = None
    try:
        logger.info("Predicting performance for new candidates...")
        with torch.no_grad():
            candidate_normalized = transformer.physical_to_unit_model(final_candidates, as_tensor=True)
            candidate_normalized = candidate_normalized.double()
            pred_standardized = torch.stack(
                [model(candidate_normalized).mean for model in multi_model.models],
                dim=-1
            ).numpy()
            std_standardized = torch.stack(
                [model(candidate_normalized).variance.sqrt() for model in multi_model.models],
                dim=-1
            ).numpy()

            # Per-candidate acquisition value at the optimized points
            acquisition_values = np.asarray(
                [float(acq_function(c.unsqueeze(0)).detach().cpu().reshape(-1)[0])
                 for c in candidates_normalized],
                dtype=float,
            )

        # Convert predictions (and the +1 sigma upper edge, differenced back
        # to a standard deviation) to original units
        pred_original = inverse_transform_objectives(torch.from_numpy(pred_standardized), model_scalers)
        pred_upper_original = inverse_transform_objectives(
            torch.from_numpy(pred_standardized + std_standardized), model_scalers
        )
        pred_std_original = pred_upper_original - pred_original

        # Constraint context per candidate: margin above the feasibility
        # boundary and the urea concentration the refolding buffer must have
        constraint_margins = []
        urea_refolding = []
        for candidate in final_candidates:
            final_urea = candidate[ConstraintConfig.FINAL_UREA_IDX].item()
            dilution_factor = candidate[ConstraintConfig.DILUTION_FACTOR_IDX].item()
            constraint_margins.append(
                final_urea * dilution_factor - ConstraintConfig.SOLUBILIZATION_UREA
            )
            urea_refolding.append(
                calculate_urea_refolding_concentration(final_urea, dilution_factor)
            )

        candidate_report = build_candidate_report(
            new_experiments_df,
            ExperimentConfig.PARAMETER_NAMES,
            ExperimentConfig.OBJECTIVE_NAMES,
            predicted_means=pred_original,
            predicted_stds=pred_std_original,
            acquisition_values=acquisition_values,
            existing_unit_x=train_x_normalized,
            candidate_unit_x=candidates_normalized,
            objective_directions=ExperimentConfig.OBJECTIVE_DIRECTIONS,
            constraint_margins=np.asarray(constraint_margins),
            urea_refolding=np.asarray(urea_refolding),
        )
        report_path = output_dir / f"Iteration_{args.iteration}_candidate_report.xlsx"
        save_dataframe_report(candidate_report, report_path)
        save_dataframe_report(
            candidate_report, output_dir / f"Iteration_{args.iteration}_candidate_report.csv"
        )
        save_json_report(
            acq_metadata, output_dir / f"Iteration_{args.iteration}_acquisition_metadata.json"
        )
        logger.info(f"Saved candidate report to {report_path}")

        print("\nPredicted Performance:")
        for i, pred in enumerate(pred_original):
            print(f"  Candidate {i+1}:")
            for j, obj_name in enumerate(ExperimentConfig.OBJECTIVE_NAMES):
                print(f"    {obj_name}: {pred[j]:.3f}")
    except Exception as e:
        logger.warning(f"Could not predict performance for candidates: {e}")
        logger.info("Optimization completed successfully (predictions skipped)")

    # Campaign-level diagnostics: cumulative Pareto count, hypervolume
    # progress, objective trade-off, and the design space covered so far
    try:
        progress_reference = auto_reference_point(
            y_raw,
            ExperimentConfig.OBJECTIVE_DIRECTIONS,
            OptimizationConfig.REFERENCE_POINT_MARGIN_FRACTION,
        )
        progress_df = summarize_campaign_progress(
            completed_df,
            ExperimentConfig.OBJECTIVE_NAMES,
            ExperimentConfig.OBJECTIVE_DIRECTIONS,
            progress_reference,
        )
        save_dataframe_report(
            progress_df, output_dir / f"Iteration_{args.iteration}_campaign_progress.xlsx"
        )
        plot_hypervolume_progress(
            progress_df, output_dir / f"Iteration_{args.iteration}_hypervolume_progress.png"
        )
        plot_objective_tradeoff(
            completed_df,
            ExperimentConfig.OBJECTIVE_NAMES,
            output_dir / f"Iteration_{args.iteration}_objective_tradeoff.png",
            candidate_df=candidate_report,
            objective_directions=ExperimentConfig.OBJECTIVE_DIRECTIONS,
        )
        plot_design_space_pairplot(
            pd.concat([completed_df, new_experiments_df], ignore_index=True),
            ExperimentConfig.PARAMETER_NAMES,
            output_dir / f"Iteration_{args.iteration}_design_space_pairplot.png",
        )
    except Exception as e:
        logger.warning(f"Could not create campaign analysis plots: {e}")

    # Run metadata for traceability of this optimization step
    save_run_metadata(
        output_dir / f"Iteration_{args.iteration}_optimization_metadata.json",
        command="run_optimization.py",
        config={
            "parameter_names": ExperimentConfig.PARAMETER_NAMES,
            "objective_names": ExperimentConfig.OBJECTIVE_NAMES,
            "objective_directions": ExperimentConfig.OBJECTIVE_DIRECTIONS,
            "reference_point": reference_point.detach().cpu().tolist(),
            "auto_reference_point": OptimizationConfig.AUTO_REFERENCE_POINT,
            "batch_size": args.n_candidates,
            "seed": args.seed,
            "optimization_params": opt_params,
            "urea_constraint_enabled": ConstraintConfig.ENABLE_UREA_CONSTRAINT,
        },
        inputs={
            "data_file": args.data_file,
            "data_file_sha256": file_sha256(args.data_file),
            "model_dir": args.model_dir,
        },
        extra={
            "iteration": args.iteration,
            "acquisition": acq_metadata,
            "plan_path": str(plan_path),
            "database_path": str(db_path),
        },
    )

    logger.info("Optimization completed successfully")


if __name__ == "__main__":
    main()
