#!/usr/bin/env python3
"""
Bayesian Optimization loop for protein refolding.

This script implements the main BO loop using qNEHVI for multi-objective
optimization of Delta AEW and p_proxy.
"""

import os
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.sampling import SobolQMCNormalSampler

from config import (
    ExperimentConfig, OptimizationConfig, ModelConfig, ConstraintConfig,
    get_normalized_bounds, get_optimization_params
)
from data.preprocessing import prepare_data, load_scalers, inverse_transform_objectives
from data.transformation import build_transformer
from models import GPModel, load_gp_model
from acquisition import create_qnehvi_acquisition, optimize_qnehvi
from acquisition.utils import update_experimental_database
from constraints import correct_constraints_iterative, calculate_urea_refolding_concentration
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trained_models(model_dir: str, train_x: torch.Tensor, train_y: torch.Tensor):
    """Load previously trained GP models.

    Args:
        model_dir: Directory containing saved models.
        train_x: Training inputs (for model reconstruction).
        train_y: Training outputs (for model reconstruction).

    Returns:
        Tuple of (ModelListGP, list of scalers).
    """
    models = []
    scalers = []

    # Load models and scalers
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    scaler_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

    for i, model_file in enumerate(sorted(model_files)):
        model_path = os.path.join(model_dir, model_file)
        model, _ = load_gp_model(model_path, GPModel, train_x, train_y, i)
        models.append(model)

    for i, scaler_file in enumerate(sorted(scaler_files)):
        scaler_path = os.path.join(model_dir, scaler_file)
        scaler = load_scalers(scaler_path)[0]
        scalers.append(scaler)

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
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run in smoke test mode (reduced computation)')

    args = parser.parse_args()

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

    # Load existing data
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
    X_raw = completed_df[ExperimentConfig.PARAMETER_NAMES].to_numpy()
    y_raw = completed_df[ExperimentConfig.OBJECTIVE_NAMES].to_numpy()

    logger.info(f"Loaded {len(completed_df)} completed experiments")

    # Prepare data
    transformer = build_transformer(ExperimentConfig)
    train_x_normalized, train_y_standardized, scalers = prepare_data(X_raw, y_raw, transformer)

    # Load trained models
    logger.info("Loading trained models...")
    multi_model, model_scalers = load_trained_models(args.model_dir, train_x_normalized, train_y_standardized)

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
    reference_point = OptimizationConfig.REFERENCE_POINT
    if OptimizationConfig.AUTO_REFERENCE_POINT:
        reference_point = torch.tensor(
            auto_reference_point(
                train_y_standardized.detach().cpu().numpy(),
                ["maximize"] * len(ExperimentConfig.OBJECTIVE_NAMES),
                OptimizationConfig.REFERENCE_POINT_MARGIN_FRACTION,
            ),
            dtype=torch.float64,
        )
        logger.info(f"Using auto reference point in standardized space: {reference_point.tolist()}")

    acq_function = create_qnehvi_acquisition(
        model=multi_model,
        reference_point=reference_point,
        X_baseline=train_x_normalized,
        sampler=qnehvi_sampler
    )

    # Set up nonlinear constraints if enabled
    nonlinear_inequality_constraints = None
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info(f"Urea constraint enabled (solubilization_urea={ConstraintConfig.SOLUBILIZATION_UREA} M)")
        def model_space_urea_constraint(samples: torch.Tensor) -> torch.Tensor:
            samples_physical = transformer.unit_to_physical_model(samples, as_tensor=True)
            final_urea = samples_physical[..., ConstraintConfig.FINAL_UREA_IDX]
            dilution_factor = samples_physical[..., ConstraintConfig.DILUTION_FACTOR_IDX]
            return final_urea * dilution_factor - ConstraintConfig.SOLUBILIZATION_UREA

        nonlinear_inequality_constraints = [(model_space_urea_constraint, True)]

    # Optimize acquisition function
    logger.info(f"Optimizing acquisition function for {args.n_candidates} candidates...")
    candidates_normalized, acq_metadata = optimize_qnehvi(
        acq_function=acq_function,
        bounds=normalized_bounds,
        batch_size=args.n_candidates,
        sequential=True,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        return_metadata=True,
        **opt_params
    )

    # Convert model-space candidates back to physical units
    candidates_original = transformer.unit_to_physical_model(candidates_normalized, as_tensor=True)

    # The optimizer should return feasible points already. Keep a repair fallback
    # for numerical edge cases or future constraint changes.
    final_candidates = candidates_original.double()

    # Verify constraint satisfaction (sanity check when constraint is enabled)
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info("Verifying constraint satisfaction for generated candidates...")
        repaired_candidates = []
        for i, candidate in enumerate(final_candidates):
            final_urea = candidate[ConstraintConfig.FINAL_UREA_IDX].item()
            dilution_factor = candidate[ConstraintConfig.DILUTION_FACTOR_IDX].item()
            constraint_value = final_urea * dilution_factor - ConstraintConfig.SOLUBILIZATION_UREA
            if constraint_value <= 0:
                logger.warning(f"Candidate {i+1} violates constraint: "
                             f"final_urea={final_urea:.3f}, dilution_factor={dilution_factor:.3f}, "
                             f"constraint_value={constraint_value:.3f}")
                repaired_candidates.append(candidate.numpy())
            else:
                repaired_candidates.append(candidate.numpy())
        repaired_candidates = correct_constraints_iterative(repaired_candidates)
        final_candidates = torch.from_numpy(np.array(repaired_candidates)).double()

    # Create DataFrame for new experiments
    new_experiments_df = pd.DataFrame(
        final_candidates.numpy(),
        columns=ExperimentConfig.PARAMETER_NAMES
    )
    new_experiments_df.insert(0, "Iteration", args.iteration)
    new_experiments_df.insert(1, "Status", "suggested")

    # Add placeholder columns for objectives (to be filled after experiments)
    for obj_name in ExperimentConfig.OBJECTIVE_NAMES:
        new_experiments_df[obj_name] = np.nan

    # Save new experimental plan
    plan_path = output_dir / f"Iteration_{args.iteration}_experimental_plan.xlsx"
    new_experiments_df = add_experiment_ids(new_experiments_df)
    new_experiments_df = mark_duplicate_parameters(new_experiments_df, ExperimentConfig.PARAMETER_NAMES)
    new_experiments_df.to_excel(plan_path, index=False)
    logger.info(f"Saved experimental plan to {plan_path}")

    # Update experimental database
    db_path = output_dir.parent / "experimental_database.xlsx"
    update_experimental_database(new_experiments_df.copy(), args.iteration, str(db_path))

    # Print summary
    print(f"\nOptimization Results:")
    print(f"Iteration: {args.iteration}")
    print(f"New candidates: {args.n_candidates}")
    print(f"Plan saved to: {plan_path}")
    print(f"Database updated: {db_path}")

    print(f"\nCandidate Summary:")
    for i, candidate in enumerate(final_candidates):
        print(f"  Candidate {i+1}:")
        for j, param_name in enumerate(ExperimentConfig.PARAMETER_NAMES):
            print(f"    {param_name}: {candidate[j]:.3f}")

    # Calculate predicted performance for candidates (optional - may fail for some model configurations)
    candidate_report = None
    try:
        logger.info("Predicting performance for new candidates...")
        with torch.no_grad():
            candidate_normalized = transformer.physical_to_unit_model(final_candidates, as_tensor= True)
            candidate_normalized = candidate_normalized.double()
            posterior = multi_model.posterior(candidate_normalized)

        pred_standardized = posterior.mean.detach().cpu().numpy()
        pred_std_standardized = posterior.variance.sqrt().detach().cpu().numpy()

        # Convert predictions back to original scale
        pred_original = inverse_transform_objectives(torch.from_numpy(pred_standardized), model_scalers)
        pred_upper_original = inverse_transform_objectives(
            torch.from_numpy(pred_standardized + pred_std_standardized), model_scalers
        )
        pred_std_original = pred_upper_original - pred_original

        with torch.no_grad():
            acquisition_values = []
            for candidate in candidates_normalized:
                acquisition_values.append(float(acq_function(candidate.unsqueeze(0)).detach().cpu().reshape(-1)[0]))
        acquisition_values = np.asarray(acquisition_values, dtype=float)

        constraint_margins = []
        urea_refolding = []
        for candidate in final_candidates:
            final_urea = candidate[ConstraintConfig.FINAL_UREA_IDX].item()
            dilution_factor = candidate[ConstraintConfig.DILUTION_FACTOR_IDX].item()
            constraint_margins.append(final_urea * dilution_factor - ConstraintConfig.SOLUBILIZATION_UREA)
            urea_refolding.append(calculate_urea_refolding_concentration(final_urea, dilution_factor))

        candidate_report = build_candidate_report(
            new_experiments_df,
            ExperimentConfig.PARAMETER_NAMES,
            ExperimentConfig.OBJECTIVE_NAMES,
            predicted_means=pred_original,
            predicted_stds=pred_std_original,
            acquisition_values=acquisition_values,
            existing_unit_x=train_x_normalized,
            candidate_unit_x=candidate_normalized,
            objective_directions=ExperimentConfig.OBJECTIVE_DIRECTIONS,
            constraint_margins=np.asarray(constraint_margins),
            urea_refolding=np.asarray(urea_refolding),
        )
        report_path = output_dir / f"Iteration_{args.iteration}_candidate_report.xlsx"
        save_dataframe_report(candidate_report, report_path)
        save_dataframe_report(candidate_report, output_dir / f"Iteration_{args.iteration}_candidate_report.csv")
        save_json_report(acq_metadata, output_dir / f"Iteration_{args.iteration}_acquisition_metadata.json")
        logger.info(f"Saved candidate report to {report_path}")

        print(f"\nPredicted Performance:")
        for i, pred in enumerate(pred_original):
            print(f"  Candidate {i+1}:")
            for j, obj_name in enumerate(ExperimentConfig.OBJECTIVE_NAMES):
                print(f"    {obj_name}: {pred[j]:.3f}")
    except Exception as e:
        logger.warning(f"Could not predict performance for candidates: {e}")
        logger.info("Optimization completed successfully (predictions skipped)")

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
        save_dataframe_report(progress_df, output_dir / f"Iteration_{args.iteration}_campaign_progress.xlsx")
        plot_hypervolume_progress(progress_df, output_dir / f"Iteration_{args.iteration}_hypervolume_progress.png")
        plot_objective_tradeoff(
            df,
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
