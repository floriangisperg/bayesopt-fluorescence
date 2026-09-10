#!/usr/bin/env python3
"""
Demo script for the Bayesian optimization workflow.

This script demonstrates the complete pipeline:
1. Generate initial experimental design
2. Mock experimental results using synthetic functions
3. Train GP models
4. Run Bayesian optimization to generate new candidates
5. Iterate multiple times

Usage:
    python demo_workflow.py [--n_iterations N] [--n_initial N] [--smoke_test]
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is on the path when running from the workshop/ directory
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import project modules
from botorch.models import ModelListGP
from botorch.sampling import SobolQMCNormalSampler

from acquisition import create_qnehvi_acquisition, optimize_qnehvi
from acquisition.utils import generate_initial_design
from config import (
    ConstraintConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
)
from constraints import assert_urea_feasible, get_urea_linear_constraint
from data.preprocessing import load_scalers, prepare_data, save_scalers, standardize_reference_point
from data.transformation import build_transformer
from models import GPModel, fit_gp_model, load_gp_model, save_gp_model, sort_objective_files

# Set up logging
logging.basicConfig(
    level=LoggingConfig.LOG_LEVEL,
    format=LoggingConfig.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def synthetic_objective_1(X: np.ndarray) -> np.ndarray:
    """
    Synthetic objective function 1: Delta AEW (simulated)
    Uses a multimodal function with some noise.

    Physics-inspired: High DTT and moderate GSSG tend to give better yields.
    """
    DTT = X[:, 0]      # 0-25 mM
    GSSG = X[:, 1]     # 0-2.5 mM
    Dilution = X[:, 2] # 2-40
    pH = X[:, 3]       # 8-11
    Urea = X[:, 4]     # 0-6 M

    # Synthetic function with multiple peaks
    obj1 = (
        0.5 * np.sin(DTT / 3.0) +
        0.3 * np.cos(GSSG * 2.0) +
        0.2 * np.sin(Dilution / 5.0) +
        0.15 * (pH - 9.5)**2 +
        0.1 * np.exp(-(Urea - 2.0)**2)
    )

    # Add some noise
    noise = np.random.normal(0, 0.1, size=len(X))
    obj1 = obj1 + noise + 1.0  # Shift to be mostly positive

    return obj1


def synthetic_objective_2(X: np.ndarray) -> np.ndarray:
    """
    Synthetic objective function 2: p_proxy (simulated)
    Measured concentration proxy tends to be better with different conditions.

    Physics-inspired: Higher dilution improves signal (reduces inner filter effects)
    and lower urea improves protein stability, both leading to better measured values.
    """
    DTT = X[:, 0]      # 0-25 mM
    GSSG = X[:, 1]     # 0-2.5 mM
    Dilution = X[:, 2] # 2-40
    pH = X[:, 3]       # 8-11
    Urea = X[:, 4]     # 0-6 M

    # Synthetic function (different landscape from objective 1)
    obj2 = (
        0.4 * np.cos(DTT / 4.0) +
        0.35 * np.sin(GSSG * 3.0) +
        0.25 * np.log(Dilution + 1) +
        0.2 * np.abs(pH - 10.0) +
        0.15 * (6.0 - Urea) / 6.0
    )

    # Add some noise
    noise = np.random.normal(0, 0.04, size=len(X))
    obj2 = obj2 + noise + 0.5  # Shift to be mostly positive

    return obj2


def generate_initial_design_with_mock_results(
    n_samples: int,
    output_dir: str,
    seed: int = 42
) -> pd.DataFrame:
    """Generate initial design and add mock experimental results."""
    logger.info(f"=== STEP 1: Generating Initial Design ({n_samples} samples) ===")

    # Create output directory for the initial design
    output_path = Path(output_dir) / "Iteration_0"
    output_path.mkdir(parents=True, exist_ok=True)

    # Build the current parameter transformer and use its physical bounds.
    transformer = build_transformer(ExperimentConfig)
    bounds = transformer.get_physical_bounds(as_tensor=True)

    # Apply the same constraint-aware initial design as the CLI workflow
    design_strategy = "lhs"
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info(f"Urea constraint enabled (solubilization_urea={ConstraintConfig.SOLUBILIZATION_UREA} M)")
        design_strategy = "feasible_coverage"

    # Generate a feasible coverage design (plain LHS if unconstrained)
    samples = generate_initial_design(
        n_samples=n_samples,
        bounds=bounds,
        transformer=transformer,
        seed=seed,
        n_candidates=50,  # Reduced for testing
        use_maximin=True,
        design_strategy=design_strategy,
        solubilization_urea=ConstraintConfig.SOLUBILIZATION_UREA
    )

    # The selected samples are feasible by construction; validate loudly
    # rather than repairing silently.
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        assert_urea_feasible(samples.numpy())
    final_samples = samples

    # Create DataFrame with parameters
    df = pd.DataFrame(
        data=final_samples.numpy(),
        columns=ExperimentConfig.PARAMETER_NAMES
    )

    # Generate mock experimental results
    X_raw = df[ExperimentConfig.PARAMETER_NAMES].to_numpy()
    df[ExperimentConfig.OBJECTIVE_NAMES[0]] = synthetic_objective_1(X_raw)
    df[ExperimentConfig.OBJECTIVE_NAMES[1]] = synthetic_objective_2(X_raw)

    # Save to Excel
    results_file = output_path / "Iteration_0_analysis_results_combined.xlsx"
    df.to_excel(results_file, index=False)
    logger.info(f"Saved initial design with mock results to {results_file}")

    # Print summary
    print("\nInitial Design Summary:")
    print(f"  Total samples: {len(df)}")
    for i, obj_name in enumerate(ExperimentConfig.OBJECTIVE_NAMES):
        print(f"  Objective {i+1} ({obj_name}): {df[obj_name].mean():.3f} ± {df[obj_name].std():.3f}")

    return df


def train_gp_models(
    data_file: str,
    model_save_dir: str
):
    """Train GP models from experimental data."""
    logger.info("=== STEP 2: Training GP Models ===")

    # Load data
    df = pd.read_excel(data_file)
    logger.info(f"Loaded {len(df)} samples from {data_file}")

    # Prepare training data
    parameter_names = ExperimentConfig.PARAMETER_NAMES
    objective_names = ExperimentConfig.OBJECTIVE_NAMES
    transformer = build_transformer(ExperimentConfig)

    X_raw = df[parameter_names].to_numpy()
    y_raw = df[objective_names].to_numpy()

    train_x_normalized, train_y_standardized, scalers = prepare_data(X_raw, y_raw, transformer)

    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)

    # Train models for each objective
    models = []

    for i, obj_name in enumerate(objective_names):
        logger.info(f"Training model {i+1}/{len(objective_names)}: {obj_name}")

        # Extract single objective (keep as 1D for GPyTorch compatibility)
        train_y_single = train_y_standardized[:, i]

        # Train model
        model, likelihood = fit_gp_model(
            train_x=train_x_normalized,
            train_y=train_y_single,
            model_class=GPModel,
            noise=ModelConfig.INITIAL_NOISE_LEVEL
        )

        # Save model
        model_name = f"model_{i+1}_{obj_name.replace(' ', '_').lower()}.pth"
        model_path = os.path.join(model_save_dir, model_name)
        save_gp_model(model, likelihood, model_path)

        # Save scaler
        scaler_name = f"scaler_{i+1}_{obj_name.replace(' ', '_').lower()}.pkl"
        scaler_path = os.path.join(model_save_dir, scaler_name)
        save_scalers([scalers[i]], scaler_path)

        logger.info(f"  Fitted noise: {likelihood.noise.item():.4f}")
        models.append((model, likelihood))

    logger.info(f"Model training completed. Models saved to {model_save_dir}")
    return models, scalers, train_x_normalized, train_y_standardized


def run_bayesian_optimization(
    data_file: str,
    model_dir: str,
    output_dir: str,
    iteration: int,
    n_candidates: int = 4,
    smoke_test: bool = False
) -> pd.DataFrame:
    """Run Bayesian optimization to generate new candidates."""
    logger.info(f"=== STEP 3: Running Bayesian Optimization (Iteration {iteration}) ===")

    # Load existing data
    df = pd.read_excel(data_file)
    X_raw = df[ExperimentConfig.PARAMETER_NAMES].to_numpy()
    y_raw = df[ExperimentConfig.OBJECTIVE_NAMES].to_numpy()

    logger.info(f"Loaded {len(df)} existing experiments")

    # Prepare data
    transformer = build_transformer(ExperimentConfig)
    train_x_normalized, train_y_standardized, _ = prepare_data(X_raw, y_raw, transformer)

    # Ensure float64 dtype for consistency
    train_x_normalized = train_x_normalized.double()
    train_y_standardized = train_y_standardized.double()

    # Load trained models
    logger.info("Loading trained models...")
    models = []
    scalers = []

    # Ordered by embedded objective index
    model_files = sort_objective_files(
        [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    )
    scaler_files = sort_objective_files(
        [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    )

    for i, model_file in enumerate(model_files):
        model_path = os.path.join(model_dir, model_file)
        model, _ = load_gp_model(model_path, GPModel, train_x_normalized, train_y_standardized, i)
        models.append(model)
        logger.info(f"  Loaded: {model_file}")

    for i, scaler_file in enumerate(scaler_files):
        scaler_path = os.path.join(model_dir, scaler_file)
        scaler = load_scalers(scaler_path)[0]
        scalers.append(scaler)

    # Create ModelListGP
    multi_model = ModelListGP(*models)

    # Get optimization parameters
    if smoke_test:
        mc_samples, num_restarts, raw_samples = 128, 10, 64
    else:
        mc_samples = OptimizationConfig.MC_SAMPLES
        num_restarts = OptimizationConfig.NUM_RESTARTS
        raw_samples = OptimizationConfig.RAW_SAMPLES

    # Initialize sampler
    qnehvi_sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([mc_samples])
    )

    normalized_bounds = torch.stack([
        torch.zeros(train_x_normalized.shape[1], dtype=torch.float64),
        torch.ones(train_x_normalized.shape[1], dtype=torch.float64)
    ])

    # Map the real-space reference point into standardized objective space
    reference_point = standardize_reference_point(OptimizationConfig.REFERENCE_POINT, scalers)

    acq_function = create_qnehvi_acquisition(
        model=multi_model,
        reference_point=reference_point,
        X_baseline=train_x_normalized,
        sampler=qnehvi_sampler
    )

    # Apply the same linear constraint inside the acquisition optimizer as
    # the CLI workflow, so the demo exercises the production code path
    inequality_constraints = None
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info(f"Urea constraint enabled (solubilization_urea={ConstraintConfig.SOLUBILIZATION_UREA} M)")
        inequality_constraints = [get_urea_linear_constraint()]

    # Optimize acquisition function
    logger.info(f"Optimizing acquisition function for {n_candidates} candidates...")
    candidates_normalized = optimize_qnehvi(
        acq_function=acq_function,
        bounds=normalized_bounds,
        batch_size=n_candidates,
        mc_samples=mc_samples,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=True,
        inequality_constraints=inequality_constraints
    )

    # Convert candidates from model unit space back to physical experiment units.
    final_candidates = transformer.unit_to_physical_model(candidates_normalized, as_tensor=True).double()

    # Validate in physical units before the plan is written
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        assert_urea_feasible(final_candidates.numpy())

    # Create DataFrame for new experiments
    new_experiments_df = pd.DataFrame(
        final_candidates.numpy(),
        columns=ExperimentConfig.PARAMETER_NAMES
    )

    # Add mock results for new candidates
    X_new = new_experiments_df[ExperimentConfig.PARAMETER_NAMES].to_numpy()
    new_experiments_df[ExperimentConfig.OBJECTIVE_NAMES[0]] = synthetic_objective_1(X_new)
    new_experiments_df[ExperimentConfig.OBJECTIVE_NAMES[1]] = synthetic_objective_2(X_new)

    # Save new experimental plan
    output_path = Path(output_dir) / f"Iteration_{iteration}"
    output_path.mkdir(parents=True, exist_ok=True)

    plan_path = output_path / f"Iteration_{iteration}_experimental_plan.xlsx"
    new_experiments_df.to_excel(plan_path, index=False)
    logger.info(f"Saved experimental plan to {plan_path}")

    # Combine with existing data
    combined_df = pd.concat([df, new_experiments_df], ignore_index=True)

    # Save combined results
    results_file = output_path / f"Iteration_{iteration}_analysis_results_combined.xlsx"
    combined_df.to_excel(results_file, index=False)
    logger.info(f"Saved combined results to {results_file}")

    # Print summary
    print(f"\nIteration {iteration} Summary:")
    print(f"  New candidates: {n_candidates}")
    print(f"  Total experiments: {len(combined_df)}")

    # Print predicted vs actual performance
    print("\n  New Candidate Performance:")
    for i, (_, row) in enumerate(new_experiments_df.iterrows()):
        print(f"    Candidate {i+1}:")
        print(f"      {ExperimentConfig.OBJECTIVE_NAMES[0]}: {row[ExperimentConfig.OBJECTIVE_NAMES[0]]:.3f}")
        print(f"      {ExperimentConfig.OBJECTIVE_NAMES[1]}: {row[ExperimentConfig.OBJECTIVE_NAMES[1]]:.3f}")

    # Find best results so far
    best_idx_1 = combined_df[ExperimentConfig.OBJECTIVE_NAMES[0]].idxmax()
    best_idx_2 = combined_df[ExperimentConfig.OBJECTIVE_NAMES[1]].idxmax()

    print("\n  Best Results So Far:")
    print(f"    {ExperimentConfig.OBJECTIVE_NAMES[0]}: "
          f"{combined_df.loc[best_idx_1, ExperimentConfig.OBJECTIVE_NAMES[0]]:.3f}")
    print(f"    {ExperimentConfig.OBJECTIVE_NAMES[1]}: "
          f"{combined_df.loc[best_idx_2, ExperimentConfig.OBJECTIVE_NAMES[1]]:.3f}")

    return combined_df


def main():
    """Main test workflow."""
    parser = argparse.ArgumentParser(description='Integration test for BO workflow')
    parser.add_argument('--n_iterations', type=int, default=5,
                       help='Number of BO iterations')
    parser.add_argument('--n_initial', type=int, default=9,
                       help='Number of initial design samples')
    parser.add_argument('--n_candidates', type=int, default=4,
                       help='Number of candidates per iteration')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Output directory for test results')
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run in smoke test mode (faster)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("BAYESIAN OPTIMIZATION WORKFLOW INTEGRATION TEST")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"  Iterations: {args.n_iterations}")
    logger.info(f"  Initial samples: {args.n_initial}")
    logger.info(f"  Candidates per iteration: {args.n_candidates}")
    logger.info(f"  Smoke test: {args.smoke_test}")
    logger.info(f"  Random seed: {args.seed}")
    logger.info(f"  Output directory: {args.output_dir}")

    # Clean output directory if it exists
    if os.path.exists(args.output_dir):
        logger.info(f"Cleaning existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # STEP 1: Generate Iteration_0 (initial LHS design)
    generate_initial_design_with_mock_results(
        n_samples=args.n_initial,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # STEP 2: Train models on Iteration_0 and generate Iteration_1
    for iteration in range(1, args.n_iterations):
        logger.info("\n" + "=" * 70)
        logger.info(f"ITERATION {iteration}: Generating candidates from Iteration_{iteration-1} data")
        logger.info("=" * 70)

        # Data from all previous iterations (combined results)
        # For iteration 1: use Iteration_0 data
        # For iteration 2: use Iteration_1 combined data (which includes 0+1)
        if iteration == 1:
            prev_data_file = (Path(args.output_dir) / f"Iteration_{iteration-1}" /
                              f"Iteration_{iteration-1}_analysis_results_combined.xlsx")
        else:
            prev_data_file = (Path(args.output_dir) / f"Iteration_{iteration-1}" /
                              f"Iteration_{iteration-1}_analysis_results_combined.xlsx")

        # Train models on all data so far
        model_dir = Path(args.output_dir) / f"Iteration_{iteration-1}" / "models" / "gpytorch_singletaskgp_matern_25"
        models, scalers, train_x, train_y = train_gp_models(
            data_file=str(prev_data_file),
            model_save_dir=str(model_dir)
        )

        # Run optimization to generate Iteration N candidates
        run_bayesian_optimization(
            data_file=str(prev_data_file),
            model_dir=str(model_dir),
            output_dir=args.output_dir,
            iteration=iteration,
            n_candidates=args.n_candidates,
            smoke_test=args.smoke_test
        )

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    # Load final results (last iteration)
    final_iter = args.n_iterations - 1
    final_file = Path(args.output_dir) / f"Iteration_{final_iter}" / \
                 f"Iteration_{final_iter}_analysis_results_combined.xlsx"
    final_df = pd.read_excel(final_file)

    print("\nFinal Summary:")
    print(f"  Total experiments: {len(final_df)}")
    print(f"  Initial LHS samples (Iteration 0): {args.n_initial}")
    print(f"  BO iterations: {args.n_iterations - 1}")
    print(f"  Candidates per iteration: {args.n_candidates}")

    # Print best results
    for obj_name in ExperimentConfig.OBJECTIVE_NAMES:
        best_val = final_df[obj_name].max()
        best_idx = final_df[obj_name].idxmax()
        print(f"\n  Best {obj_name}:")
        print(f"    Value: {best_val:.3f}")
        print("    Parameters:")
        for param_name in ExperimentConfig.PARAMETER_NAMES:
            print(f"      {param_name}: {final_df.loc[best_idx, param_name]:.3f}")

    print(f"\nAll results saved to: {args.output_dir}")
    logger.info("Integration test completed successfully!")


if __name__ == "__main__":
    main()
