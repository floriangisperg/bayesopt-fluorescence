#!/usr/bin/env python3
"""
Generate a space-filling initial experimental design.

This script creates the initial set of experiments for the Bayesian optimization
loop using feasible-region coverage selection, or LHS without constraints.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from acquisition.utils import generate_initial_design
from config import ConstraintConfig, ExperimentConfig, LoggingConfig, get_transposed_bounds
from constraints.urea_dilution import calculate_urea_refolding_concentration
from data.transformation import build_transformer

# Set up logging
logging.basicConfig(
    level=LoggingConfig.LOG_LEVEL,
    format=LoggingConfig.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def main():
    """Main function to generate initial experimental design."""
    parser = argparse.ArgumentParser(description='Generate initial experimental design')
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of initial experiments')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--project_name', type=str, default='initial_design',
                       help='Project name for file naming')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--n_candidates', type=int, default=100,
                       help='Number of coverage starts or conditional LHD candidate designs')
    parser.add_argument('--no_maximin', action='store_true',
                       help='Disable design selection; use raw feasible Sobol points or plain LHS')
    parser.add_argument('--design_strategy', choices=['feasible_coverage', 'constrained_lhd'],
                       default='feasible_coverage', help='Initial strategy when the urea constraint is enabled')

    args = parser.parse_args()

    logger.info(f"Generating initial design with {args.n_samples} samples")
    logger.info(f"Random seed: {args.seed}")
    if not args.no_maximin:
        logger.info(f"Using design selection with {args.n_candidates} starts/candidates")
    else:
        logger.info("Design selection disabled")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get bounds from config (in BoTorch format: 2 x d tensor)
    bounds = get_transposed_bounds()

    # Create transformer for parameter scaling (if needed)
    transformer = build_transformer(ExperimentConfig)

    # Set up the constraint-aware design strategy if enabled
    design_strategy = "lhs"
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info(f"Urea constraint enabled (solubilization_urea={ConstraintConfig.SOLUBILIZATION_UREA} M)")
        design_strategy = args.design_strategy

    # Generate initial design
    samples = generate_initial_design(
        n_samples=args.n_samples,
        bounds=bounds,
        transformer=transformer,
        seed=args.seed,
        n_candidates=args.n_candidates,
        use_maximin=not args.no_maximin,
        design_strategy=design_strategy,
        solubilization_urea=ConstraintConfig.SOLUBILIZATION_UREA
    )

    final_samples = samples

    # Build the complete experimental plan and write it once: parameters,
    # derived urea refolding concentrations, and empty objective columns so
    # the plan is ready to be filled in and accepted by train_models.py
    # without manual column creation
    df = pd.DataFrame(final_samples.numpy(), columns=ExperimentConfig.PARAMETER_NAMES)

    urea_refolding = [
        calculate_urea_refolding_concentration(row["Final Urea [M]"], row["Dilution Factor"])
        for _, row in df.iterrows()
    ]
    df["Urea Refolding [M]"] = urea_refolding

    for obj_name in ExperimentConfig.OBJECTIVE_NAMES:
        df[obj_name] = np.nan

    output_path = output_dir / f"{args.project_name}_experimental_plan.xlsx"
    df.to_excel(output_path, index=False)
    logger.info(f"Saved {len(df)} experiments to {output_path}")

    # Print summary statistics
    print("\nInitial Design Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Saved to: {output_path}")
    print("\nParameter ranges:")
    for i, name in enumerate(ExperimentConfig.PARAMETER_NAMES):
        print(f"{name}: {df[name].min():.2f} - {df[name].max():.2f}")

    print("\nUrea Refolding Concentration:")
    print(f"Min: {min(urea_refolding):.2f} M")
    print(f"Max: {max(urea_refolding):.2f} M")
    print(f"Mean: {np.mean(urea_refolding):.2f} M")

    logger.info("Initial design generation completed successfully")


if __name__ == "__main__":
    main()
