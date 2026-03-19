#!/usr/bin/env python3
"""
Generate initial experimental design using Latin Hypercube Sampling.

This script creates the initial set of experiments for the Bayesian optimization
loop using space-filling Latin Hypercube Sampling with maximin criterion
optimization and physical constraints.
"""

import os
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig, ConstraintConfig
from acquisition.utils import save_experiments_to_excel, generate_initial_design
from constraints.urea_dilution import urea_constraint_callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
                       help='Number of candidate designs for maximin optimization')
    parser.add_argument('--no_maximin', action='store_true',
                       help='Disable maximin criterion optimization')

    args = parser.parse_args()

    logger.info(f"Generating initial design with {args.n_samples} samples")
    logger.info(f"Random seed: {args.seed}")
    if not args.no_maximin:
        logger.info(f"Using maximin criterion with {args.n_candidates} candidates")
    else:
        logger.info("Maximin criterion disabled")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get bounds from config
    bounds = torch.from_numpy(ExperimentConfig.PARAMETER_BOUNDS).double()

    # Set up constraint callable if enabled
    constraint_callable = None
    if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        logger.info(f"Urea constraint enabled (solubilization_urea={ConstraintConfig.SOLUBILIZATION_UREA} M)")
        constraint_callable = urea_constraint_callable

    # Generate initial design with constraint-aware rejection sampling
    samples = generate_initial_design(
        n_samples=args.n_samples,
        bounds=bounds,
        seed=args.seed,
        n_candidates=args.n_candidates,
        use_maximin=not args.no_maximin,
        constraint_callable=constraint_callable
    )

    final_samples = samples

    # Save to Excel
    output_path = output_dir / f"{args.project_name}_experimental_plan.xlsx"
    df = save_experiments_to_excel(final_samples, str(output_path))

    # Print summary statistics
    print(f"\nInitial Design Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Saved to: {output_path}")
    print(f"\nParameter ranges:")
    for i, name in enumerate(ExperimentConfig.PARAMETER_NAMES):
        print(f"{name}: {df[name].min():.2f} - {df[name].max():.2f}")

    # Calculate and display urea refolding concentrations
    urea_refolding = []
    for _, row in df.iterrows():
        final_urea = row["Final Urea [M]"]
        dilution_factor = row["Dilution Factor"]
        urea_ref = ((final_urea * dilution_factor) - ConstraintConfig.SOLUBILIZATION_UREA) / (dilution_factor - 1)
        urea_refolding.append(urea_ref)

    df["Urea Refolding [M]"] = urea_refolding
    print(f"\nUrea Refolding Concentration:")
    print(f"Min: {min(urea_refolding):.2f} M")
    print(f"Max: {max(urea_refolding):.2f} M")
    print(f"Mean: {np.mean(urea_refolding):.2f} M")

    # Save updated DataFrame with refolding concentrations
    df.to_excel(output_path, index=False)
    logger.info("Initial design generation completed successfully")


if __name__ == "__main__":
    main()