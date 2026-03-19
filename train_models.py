#!/usr/bin/env python3
"""
Train Gaussian Process models from experimental data.

This script trains single-task GP models for each objective using
experimental data from previous iterations.
"""

import os
import logging
import argparse
from pathlib import Path

import pandas as pd
import torch

from config import ExperimentConfig, ModelConfig, PathConfig
from data.preprocessing import prepare_data
from models import GPModel, fit_gp_model, save_gp_model, loocv_gp_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experimental_data(data_file: str) -> pd.DataFrame:
    """Load experimental data from Excel file.

    Args:
        data_file: Path to Excel file with experimental data.

    Returns:
        DataFrame with experimental data.
    """
    logger.info(f"Loading experimental data from {data_file}")
    df = pd.read_excel(data_file)

    # Validate required columns
    required_param_cols = set(ExperimentConfig.PARAMETER_NAMES)
    required_obj_cols = set(ExperimentConfig.OBJECTIVE_NAMES)

    missing_cols = (required_param_cols | required_obj_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} experimental samples")
    return df


def train_objective_models(df: pd.DataFrame, model_save_dir: str):
    """Train GP models for each objective.

    Args:
        df: Experimental data.
        model_save_dir: Directory to save trained models.
    """
    # Prepare training data
    parameter_names = ExperimentConfig.PARAMETER_NAMES
    objective_names = ExperimentConfig.OBJECTIVE_NAMES
    bounds = ExperimentConfig.PARAMETER_BOUNDS

    logger.info("Preparing training data...")
    X_raw = df[parameter_names].to_numpy()
    y_raw = df[objective_names].to_numpy()

    train_x_normalized, train_y_standardized, scalers = prepare_data(X_raw, y_raw, bounds)

    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)

    # Train separate model for each objective
    models = []
    validation_results = {}

    for i, obj_name in enumerate(objective_names):
        logger.info(f"Training model for objective: {obj_name}")

        # Extract single objective (keep as 1D for GPyTorch compatibility)
        train_y_single = train_y_standardized[:, i]

        # Train model
        model, likelihood, losses = fit_gp_model(
            train_x=train_x_normalized,
            train_y=train_y_single,
            model_class=GPModel,
            noise=ModelConfig.INITIAL_NOISE_LEVEL,
            num_train_iters=ModelConfig.NUM_TRAINING_ITERATIONS,
            lr=ModelConfig.LEARNING_RATE
        )

        # Save model and likelihood
        model_name = f"model_{i+1}_{obj_name.replace(' ', '_').lower()}.pth"
        model_path = os.path.join(model_save_dir, model_name)
        save_gp_model(model, likelihood, model_path)

        # Save scaler
        from data.preprocessing import save_scalers
        scaler_name = f"scaler_{i+1}_{obj_name.replace(' ', '_').lower()}.pkl"
        scaler_path = os.path.join(model_save_dir, scaler_name)
        save_scalers([scalers[i]], scaler_path)

        # Cross-validation
        if ModelConfig.ENABLE_CROSS_VALIDATION:
            logger.info(f"Running LOOCV for {obj_name}")
            base_path = os.path.join(model_save_dir, f"objective_{i+1}_validation")
            cv_scores = loocv_gp_model(
                train_x_normalized,
                train_y_standardized,
                i,
                base_path,
                GPModel,
                scalers[i],
                make_plot=True
            )
            validation_results[obj_name] = cv_scores

            logger.info(f"CV Results for {obj_name}:")
            logger.info(f"  RMSE: {cv_scores['rmse']:.4f}")
            logger.info(f"  R²: {cv_scores['r2']:.4f}")
            logger.info(f"  Coverage: {cv_scores['coverage_95']:.4f}")

        models.append((model, likelihood))

    return models, scalers, validation_results


def main():
    """Main function to train GP models."""
    parser = argparse.ArgumentParser(description='Train GP models from experimental data')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Excel file with experimental data')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--project_name', type=str, default='gpytorch_models',
                       help='Project name for model subdirectory')

    args = parser.parse_args()

    logger.info("Starting GP model training")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Model directory: {args.model_dir}")

    # Load experimental data
    df = load_experimental_data(args.data_file)

    # Create model save path
    model_save_dir = Path(args.model_dir) / args.project_name

    # Train models
    models, scalers, validation_results = train_objective_models(df, str(model_save_dir))

    # Print summary
    print(f"\nTraining Summary:")
    print(f"Models trained: {len(models)}")
    print(f"Training samples: {len(df)}")
    print(f"Models saved to: {model_save_dir}")

    if validation_results:
        print(f"\nValidation Results:")
        for obj_name, scores in validation_results.items():
            print(f"{obj_name}:")
            print(f"  RMSE: {scores['rmse']:.4f}")
            print(f"  R²: {scores['r2']:.4f}")
            print(f"  Coverage: {scores['coverage_95']:.4f}")

    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()