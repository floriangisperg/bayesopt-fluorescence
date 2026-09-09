#!/usr/bin/env python3
"""
Train Gaussian Process models from experimental data.

This script trains single-task GP models for each objective using
experimental data from previous iterations.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from analysis import plot_calibration_curve, plot_residuals_by_parameter, save_run_metadata
from analysis.metadata import file_sha256
from config import ExperimentConfig, LoggingConfig, ModelConfig
from data.preprocessing import prepare_data, save_scalers, validate_experiment_data
from data.transformation import ParameterTransformer, build_transformer
from models import GPModel, fit_gp_model, loocv_gp_model, save_gp_model


def extract_ard_lengthscales(model, parameter_names):
    """Extract ARD lengthscales from the fitted kernel.

    One lengthscale per input dimension; shorter lengthscales indicate the
    model relies more on that parameter. Relevance is shown as the inverse
    for easier ranking.
    """
    base_kernel = model.covar_module.base_kernel
    lengthscale = base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
    return pd.DataFrame({
        "Parameter": parameter_names,
        "ARD Lengthscale": lengthscale,
        "Relative Relevance": 1.0 / lengthscale,
    }).sort_values("Relative Relevance", ascending=False)

# Set up logging
logging.basicConfig(
    level=LoggingConfig.LOG_LEVEL,
    format=LoggingConfig.LOG_FORMAT
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

    # Objectives must be filled in, and duplicates deserve a warning
    validate_experiment_data(df)

    logger.info(f"Loaded {len(df)} experimental samples")
    return df


def train_objective_models(df: pd.DataFrame, transformer: ParameterTransformer,  model_save_dir: str):
    """Train GP models for each objective.

    Args:
        df: Experimental data.
        transformer: ParameterTransformer object.
        model_save_dir: Directory to save trained models.
    """
    # Prepare training data
    parameter_names = ExperimentConfig.PARAMETER_NAMES
    objective_names = ExperimentConfig.OBJECTIVE_NAMES

    logger.info("Preparing training data...")
    x_raw = df[parameter_names].to_numpy()
    y_raw = df[objective_names].to_numpy()

    train_x_normalized, train_y_standardized, scalers = prepare_data(x_raw, y_raw, transformer)

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
        model, likelihood = fit_gp_model(
            train_x=train_x_normalized,
            train_y=train_y_single,
            model_class=GPModel,
            noise=ModelConfig.INITIAL_NOISE_LEVEL
        )

        # Save model and likelihood
        model_name = f"model_{i+1}_{obj_name.replace(' ', '_').lower()}.pth"
        model_path = os.path.join(model_save_dir, model_name)
        save_gp_model(model, likelihood, model_path)

        # Export ARD lengthscales as a per-objective parameter relevance table
        lengthscale_df = extract_ard_lengthscales(model, parameter_names)
        lengthscale_df.insert(0, "Objective", obj_name)
        lengthscale_path = os.path.join(
            model_save_dir,
            f"objective_{i+1}_{obj_name.replace(' ', '_').lower()}_lengthscales.xlsx",
        )
        lengthscale_df.to_excel(lengthscale_path, index=False)

        # Save scaler
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
                noise=ModelConfig.INITIAL_NOISE_LEVEL,
                make_plot=True
            )
            validation_results[obj_name] = cv_scores

            # Export per-sample validation table and diagnostic plots
            validation_table = pd.DataFrame({
                "Actual Standardized": cv_scores["actual_standardized"],
                "Predicted Standardized": cv_scores["predictions_standardized"],
                "Uncertainty Standardized": cv_scores["uncertainties_standardized"],
                "Actual Original": cv_scores["actual_original"],
                "Predicted Original": cv_scores["predictions_original"],
                "Uncertainty Original": cv_scores["uncertainties_original"],
                "Residual Original": cv_scores["residuals_original"],
            })
            validation_table.to_excel(f"{base_path}_validation_table.xlsx", index=False)
            plot_calibration_curve(
                validation_table["Actual Original"].to_numpy(),
                validation_table["Predicted Original"].to_numpy(),
                validation_table["Uncertainty Original"].to_numpy(),
                f"{base_path}_calibration.png",
            )
            plot_residuals_by_parameter(
                df[parameter_names].reset_index(drop=True),
                validation_table["Residual Original"].to_numpy(),
                f"{base_path}_residuals_by_parameter.png",
                objective_name=obj_name,
            )

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
    parser.add_argument('--model_dir', type=str, default='trained_models',
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

    transformer = build_transformer(ExperimentConfig)

    # Train models
    models, scalers, validation_results = train_objective_models(df, transformer, str(model_save_dir))

    # Print summary
    print("\nTraining Summary:")
    print(f"Models trained: {len(models)}")
    print(f"Training samples: {len(df)}")
    print(f"Models saved to: {model_save_dir}")

    if validation_results:
        print("\nValidation Results:")
        for obj_name, scores in validation_results.items():
            print(f"{obj_name}:")
            print(f"  RMSE: {scores['rmse']:.4f}")
            print(f"  R²: {scores['r2']:.4f}")
            print(f"  Coverage: {scores['coverage_95']:.4f}")

    # Run metadata for traceability: software versions, configuration,
    # and the exact data file (with content hash) behind this training run
    save_run_metadata(
        os.path.join(str(model_save_dir), "training_metadata.json"),
        command="train_models.py",
        config={
            "parameter_names": ExperimentConfig.PARAMETER_NAMES,
            "objective_names": ExperimentConfig.OBJECTIVE_NAMES,
            "objective_directions": ExperimentConfig.OBJECTIVE_DIRECTIONS,
            "kernel_nu": ModelConfig.KERNEL_NU,
            "initial_noise_level": ModelConfig.INITIAL_NOISE_LEVEL,
            "cross_validation_enabled": ModelConfig.ENABLE_CROSS_VALIDATION,
        },
        inputs={
            "data_file": args.data_file,
            "data_file_sha256": file_sha256(args.data_file),
        },
        extra={
            "n_samples": len(df),
            "validation_summary": {
                name: {key: scores[key] for key in ("rmse", "r2", "coverage_95")}
                for name, scores in validation_results.items()
            },
        },
    )

    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()
