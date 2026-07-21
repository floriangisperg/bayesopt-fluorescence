#!/usr/bin/env python3
"""
Train Gaussian Process models from experimental data.

This script trains single-task GP models for each objective using
experimental data from previous iterations.
"""

import os
import logging
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from config import ExperimentConfig, ModelConfig, PathConfig
from data.preprocessing import prepare_data
from data.transformation import ParameterTransformer, build_transformer
from models import GPModel, fit_gp_model, save_gp_model, loocv_gp_model, plot_training_loss
from analysis import (
    plot_calibration_curve,
    plot_residuals_by_parameter,
    save_run_metadata,
    validate_experimental_dataframe,
)
from analysis.metadata import file_sha256

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

    validate_experimental_dataframe(
        df,
        ExperimentConfig.PARAMETER_NAMES,
        ExperimentConfig.OBJECTIVE_NAMES,
        require_objectives=True,
    )
    complete_mask = df[ExperimentConfig.OBJECTIVE_NAMES].notna().all(axis=1)
    if not complete_mask.all():
        dropped = int((~complete_mask).sum())
        logger.warning(f"Dropping {dropped} rows without complete objective values")
        df = df.loc[complete_mask].copy()

    logger.info(f"Loaded {len(df)} experimental samples")
    return df


def extract_ard_lengthscales(model, parameter_names):
    """Extract ARD lengthscales from the fitted kernel."""
    base_kernel = model.covar_module.base_kernel
    lengthscale = base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
    return pd.DataFrame({
        "Parameter": parameter_names,
        "ARD Lengthscale": lengthscale,
        "Relative Relevance": 1.0 / lengthscale,
    }).sort_values("Relative Relevance", ascending=False)


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
        plot_training_loss(
            losses,
            os.path.join(model_save_dir, f"objective_{i+1}_{obj_name.replace(' ', '_').lower()}"),
            make_plot=True,
        )

        lengthscale_df = extract_ard_lengthscales(model, parameter_names)
        lengthscale_df.insert(0, "Objective", obj_name)
        lengthscale_path = os.path.join(
            model_save_dir,
            f"objective_{i+1}_{obj_name.replace(' ', '_').lower()}_lengthscales.xlsx",
        )
        lengthscale_df.to_excel(lengthscale_path, index=False)

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
            validation_table = pd.DataFrame({
                "Actual Standardized": cv_scores["actual_standardized"],
                "Predicted Standardized": cv_scores["predictions_standardized"],
                "Uncertainty Standardized": cv_scores["uncertainties_standardized"],
                "Actual Original": cv_scores["actual_original"],
                "Predicted Original": cv_scores["predictions_original"],
                "Uncertainty Original": cv_scores["uncertainties_original"],
                "Residual Original": cv_scores["residuals_original"],
            })
            validation_table = pd.concat(
                [df[parameter_names].reset_index(drop=True), validation_table],
                axis=1,
            )
            validation_table.to_excel(f"{base_path}_details.xlsx", index=False)
            plot_calibration_curve(
                validation_table["Actual Original"].to_numpy(),
                validation_table["Predicted Original"].to_numpy(),
                validation_table["Uncertainty Original"].to_numpy(),
                f"{base_path}_calibration.png",
                title=f"Predictive Calibration - {obj_name}",
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

    validation_summary = {
        obj: {
            key: float(value)
            for key, value in scores.items()
            if not isinstance(value, list)
        }
        for obj, scores in validation_results.items()
    }
    if validation_summary:
        with open(os.path.join(model_save_dir, "validation_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(validation_summary, handle, indent=2)
        pd.DataFrame.from_dict(validation_summary, orient="index").to_excel(
            os.path.join(model_save_dir, "validation_summary.xlsx")
        )

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

    transformer = build_transformer(ExperimentConfig)

    # Train models
    models, scalers, validation_results = train_objective_models(df, transformer, str(model_save_dir))
    save_run_metadata(
        model_save_dir / "training_metadata.json",
        command="train_models.py",
        config={
            "parameter_names": ExperimentConfig.PARAMETER_NAMES,
            "objective_names": ExperimentConfig.OBJECTIVE_NAMES,
            "objective_directions": ExperimentConfig.OBJECTIVE_DIRECTIONS,
            "num_training_iterations": ModelConfig.NUM_TRAINING_ITERATIONS,
            "learning_rate": ModelConfig.LEARNING_RATE,
            "initial_noise_level": ModelConfig.INITIAL_NOISE_LEVEL,
            "kernel_nu": ModelConfig.KERNEL_NU,
            "enable_cross_validation": ModelConfig.ENABLE_CROSS_VALIDATION,
        },
        inputs={
            "data_file": args.data_file,
            "data_file_sha256": file_sha256(args.data_file),
        },
        extra={
            "training_samples": len(df),
            "model_count": len(models),
        },
    )

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