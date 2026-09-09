"""Analysis and reporting utilities for Bayesian optimization campaigns."""

from .database import (
    add_experiment_ids,
    validate_experimental_dataframe,
)
from .metadata import save_run_metadata
from .pareto import (
    auto_reference_point,
    compute_hypervolume_2d,
    is_non_dominated,
    summarize_campaign_progress,
)
from .plots import (
    plot_calibration_curve,
    plot_design_space_pairplot,
    plot_hypervolume_progress,
    plot_objective_tradeoff,
    plot_residuals_by_parameter,
)
from .reporting import (
    build_candidate_report,
    save_dataframe_report,
    save_json_report,
)

__all__ = [
    "add_experiment_ids",
    "auto_reference_point",
    "build_candidate_report",
    "compute_hypervolume_2d",
    "is_non_dominated",
    "plot_calibration_curve",
    "plot_design_space_pairplot",
    "plot_hypervolume_progress",
    "plot_objective_tradeoff",
    "plot_residuals_by_parameter",
    "save_dataframe_report",
    "save_json_report",
    "save_run_metadata",
    "summarize_campaign_progress",
    "validate_experimental_dataframe",
]
