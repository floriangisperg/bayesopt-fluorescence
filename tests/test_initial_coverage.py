"""Feasibility and held-out geometric coverage of the initial design."""

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from acquisition.utils import generate_initial_design
from config import ExperimentConfig
from data.transformation import ParameterTransformer, build_transformer


@pytest.mark.parametrize("n_samples,select", [(1, True), (12, True), (12, False)])
def test_coverage_is_feasible_bounded_unique_and_reproducible(n_samples, select):
    transformer = build_transformer(ExperimentConfig)
    kwargs = dict(
        n_samples=n_samples, bounds=transformer.get_physical_bounds(as_tensor=True),
        transformer=transformer, design_strategy="feasible_coverage", seed=17,
        n_candidates=3, use_maximin=select,
    )
    samples = generate_initial_design(**kwargs)
    torch.testing.assert_close(samples, generate_initial_design(**kwargs), rtol=0, atol=0)
    assert samples.shape == (n_samples, 5)
    assert len(torch.unique(samples, dim=0)) == n_samples
    assert (samples[:, 2] * samples[:, 4] >= 8).all()
    bounds = kwargs["bounds"]
    assert ((samples >= bounds[0]) & (samples <= bounds[1])).all()


def test_custom_order_bounds_and_user_transforms():
    transformer = ParameterTransformer(
        ["urea", "dilution"], np.array([[0., 6.], [2., 40.]]),
        {"urea": {"user_space": "linear", "model_space": "linear"},
         "dilution": {"user_space": "log", "model_space": "1/x"}},
    )
    samples = generate_initial_design(
        12, transformer.get_physical_bounds(as_tensor=True), transformer,
        design_strategy="feasible_coverage", dilution_idx=1, urea_idx=0,
        solubilization_urea=13., n_candidates=3,
    )
    assert (samples[:, 0] * samples[:, 1] >= 13.).all()
    assert (samples[:, 1] >= 13. / 6.).all()


def test_empty_feasible_region_fails_explicitly():
    transformer = build_transformer(ExperimentConfig)
    with pytest.raises(ValueError, match="no positive-volume feasible region"):
        generate_initial_design(
            12, transformer.get_physical_bounds(as_tensor=True), transformer,
            design_strategy="feasible_coverage", solubilization_urea=300.,
        )


def test_cli_defaults_to_coverage_and_exports_feasible_plan(tmp_path, monkeypatch):
    import sys

    import pandas as pd

    import generate_initial_design as cli

    strategies = []

    def record_strategy(**kwargs):
        strategies.append(kwargs["design_strategy"])
        return generate_initial_design(**kwargs)

    monkeypatch.setattr(cli, "generate_initial_design", record_strategy)
    monkeypatch.setattr(sys, "argv", [
        "generate_initial_design.py", "--n_samples", "4", "--n_candidates", "2",
        "--output_dir", str(tmp_path),
    ])
    cli.main()
    assert strategies == ["feasible_coverage"]
    frame = pd.read_excel(tmp_path / "initial_design_experimental_plan.xlsx")
    assert len(frame) == 4
    assert (frame["Urea Refolding [M]"] >= 0).all()
    assert frame[ExperimentConfig.OBJECTIVE_NAMES].isna().all().all()


def test_coverage_improves_on_conditional_lhd_on_independent_points():
    """Evaluate gaps on a separate pool, not the points used for selection.

    This regression measures the default five-dimensional geometry, not a
    universal claim about all bounds, dimensions, or objective functions.
    """
    transformer = build_transformer(ExperimentConfig)
    unit = qmc.Sobol(d=5, scramble=True, seed=909).random_base2(14)
    physical = transformer.unit_to_physical_user(unit)
    evaluation = unit[physical[:, 2] * physical[:, 4] >= 8.]
    scores = {"constrained_lhd": [], "feasible_coverage": []}
    for seed in [0, 1, 2, 3, 42]:
        for strategy in scores:
            samples = generate_initial_design(
                20, transformer.get_physical_bounds(as_tensor=True), transformer,
                design_strategy=strategy, seed=seed, n_candidates=30,
            )
            distances = cdist(evaluation, transformer.physical_to_unit_user(samples)).min(axis=1)
            scores[strategy].append([distances.max(), distances.mean()])
    old = np.mean(scores["constrained_lhd"], axis=0)
    new = np.mean(scores["feasible_coverage"], axis=0)
    assert new[0] < 0.95 * old[0]
    assert new[1] < 0.95 * old[1]
