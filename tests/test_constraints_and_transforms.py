import numpy as np

from config import ConstraintConfig, ExperimentConfig
from constraints.urea_dilution import check_urea_constraint, iterative_urea_adjustment
from data.transformation import build_transformer


def test_parameter_transform_round_trip_with_reciprocal_dilution():
    transformer = build_transformer(ExperimentConfig)
    physical = np.array([
        [10.0, 1.0, 5.0, 9.5, 2.0],
        [20.0, 2.0, 30.0, 10.5, 5.5],
    ])

    unit = transformer.physical_to_unit_model(physical)
    restored = transformer.unit_to_physical_model(unit)

    np.testing.assert_allclose(restored, physical, rtol=1e-10, atol=1e-10)


def test_urea_repair_returns_feasible_sample():
    sample = np.array([5.0, 0.5, 2.0, 9.0, 1.0])

    repaired = iterative_urea_adjustment(sample)

    assert check_urea_constraint(repaired)
    assert repaired[ConstraintConfig.DILUTION_FACTOR_IDX] <= ConstraintConfig.MAX_DILUTION_FACTOR
    assert repaired[ConstraintConfig.FINAL_UREA_IDX] <= ConstraintConfig.MAX_FINAL_UREA
