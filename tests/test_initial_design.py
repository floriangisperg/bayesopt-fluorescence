from config import ConstraintConfig, ExperimentConfig
from acquisition.utils import generate_initial_design
from constraints.urea_dilution import urea_constraint_callable
from data.transformation import build_transformer


def test_initial_design_respects_bounds_and_urea_constraint():
    transformer = build_transformer(ExperimentConfig)
    bounds = transformer.get_physical_bounds(as_tensor=True)

    samples = generate_initial_design(
        n_samples=6,
        bounds=bounds,
        transformer=transformer,
        seed=123,
        n_candidates=3,
        use_maximin=True,
        constraint_callable=urea_constraint_callable,
        solubilization_urea=ConstraintConfig.SOLUBILIZATION_UREA,
    )

    assert samples.shape == (6, len(ExperimentConfig.PARAMETER_NAMES))
    assert ((samples >= bounds[0]) & (samples <= bounds[1])).all()
    assert (urea_constraint_callable(samples) > 0).all()
