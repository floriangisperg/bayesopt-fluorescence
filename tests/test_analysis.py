import numpy as np
import pandas as pd

from analysis.pareto import (
    auto_reference_point,
    compute_hypervolume_2d,
    is_non_dominated,
    summarize_campaign_progress,
)


def test_non_dominated_and_hypervolume_for_tradeoff_front():
    values = np.array([
        [1.0, 1.0],
        [2.0, 5.0],
        [5.0, 2.0],
        [3.0, 3.0],
    ])

    mask = is_non_dominated(values, ["maximize", "maximize"])

    assert mask.tolist() == [False, True, True, True]
    assert compute_hypervolume_2d(values, [0.0, 0.0], ["maximize", "maximize"]) == 17.0


def test_campaign_progress_uses_cumulative_iterations():
    df = pd.DataFrame({
        "Iteration": [0, 0, 1],
        "obj1": [1.0, 2.0, 3.0],
        "obj2": [3.0, 2.0, 1.0],
    })

    progress = summarize_campaign_progress(
        df,
        ["obj1", "obj2"],
        ["maximize", "maximize"],
        [0.0, 0.0],
    )

    assert progress["Iteration"].tolist() == [0, 1]
    assert progress["Completed Experiments"].tolist() == [2, 3]
    assert progress["Hypervolume"].iloc[-1] >= progress["Hypervolume"].iloc[0]


def test_auto_reference_point_is_below_observations():
    values = np.array([[2.0, 4.0], [6.0, 8.0]])

    ref = auto_reference_point(values, ["maximize", "maximize"], margin_fraction=0.25)

    assert np.all(ref < values.min(axis=0))
