import numpy as np
import pandas as pd
import pytest

from analysis.database import add_experiment_ids
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


def test_experiment_ids_unique_across_batches_and_preserved():
    batch = pd.DataFrame({"x": [1, 2]})
    first = add_experiment_ids(batch)
    second = add_experiment_ids(batch)
    assert pd.concat([first, second])["Experiment ID"].is_unique
    pd.testing.assert_frame_equal(add_experiment_ids(first), first)
    assert "Experiment ID" not in batch


def test_missing_ids_filled_and_duplicate_ids_rejected():
    frame = pd.DataFrame({"Experiment ID": ["old", None, ""]})
    result = add_experiment_ids(frame)
    assert result["Experiment ID"].iloc[0] == "old"
    assert result["Experiment ID"].notna().all()
    assert result["Experiment ID"].is_unique
    with pytest.raises(ValueError, match="Duplicate Experiment ID"):
        add_experiment_ids(pd.DataFrame({"Experiment ID": ["old", "old"]}))
