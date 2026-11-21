import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.models.evaluation.evaluators.regressor import _get_regressor_metrics


@pytest.mark.parametrize(
    ("weights", "expected_mean", "expected_sum"),
    [
        (None, 14 / 3, 14),
        ([1, 1, 1], 14 / 3, 14),
        ([1, 3, 0], 3.5, 14),
        ([10, 30, 0], 3.5, 140),
        ([0.1, 0.3, 0], 3.5, 1.4),
    ],
)
def test_target_mean_uses_the_total_sample_weight(weights, expected_mean, expected_sum):
    metrics = _get_regressor_metrics([2, 4, 8], [3, 3, 7], weights)

    assert metrics["mean_on_target"] == pytest.approx(expected_mean)
    assert metrics["sum_on_target"] == pytest.approx(expected_sum)
    assert metrics["example_count"] == 3


def test_target_mean_matches_frequency_weighted_observations():
    target = np.array([2, 4, 8])
    prediction = np.array([3, 3, 7])
    frequencies = np.array([1, 3, 2])
    repeated = _get_regressor_metrics(
        np.repeat(target, frequencies), np.repeat(prediction, frequencies), None
    )

    weighted = _get_regressor_metrics(target, prediction, frequencies)

    assert weighted["mean_on_target"] == pytest.approx(repeated["mean_on_target"])
    assert weighted["sum_on_target"] == pytest.approx(repeated["sum_on_target"])


def test_evaluate_logs_the_weighted_target_mean():
    data = pd.DataFrame({"target": [2, 4, 8], "prediction": [3, 3, 7]})

    with mlflow.start_run() as run:
        result = mlflow.models.evaluate(
            data=data,
            targets="target",
            predictions="prediction",
            model_type="regressor",
            evaluators="regressor",
            evaluator_config={"sample_weights": [10, 30, 0]},
        )

    assert result.metrics["mean_on_target"] == pytest.approx(3.5)
    logged = mlflow.get_run(run.info.run_id).data.metrics
    assert logged["mean_on_target"] == pytest.approx(3.5)
    assert logged["sum_on_target"] == pytest.approx(140)
