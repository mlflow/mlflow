import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.models.evaluation.evaluators.regressor import _get_regressor_metrics


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (np.uint8, [200, 210]),
        (np.int8, [100, 110]),
        (np.uint16, [40000, 40010]),
        (np.int16, [30000, 30010]),
        (np.float16, [40000, 40000]),
        (np.float32, [1e8, 1, 1]),
        (np.float64, [200, 210]),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_target_sum_uses_sufficient_precision_before_multiplication(dtype, values, weighted):
    targets = np.array(values, dtype=dtype)
    weights = np.arange(2, len(targets) + 2, dtype=dtype) if weighted else None
    original_targets = targets.copy()
    expected = sum(
        float(value) * (float(weights[index]) if weighted else 1)
        for index, value in enumerate(targets)
    )

    metrics = _get_regressor_metrics(targets, targets, weights)

    assert float(metrics["sum_on_target"]) == expected
    assert np.isfinite(metrics["sum_on_target"])
    if not weighted:
        assert float(metrics["mean_on_target"]) == expected / len(targets)
    np.testing.assert_array_equal(targets, original_targets)
    if weighted:
        np.testing.assert_array_equal(weights, np.arange(2, len(targets) + 2, dtype=dtype))


@pytest.mark.parametrize("container", [list, pd.Series])
def test_target_sum_accepts_non_numpy_input_containers(container):
    metrics = _get_regressor_metrics(container([200, 210]), [200, 210], container([2, 3]))

    assert metrics["sum_on_target"] == 1030


@pytest.mark.parametrize("weighted", [False, True])
def test_regression_evaluation_persists_the_integer_target_sum(weighted):
    targets = np.array([200, 210], dtype=np.uint8)
    data = pd.DataFrame({"target": targets, "prediction": targets})
    config = {"sample_weights": np.array([2, 3], dtype=np.uint8)} if weighted else {}

    with mlflow.start_run() as run:
        result = mlflow.models.evaluate(
            data=data,
            targets="target",
            predictions="prediction",
            model_type="regressor",
            evaluators="regressor",
            evaluator_config=config,
        )

    expected = 1030 if weighted else 410
    assert result.metrics["sum_on_target"] == expected
    assert mlflow.get_run(run.info.run_id).data.metrics["sum_on_target"] == expected
