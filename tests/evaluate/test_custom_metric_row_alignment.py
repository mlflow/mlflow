import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.metrics import MetricValue, make_metric


@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize(
    ("input_index", "output_index"),
    [
        pytest.param([0, 1, 2], [0, 1, 2], id="matching"),
        pytest.param([10, 20, 30], [0, 1, 2], id="disjoint"),
        pytest.param([0, 1, 2], [2, 1, 0], id="reordered"),
        pytest.param([5, 5, 9], [2, 2, 1], id="duplicate-labels"),
        pytest.param(
            pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)]),
            ["x", "y", "z"],
            id="multi-index",
        ),
        pytest.param(["c", "b", "a"], ["a", "b", "c"], id="string-labels"),
    ],
)
def test_custom_metric_combines_input_and_output_columns_by_row(
    input_index, output_index, multi_output
):
    data = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0], "target": [2.0, 1.0, 5.0], "weight": [1.0, 10.0, 100.0]},
        index=input_index,
    )
    original_data = data.copy(deep=True)
    model_outputs = []
    metric_indexes = []

    def model(frame):
        prediction = pd.Series(2 * frame["feature"].to_numpy() + 1, index=output_index)
        output = (
            pd.DataFrame({"prediction": prediction, "scale": [0.5, 1.0, 2.0]})
            if multi_output
            else prediction
        )
        model_outputs.append((output, output.copy(deep=True)))
        return output

    if multi_output:

        def row_metric(predictions, targets, weight, scale):
            assert len(predictions) == len(targets) == len(weight) == len(scale)
            metric_indexes.append((predictions.index, targets.index, weight.index, scale.index))
            scores = (predictions - targets).abs() * weight / scale
            return MetricValue(scores=scores.tolist())

        expected = [2.0, 40.0, 100.0]
    else:

        def row_metric(predictions, targets, weight):
            assert len(predictions) == len(targets) == len(weight)
            metric_indexes.append((predictions.index, targets.index, weight.index))
            scores = (predictions - targets).abs() * weight
            return MetricValue(scores=scores.tolist())

        expected = [1.0, 40.0, 200.0]

    with mlflow.start_run():
        result = mlflow.models.evaluate(
            model,
            data,
            targets="target",
            predictions="prediction" if multi_output else None,
            model_type=None if multi_output else "regressor",
            evaluators="default",
            evaluator_config={"log_model_explainability": False},
            extra_metrics=[
                make_metric(eval_fn=row_metric, greater_is_better=False, name="weighted")
            ],
        )

    assert result.metrics["weighted/mean"] == pytest.approx(np.mean(expected))
    stored = mlflow.MlflowClient().get_run(result.run_id).data.metrics
    assert stored["weighted/mean"] == pytest.approx(np.mean(expected))
    table = pd.DataFrame(**result.artifacts["eval_results_table"].content)
    assert table["weighted/score"].tolist() == expected
    assert table["feature"].tolist() == [1.0, 2.0, 3.0]
    if not multi_output:
        assert result.metrics["mean_absolute_error"] == pytest.approx(7 / 3)
    # The validation pass and full evaluation both use the input's row labels consistently.
    for indexes in metric_indexes:
        for index in indexes:
            pd.testing.assert_index_equal(index, data.index[: len(index)])
    pd.testing.assert_frame_equal(data, original_data)
    for output, original in model_outputs:
        if multi_output:
            pd.testing.assert_frame_equal(output, original)
        else:
            pd.testing.assert_series_equal(output, original)


@pytest.mark.parametrize("multi_output", [False, True])
def test_metric_alignment_preserves_nullable_integer_values(multi_output):
    data = pd.DataFrame({"feature": [1, 2]}, index=[10, 20])
    values = pd.Series([2**63 - 1, pd.NA], dtype="Int64", index=[30, 10])

    if multi_output:

        def model(frame):
            return pd.DataFrame({"prediction": [1.0, 2.0], "auxiliary": values})

        def row_metric(predictions, auxiliary):
            assert auxiliary.dtype == "Int64"
            assert auxiliary.iloc[0] == 2**63 - 1
            if len(auxiliary) == 2:
                assert auxiliary.isna().iloc[1]
            return MetricValue(scores=[0.0] * len(predictions))

    else:

        def model(frame):
            return values.copy()

        def row_metric(predictions):
            assert predictions.dtype == "Int64"
            assert predictions.iloc[0] == 2**63 - 1
            if len(predictions) == 2:
                assert predictions.isna().iloc[1]
            return MetricValue(scores=[0.0] * len(predictions))

    with mlflow.start_run():
        result = mlflow.models.evaluate(
            model,
            data,
            predictions="prediction" if multi_output else None,
            evaluators="default",
            evaluator_config={"log_model_explainability": False},
            extra_metrics=[
                make_metric(eval_fn=row_metric, greater_is_better=False, name="nullable")
            ],
        )

    assert result.metrics["nullable/mean"] == 0.0


def test_legacy_dataframe_metric_keeps_its_existing_index_contract():
    data = pd.DataFrame({"feature": [1.0, 2.0], "target": [2.0, 3.0]}, index=[10, 20])
    output_index = pd.Index(["first", "second"])

    def model(frame):
        return pd.Series(2 * frame["feature"].to_numpy() + 1, index=output_index)

    def legacy_metric(eval_df, builtin_metrics):
        pd.testing.assert_index_equal(eval_df.index, output_index[: len(eval_df)])
        assert isinstance(builtin_metrics, dict)
        return MetricValue(scores=(eval_df["prediction"] - eval_df["target"]).tolist())

    with mlflow.start_run():
        result = mlflow.models.evaluate(
            model,
            data,
            targets="target",
            model_type="regressor",
            evaluators="default",
            evaluator_config={"log_model_explainability": False},
            extra_metrics=[
                make_metric(eval_fn=legacy_metric, greater_is_better=False, name="legacy")
            ],
        )

    assert result.metrics["legacy/mean"] == 1.5


@pytest.mark.parametrize(
    ("rows", "message"),
    [(0, "Found array with 0 sample"), (2, "inconsistent numbers of samples")],
)
def test_regression_evaluation_rejects_invalid_row_counts(rows, message):
    data = pd.DataFrame({"feature": np.arange(rows), "target": np.arange(rows)})

    def model(frame):
        return np.zeros(max(0, len(frame) - 1))

    def row_metric(predictions, targets):
        return MetricValue(scores=(predictions - targets).tolist())

    with mlflow.start_run(), pytest.raises(ValueError, match=message):
        mlflow.models.evaluate(
            model,
            data,
            targets="target",
            model_type="regressor",
            evaluators="default",
            evaluator_config={"log_model_explainability": False},
            extra_metrics=[make_metric(eval_fn=row_metric, greater_is_better=False, name="rows")],
        )
