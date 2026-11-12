import pandas as pd
import pytest

import mlflow
from mlflow.models.evaluation.default_evaluator import _extract_output_and_other_columns


@pytest.mark.parametrize(
    "values", [[1.0, 2.0, 3.0], ["first", "second", "third"], [[1, 2], [3], []]]
)
@pytest.mark.parametrize("column", [None, "answer"])
def test_record_predictions_preserve_one_value_per_input_row(values, column):
    records = [{"answer": value} for value in values]

    output, other_columns, inferred_column = _extract_output_and_other_columns(records, column)

    pd.testing.assert_series_equal(output, pd.Series(values, name="answer"))
    assert inferred_column == "answer"
    assert other_columns is None or other_columns.empty


def test_a_single_prediction_record_is_not_expanded_into_multiple_rows():
    output, other_columns, inferred_column = _extract_output_and_other_columns(
        [{"answer": [1, 2]}], None
    )

    pd.testing.assert_series_equal(output, pd.Series([[1, 2]], name="answer"))
    assert other_columns is None
    assert inferred_column == "answer"


def test_column_oriented_predictions_keep_their_existing_behavior():
    output, other_columns, inferred_column = _extract_output_and_other_columns(
        {"answer": [1, 2, 3]}, None
    )

    pd.testing.assert_series_equal(output, pd.Series([1, 2, 3], name="answer"))
    assert other_columns is None
    assert inferred_column == "answer"


@pytest.mark.parametrize("predictions", [None, "answer"])
def test_regression_evaluation_logs_all_automatically_selected_record_values(predictions):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0], "target": [1.5, 1.0, 3.0, 10.0]})

    def predict(frame):
        return [{"answer": value} for value in frame["x"]]

    with mlflow.start_run() as run:
        result = mlflow.models.evaluate(
            model=predict,
            data=data,
            targets="target",
            predictions=predictions,
            model_type="regressor",
            evaluators="regressor",
        )

    expected_mae = (0.5 + 1.0 + 1.0 + 2.0) / 4
    assert result.metrics["example_count"] == 4
    assert result.metrics["mean_absolute_error"] == expected_mae
    assert mlflow.get_run(run.info.run_id).data.metrics["mean_absolute_error"] == expected_mae
