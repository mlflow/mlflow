import pandas as pd
import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.entities import EvalItem, InputDatasetColumn


@pytest.mark.parametrize(
    ("row_data", "expected_inputs", "expected_outputs", "expected_request_id", "check_request_id"),
    [
        (
            {
                InputDatasetColumn.INPUTS: {"prompt": "test"},
                InputDatasetColumn.OUTPUTS: {"response": "output"},
                InputDatasetColumn.REQUEST_ID: "req_123",
            },
            {"prompt": "test"},
            {"response": "output"},
            "req_123",
            True,
        ),
        (
            {
                InputDatasetColumn.INPUTS: '"valid json string"',
                InputDatasetColumn.OUTPUTS: "simple output",
            },
            "valid json string",
            "simple output",
            None,
            False,
        ),
    ],
)
def test_eval_item_from_dataset_row_valid(
    row_data, expected_inputs, expected_outputs, expected_request_id, check_request_id
):
    row = pd.Series(row_data)
    eval_item = EvalItem.from_dataset_row(row)

    assert eval_item.inputs == expected_inputs
    assert eval_item.outputs == expected_outputs

    if check_request_id:
        assert eval_item.request_id == expected_request_id
    else:
        # When request_id is not provided, it should be auto-generated
        assert eval_item.request_id is not None
        assert isinstance(eval_item.request_id, str)
        assert len(eval_item.request_id) > 0


@pytest.mark.parametrize(
    ("row_data", "error_match"),
    [
        (
            {
                InputDatasetColumn.INPUTS: None,
                InputDatasetColumn.OUTPUTS: {"response": "output"},
            },
            "Dataset row must contain 'inputs' or 'request' key",
        ),
        (
            {
                InputDatasetColumn.OUTPUTS: {"response": "output"},
            },
            "Dataset row must contain 'inputs' or 'request' key",
        ),
        (
            {
                InputDatasetColumn.INPUTS: "invalid json string",
                InputDatasetColumn.OUTPUTS: "simple output",
            },
            "Failed to parse inputs as JSON",
        ),
    ],
)
def test_eval_item_from_dataset_row_invalid(row_data, error_match):
    row = pd.Series(row_data)

    with pytest.raises(MlflowException, match=error_match):
        EvalItem.from_dataset_row(row)
