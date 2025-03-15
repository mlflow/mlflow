from unittest.mock import MagicMock, patch

import pytest

from mlflow.dspy.util import log_dspy_dataset, log_dspy_module_state_params


@patch("mlflow.dspy.util.mlflow")
def test_log_dspy_module_state_params(mock_mlflow):
    program = MagicMock()
    program.dump_state.return_value = {
        "param1": "value1",
        "param2": "value2",
        "metadata": "meta",
        "lm": "lm",
        "traces": "traces",
        "train": "train",
    }

    log_dspy_module_state_params(program)

    expected_params = {"param1": "value1", "param2": "value2"}
    mock_mlflow.log_params.assert_called_once_with(expected_params)


@patch("mlflow.dspy.util.mlflow")
def test_log_dataset(mock_mlflow):
    example1 = {"feature1": "value1", "feature2": "value2"}
    example2 = {"feature1": "value3", "feature2": "value4"}
    dataset = [example1, example2]

    log_dspy_dataset(dataset, "dataset.json")

    expected_result = {"feature1": ["value1", "value3"], "feature2": ["value2", "value4"]}
    mock_mlflow.log_table.assert_called_once_with(expected_result, "dataset.json")


if __name__ == "__main__":
    pytest.main()
