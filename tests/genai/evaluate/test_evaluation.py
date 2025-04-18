from unittest import mock

import pandas as pd

from mlflow.genai.evaluation.base import evaluate
from mlflow.genai.scorers import BuiltInScorer


@mock.patch("mlflow.evaluate")
@mock.patch("mlflow.genai.evaluation.base.is_model_traced", return_value=True)
@mock.patch("mlflow.get_tracking_uri", return_value="databricks")
def test_evaluate_calls_mlflow_evaluate(tracking_mock, trace_mock, mock_mlflow_evaluate):
    # Setup test data
    test_data = pd.DataFrame({"inputs": ["test input"], "expectations": ["expected output"]})

    # Create a mock predict function
    def predict_fn(text):
        return f"{text} predicted output"

    # Create a mock scorer
    mock_scorer = mock.Mock(spec=BuiltInScorer)
    mock_scorer.update_evaluation_config.return_value = {
        "databricks-agents": {"metrics": ["metric1"]}
    }

    # Call the evaluate function
    with mock.patch(
        "mlflow.genai.evaluation.base._convert_to_legacy_eval_set", return_value=test_data
    ):
        evaluate(
            data=test_data,
            predict_fn=predict_fn,
            scorers=[mock_scorer],
            model_id="models:/my-model/1",
        )

    # Assert mlflow.evaluate was called with the right parameters
    mock_mlflow_evaluate.assert_called_once()
    call_args = mock_mlflow_evaluate.call_args[1]

    assert call_args["model"] == predict_fn
    assert call_args["model_type"] == "databricks-agent"
    assert call_args["evaluator_config"] == {"databricks-agents": {"metrics": ["metric1"]}}
