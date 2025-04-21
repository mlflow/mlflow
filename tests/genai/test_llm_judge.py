from unittest import mock
from unittest.mock import patch

import pandas as pd

import mlflow
from mlflow.genai.evaluation.utils import _convert_scorer_to_legacy_metric
from mlflow.genai.scorers.llm_judge import llm_judge_scorer
from mlflow.metrics.genai import model_utils


def mock_init_auth(config_instance):
    config_instance.host = "https://databricks.com/"
    config_instance._header_factory = lambda: {}


def test_llm_judge_scorer_creation():
    """Test the basic creation of a scorer."""
    scorer = llm_judge_scorer(
        name="correctness",
        prompt_template=(
            "Does this answer the question correctly?\nQuestion: {inputs}\nAnswer: {outputs}"
        ),
        judge="openai:/gpt-4o",
        result_type=bool,
    )

    assert scorer.name == "correctness"

    with mock.patch.object(
        model_utils, "score_model_on_payload", return_value="true"
    ) as mock_score:
        test_input = "What is Python?"
        test_output = "Python is a programming language."
        scorer(inputs=pd.Series([test_input]), outputs=pd.Series([test_output]))

        # Verify the model was called with formatted prompt containing our inputs/outputs
        assert mock_score.call_count == 1
        assert f"Question: {test_input}" in mock_score.call_args[0][1]
        assert f"Answer: {test_output}" in mock_score.call_args[0][1]


def test_llm_judge_scorer_with_mlflow_evaluate():
    """Test that llm_judge_scorer works with mlflow.evaluate."""
    eval_data = pd.DataFrame(
        {
            "request": ["What is Python?", "What is TensorFlow?"],
            "response": ["Python is a programming language.", "I don't know."],
            "expected_response": [
                "Python is a programming language.",
                "TensorFlow is a machine learning framework.",
            ],
        }
    )

    quality_scorer = _convert_scorer_to_legacy_metric(
        llm_judge_scorer(
            name="quality",
            prompt_template="Rate the quality from 1-5:\nQuestion: {inputs}\nAnswer: {outputs}",
            judge="openai:/gpt-4o",
            result_type=int,
        )
    )

    with mock.patch.object(model_utils, "score_model_on_payload", return_value="4") as mock_score:
        with patch("databricks.sdk.config.Config.init_auth", new=mock_init_auth):
            results = mlflow.evaluate(
                data=eval_data,
                model_type="databricks-agent",
                extra_metrics=[quality_scorer],
            )
            assert 0 < mock_score.call_count
            table_metric_names = results.tables["eval_results"].keys()
            assert any("quality" in metric for metric in table_metric_names)
            assert all("quality/error" not in metric for metric in table_metric_names)
            assert results.tables["eval_results"]["metric/quality/value"].mean() == 4
