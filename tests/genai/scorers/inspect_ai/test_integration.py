# End-to-end runs of Inspect's model_graded_fact through the MLflow wrapper. The scorer,
# its grading template, the grade parsing and the model plumbing are real Inspect AI code.
# Only the LLM call is replaced, once by MLflow's backend mock and once by Inspect's mockllm.
from importlib.util import find_spec
from unittest.mock import patch

import pytest

import mlflow
from mlflow.genai.judges.utils import CategoricalRating

_COMPLETE = "mlflow.genai.scorers.llm_backend.ScorerLLMClient.complete"

pytestmark = pytest.mark.skipif(
    find_spec("inspect_ai") is None, reason="inspect-ai is not installed"
)


def test_model_graded_fact_end_to_end_through_mlflow_backend():
    from mlflow.genai.scorers.inspect_ai import ModelGradedFact

    def grade(messages, **kwargs):
        # Inspect owns the grading template and its field markers change between
        # releases, so check the data it carries rather than the labels around it.
        prompt = messages[-1]["content"]
        assert "What is the capital of France?" in prompt
        assert "Paris" in prompt
        assert "Paris is the capital of France." in prompt
        assert "GRADE:" in prompt
        return "The submission states the expert answer.\nGRADE: C"

    with patch(_COMPLETE, side_effect=grade) as mock_complete:
        scorer = ModelGradedFact(model="openai:/gpt-4o-mini")
        feedback = scorer(
            inputs="What is the capital of France?",
            outputs="Paris is the capital of France.",
            expectations={"expected_response": "Paris"},
        )

    mock_complete.assert_called_once()
    assert feedback.error is None
    assert feedback.value == CategoricalRating.YES
    assert feedback.metadata["raw_value"] == "C"
    assert feedback.rationale == "The submission states the expert answer.\nGRADE: C"


def test_model_graded_fact_end_to_end_with_inspect_mockllm_provider():
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.scorer import model_graded_fact

    from mlflow.genai.scorers.inspect_ai import InspectAIScorer

    grader = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", "Wrong city.\nGRADE: I")],
    )
    scorer = InspectAIScorer("facts", scorer=model_graded_fact(model=grader))

    feedback = scorer(
        inputs="What is the capital of France?",
        outputs="Rome is the capital of France.",
        expectations={"expected_response": "Paris"},
    )

    assert feedback.error is None
    assert feedback.value == CategoricalRating.NO
    assert feedback.metadata["raw_value"] == "I"
    assert feedback.rationale == "Wrong city.\nGRADE: I"


def test_evaluate_runs_inspect_scorers_over_a_dataset():
    from mlflow.genai.scorers.inspect_ai import ExactMatch, ModelGradedFact

    data = [
        {
            "inputs": {"question": "What is the capital of France?"},
            "outputs": "Paris",
            "expectations": {"expected_response": "Paris"},
        },
        {
            "inputs": {"question": "What is 2 + 2?"},
            "outputs": "5",
            "expectations": {"expected_response": "4"},
        },
    ]

    with patch(_COMPLETE, side_effect=["GRADE: C", "GRADE: I"]):
        results = mlflow.genai.evaluate(
            data=data,
            scorers=[ExactMatch(), ModelGradedFact(model="openai:/gpt-4o-mini")],
        )

    assert results.metrics["ExactMatch/mean"] == 0.5
    assert results.metrics["ModelGradedFact/mean"] == 0.5
