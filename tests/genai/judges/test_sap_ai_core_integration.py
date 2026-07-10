"""Tests for extra_headers threading through BuiltInScorer and judge functions."""

from unittest import mock

import pytest

from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType, Feedback
from mlflow.genai.judges.utils import CategoricalRating


def _make_feedback(name: str = "test", value: str = "yes") -> Feedback:
    return Feedback(
        name=name,
        value=CategoricalRating.YES if value == "yes" else CategoricalRating.NO,
        rationale="test rationale",
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="sap-ai-core:/gpt-4o-mini",
        ),
    )


# ---------------------------------------------------------------------------
# BuiltInScorer — extra_headers field
# ---------------------------------------------------------------------------


def test_builtin_scorer_has_extra_headers_field():
    from mlflow.genai.scorers import Safety

    scorer = Safety(model="sap-ai-core:/gpt-4o-mini", extra_headers={"AI-Resource-Group": "grp"})
    assert scorer.extra_headers == {"AI-Resource-Group": "grp"}


def test_builtin_scorer_extra_headers_defaults_to_none():
    from mlflow.genai.scorers import Safety

    scorer = Safety(model="sap-ai-core:/gpt-4o-mini")
    assert scorer.extra_headers is None


# ---------------------------------------------------------------------------
# Safety scorer — extra_headers forwarded to judges.is_safe
# ---------------------------------------------------------------------------


def test_safety_scorer_forwards_extra_headers():
    with mock.patch(
        "mlflow.genai.scorers.builtin_scorers.judges.is_safe",
        return_value=_make_feedback("safety"),
    ) as mock_judge:
        from mlflow.genai.scorers import Safety

        scorer = Safety(
            model="sap-ai-core:/gpt-4o-mini",
            extra_headers={"AI-Resource-Group": "default"},
        )
        scorer(outputs="The capital of France is Paris.")

    mock_judge.assert_called_once()
    _, kwargs = mock_judge.call_args
    assert kwargs.get("extra_headers") == {"AI-Resource-Group": "default"}
    assert kwargs.get("model") == "sap-ai-core:/gpt-4o-mini"


def test_safety_scorer_no_extra_headers_passes_none():
    with mock.patch(
        "mlflow.genai.scorers.builtin_scorers.judges.is_safe",
        return_value=_make_feedback("safety"),
    ) as mock_judge:
        from mlflow.genai.scorers import Safety

        Safety(model="sap-ai-core:/gpt-4o-mini")(outputs="Hello world.")

    _, kwargs = mock_judge.call_args
    assert kwargs.get("extra_headers") is None


# ---------------------------------------------------------------------------
# Correctness scorer — extra_headers forwarded to judges.is_correct
# ---------------------------------------------------------------------------


def test_correctness_scorer_forwards_extra_headers():
    with mock.patch(
        "mlflow.genai.scorers.builtin_scorers.judges.is_correct",
        return_value=_make_feedback("correctness"),
    ) as mock_judge:
        from mlflow.genai.scorers import Correctness

        scorer = Correctness(
            model="sap-ai-core:/gpt-4o-mini",
            extra_headers={"AI-Resource-Group": "team-a"},
        )
        scorer(
            inputs={"question": "What is the capital of France?"},
            outputs="Paris is the capital of France.",
            expectations={"expected_response": "Paris"},
        )

    mock_judge.assert_called_once()
    _, kwargs = mock_judge.call_args
    assert kwargs.get("extra_headers") == {"AI-Resource-Group": "team-a"}


# ---------------------------------------------------------------------------
# Guidelines scorer — extra_headers forwarded to judges.meets_guidelines
# ---------------------------------------------------------------------------


def test_guidelines_scorer_forwards_extra_headers():
    with mock.patch(
        "mlflow.genai.scorers.builtin_scorers.judges.meets_guidelines",
        return_value=_make_feedback("guidelines"),
    ) as mock_judge:
        from mlflow.genai.scorers import Guidelines

        scorer = Guidelines(
            guidelines=["Must be polite"],
            model="sap-ai-core:/gpt-4o-mini",
            extra_headers={"AI-Resource-Group": "default"},
        )
        scorer(
            inputs={"question": "hello"},
            outputs="Hello, how can I help you?",
        )

    mock_judge.assert_called_once()
    _, kwargs = mock_judge.call_args
    assert kwargs.get("extra_headers") == {"AI-Resource-Group": "default"}


# ---------------------------------------------------------------------------
# Public judge functions — extra_headers forwarded to invoke_judge_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("judge_fn_path", "judge_call_kwargs"),
    [
        (
            "mlflow.genai.judges.builtin.is_safe",
            {
                "content": "Hello world",
                "model": "sap-ai-core:/gpt-4o-mini",
                "extra_headers": {"AI-Resource-Group": "default"},
            },
        ),
        (
            "mlflow.genai.judges.builtin.is_correct",
            {
                "request": "What is 2+2?",
                "response": "4",
                "expected_response": "4",
                "model": "sap-ai-core:/gpt-4o-mini",
                "extra_headers": {"AI-Resource-Group": "default"},
            },
        ),
        (
            "mlflow.genai.judges.builtin.is_grounded",
            {
                "request": "What is the capital?",
                "response": "Paris",
                "context": [{"content": "Paris is the capital."}],
                "model": "sap-ai-core:/gpt-4o-mini",
                "extra_headers": {"AI-Resource-Group": "default"},
            },
        ),
        (
            "mlflow.genai.judges.builtin.is_context_relevant",
            {
                "request": "What is the capital of France?",
                "context": "Paris is the capital of France.",
                "model": "sap-ai-core:/gpt-4o-mini",
                "extra_headers": {"AI-Resource-Group": "default"},
            },
        ),
        (
            "mlflow.genai.judges.builtin.meets_guidelines",
            {
                "guidelines": "Be polite",
                "context": {"response": "Hello!"},
                "model": "sap-ai-core:/gpt-4o-mini",
                "extra_headers": {"AI-Resource-Group": "default"},
            },
        ),
    ],
)
def test_judge_function_forwards_extra_headers_to_invoke(judge_fn_path, judge_call_kwargs):
    """Verify each public judge function passes extra_headers through to invoke_judge_model."""
    with mock.patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=_make_feedback(),
    ) as mock_invoke:
        import importlib

        module_path, fn_name = judge_fn_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        fn = getattr(module, fn_name)
        fn(**judge_call_kwargs)

    mock_invoke.assert_called_once()
    _, kwargs = mock_invoke.call_args
    assert kwargs.get("extra_headers") == {"AI-Resource-Group": "default"}
