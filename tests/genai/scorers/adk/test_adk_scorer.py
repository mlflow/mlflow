"""Tests for ADK evaluation metrics as MLflow scorers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.adk._mapping import (
    _extract_text_from_inputs,
    _extract_text_from_outputs,
    _parse_expected_tool_calls,
    _text_to_content,
    map_adk_result_to_feedback,
    map_to_adk_invocations,
)
from mlflow.genai.scorers.adk._registry import (
    _METRIC_REGISTRY,
    get_adk_metric_name,
    is_deterministic_metric,
)
from mlflow.genai.scorers.base import ScorerKind


# --- Tests for _registry.py ---


class TestRegistry:
    def test_get_adk_metric_name(self):
        assert get_adk_metric_name("ToolTrajectory") == "tool_trajectory_avg_score"
        assert get_adk_metric_name("Hallucinations") == "hallucinations_v1"
        assert get_adk_metric_name("ResponseMatch") == "response_match_score"

    def test_get_adk_metric_name_unknown(self):
        with pytest.raises(Exception, match="Unknown ADK metric"):
            get_adk_metric_name("NonExistentMetric")

    def test_is_deterministic_metric(self):
        assert is_deterministic_metric("ToolTrajectory") is True
        assert is_deterministic_metric("ResponseMatch") is True
        assert is_deterministic_metric("Hallucinations") is False
        assert is_deterministic_metric("FinalResponseMatch") is False

    def test_registry_completeness(self):
        expected_metrics = {
            "ToolTrajectory",
            "FinalResponseMatch",
            "ResponseMatch",
            "RubricBasedResponseQuality",
            "RubricBasedToolUseQuality",
            "Hallucinations",
        }
        assert set(_METRIC_REGISTRY.keys()) == expected_metrics


# --- Tests for _mapping.py ---


class TestTextToContent:
    def test_basic_text(self):
        content = _text_to_content("Hello world")
        assert content.parts[0].text == "Hello world"

    def test_empty_text(self):
        content = _text_to_content("")
        assert content.parts[0].text == ""


class TestExtractTextFromInputs:
    def test_dict_with_query(self):
        assert _extract_text_from_inputs({"query": "test"}) == "test"

    def test_dict_with_question(self):
        assert _extract_text_from_inputs({"question": "test"}) == "test"

    def test_dict_with_input(self):
        assert _extract_text_from_inputs({"input": "test"}) == "test"

    def test_string_input(self):
        assert _extract_text_from_inputs("test string") == "test string"

    def test_none_input(self):
        assert _extract_text_from_inputs(None) == ""

    def test_dict_fallback_to_first_string(self):
        result = _extract_text_from_inputs({"custom_key": "value"})
        assert result == "value"


class TestExtractTextFromOutputs:
    def test_string_output(self):
        assert _extract_text_from_outputs("response text") == "response text"

    def test_dict_with_response(self):
        assert _extract_text_from_outputs({"response": "text"}) == "text"

    def test_none_output(self):
        assert _extract_text_from_outputs(None) == ""


class TestParseExpectedToolCalls:
    def test_list_of_strings(self):
        result = _parse_expected_tool_calls(["search", "book"])
        assert len(result.tool_uses) == 2
        assert result.tool_uses[0].name == "search"
        assert result.tool_uses[1].name == "book"

    def test_list_of_dicts(self):
        result = _parse_expected_tool_calls([
            {"name": "search", "args": {"q": "flights"}},
            {"name": "book", "args": {"id": "123"}},
        ])
        assert len(result.tool_uses) == 2
        assert result.tool_uses[0].name == "search"
        assert result.tool_uses[0].args == {"q": "flights"}


class TestMapToAdkInvocations:
    def test_basic_mapping(self):
        actual, expected = map_to_adk_invocations(
            inputs={"query": "What is Python?"},
            outputs="Python is a language.",
        )

        assert actual.user_content.parts[0].text == "What is Python?"
        assert actual.final_response.parts[0].text == "Python is a language."
        assert expected is None

    def test_with_expectations(self):
        actual, expected = map_to_adk_invocations(
            inputs={"query": "What is Python?"},
            outputs="Python is a language.",
            expectations={
                "expected_response": "Python is a programming language.",
                "expected_tool_calls": ["search"],
            },
        )

        assert expected is not None
        assert expected.final_response.parts[0].text == "Python is a programming language."
        assert len(expected.intermediate_data.tool_uses) == 1
        assert expected.intermediate_data.tool_uses[0].name == "search"

    def test_string_inputs(self):
        actual, _ = map_to_adk_invocations(
            inputs="raw string input",
            outputs="raw output",
        )
        assert actual.user_content.parts[0].text == "raw string input"

    def test_none_outputs(self):
        actual, _ = map_to_adk_invocations(
            inputs="test",
            outputs=None,
        )
        assert actual.final_response is None


class TestMapAdkResultToFeedback:
    def _make_eval_result(self, overall_score, status_name):
        from google.adk.evaluation.eval_metrics import EvalStatus
        from google.adk.evaluation.evaluator import EvaluationResult

        status = getattr(EvalStatus, status_name)
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=status,
        )

    def test_passed_result(self):
        eval_result = self._make_eval_result(0.9, "PASSED")
        source = AssessmentSource(source_type=AssessmentSourceType.CODE)

        feedback = map_adk_result_to_feedback(
            name="test_metric",
            eval_result=eval_result,
            source=source,
        )

        assert isinstance(feedback, Feedback)
        assert feedback.name == "test_metric"
        assert feedback.value == 0.9
        assert feedback.metadata[FRAMEWORK_METADATA_KEY] == "adk"
        assert feedback.metadata["eval_status"] == "PASSED"

    def test_failed_result(self):
        eval_result = self._make_eval_result(0.3, "FAILED")
        source = AssessmentSource(source_type=AssessmentSourceType.CODE)

        feedback = map_adk_result_to_feedback(
            name="test_metric",
            eval_result=eval_result,
            source=source,
        )

        assert feedback.value == 0.3

    def test_not_evaluated_result(self):
        eval_result = self._make_eval_result(None, "NOT_EVALUATED")
        source = AssessmentSource(source_type=AssessmentSourceType.CODE)

        feedback = map_adk_result_to_feedback(
            name="test_metric",
            eval_result=eval_result,
            source=source,
        )

        assert feedback.error is not None


# --- Tests for AdkScorer ---

# Patch _create_evaluator at the class level since it's a static method
# that creates real ADK objects internally. We mock it to return a mock
# evaluator directly.
_PATCH_CREATE_EVALUATOR = "mlflow.genai.scorers.adk.AdkScorer._create_evaluator"


class TestAdkScorer:
    @patch(_PATCH_CREATE_EVALUATOR)
    def test_scorer_kind(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import AdkScorer

        scorer = AdkScorer(metric_name="ToolTrajectory")
        assert scorer.kind == ScorerKind.THIRD_PARTY

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_is_not_session_level(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import AdkScorer

        scorer = AdkScorer(metric_name="ToolTrajectory")
        assert scorer.is_session_level_scorer is False

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_register_not_supported(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import AdkScorer

        scorer = AdkScorer(metric_name="ToolTrajectory")
        with pytest.raises(Exception, match="not supported"):
            scorer.register()
        with pytest.raises(Exception, match="not supported"):
            scorer.start()
        with pytest.raises(Exception, match="not supported"):
            scorer.stop()

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_call_with_successful_evaluation(self, mock_create):
        from google.adk.evaluation.eval_case import Invocation
        from google.adk.evaluation.eval_metrics import EvalStatus
        from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult
        from google.genai import types as genai_types

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_invocations.return_value = EvaluationResult(
            overall_score=1.0,
            overall_eval_status=EvalStatus.PASSED,
            per_invocation_results=[
                PerInvocationResult(
                    actual_invocation=Invocation(
                        user_content=genai_types.Content(
                            parts=[genai_types.Part(text="test")]
                        ),
                    ),
                    score=1.0,
                    eval_status=EvalStatus.PASSED,
                )
            ],
        )
        mock_create.return_value = mock_evaluator

        from mlflow.genai.scorers.adk import AdkScorer

        scorer = AdkScorer(metric_name="ToolTrajectory")
        feedback = scorer(
            inputs={"query": "test"},
            outputs="test output",
            expectations={"expected_tool_calls": ["tool1"]},
        )

        assert isinstance(feedback, Feedback)
        assert feedback.value == 1.0
        assert feedback.metadata[FRAMEWORK_METADATA_KEY] == "adk"

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_call_handles_evaluator_error(self, mock_create):
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_invocations.side_effect = RuntimeError(
            "Evaluator failed"
        )
        mock_create.return_value = mock_evaluator

        from mlflow.genai.scorers.adk import AdkScorer

        scorer = AdkScorer(metric_name="ToolTrajectory")
        feedback = scorer(
            inputs={"query": "test"},
            outputs="test output",
        )

        assert isinstance(feedback, Feedback)
        assert feedback.error is not None

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_deterministic_scorer_source_type(self, mock_create):
        from google.adk.evaluation.eval_case import Invocation
        from google.adk.evaluation.eval_metrics import EvalStatus
        from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult
        from google.genai import types as genai_types

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_invocations.return_value = EvaluationResult(
            overall_score=1.0,
            overall_eval_status=EvalStatus.PASSED,
            per_invocation_results=[
                PerInvocationResult(
                    actual_invocation=Invocation(
                        user_content=genai_types.Content(
                            parts=[genai_types.Part(text="test")]
                        ),
                    ),
                    score=1.0,
                    eval_status=EvalStatus.PASSED,
                )
            ],
        )
        mock_create.return_value = mock_evaluator

        from mlflow.genai.scorers.adk import AdkScorer

        # ToolTrajectory is deterministic
        scorer = AdkScorer(metric_name="ToolTrajectory")
        feedback = scorer(inputs="test", outputs="output")

        assert feedback.source.source_type == AssessmentSourceType.CODE

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_non_deterministic_scorer_source_type(self, mock_create):
        from google.adk.evaluation.eval_case import Invocation
        from google.adk.evaluation.eval_metrics import EvalStatus
        from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult
        from google.genai import types as genai_types

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_invocations.return_value = EvaluationResult(
            overall_score=0.8,
            overall_eval_status=EvalStatus.PASSED,
            per_invocation_results=[
                PerInvocationResult(
                    actual_invocation=Invocation(
                        user_content=genai_types.Content(
                            parts=[genai_types.Part(text="test")]
                        ),
                    ),
                    score=0.8,
                    eval_status=EvalStatus.PASSED,
                )
            ],
        )
        mock_create.return_value = mock_evaluator

        from mlflow.genai.scorers.adk import AdkScorer

        # Hallucinations is non-deterministic
        scorer = AdkScorer(metric_name="Hallucinations", model="gemini-2.5-flash")
        feedback = scorer(inputs="test", outputs="output")

        assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
        assert feedback.source.source_id == "gemini-2.5-flash"


# --- Tests for named scorer subclasses ---


class TestNamedScorers:
    @patch(_PATCH_CREATE_EVALUATOR)
    def test_tool_trajectory_class(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import ToolTrajectory

        scorer = ToolTrajectory(match_type="in_order")
        assert scorer.name == "ToolTrajectory"

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_hallucinations_class(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import Hallucinations

        scorer = Hallucinations(model="gemini-2.5-flash")
        assert scorer.name == "Hallucinations"

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_response_match_class(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import ResponseMatch

        scorer = ResponseMatch()
        assert scorer.name == "ResponseMatch"

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_final_response_match_class(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import FinalResponseMatch

        scorer = FinalResponseMatch(model="gemini-2.5-flash")
        assert scorer.name == "FinalResponseMatch"

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_rubric_based_response_quality_class(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import RubricBasedResponseQuality

        scorer = RubricBasedResponseQuality(model="gemini-2.5-flash")
        assert scorer.name == "RubricBasedResponseQuality"

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_rubric_based_tool_use_quality_class(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import RubricBasedToolUseQuality

        scorer = RubricBasedToolUseQuality(model="gemini-2.5-flash")
        assert scorer.name == "RubricBasedToolUseQuality"


# --- Tests for get_scorer factory ---


class TestGetScorer:
    @patch(_PATCH_CREATE_EVALUATOR)
    def test_get_scorer_basic(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import get_scorer

        scorer = get_scorer("ToolTrajectory", match_type="exact")
        assert scorer.name == "ToolTrajectory"

    @patch(_PATCH_CREATE_EVALUATOR)
    def test_get_scorer_with_model(self, mock_create):
        mock_create.return_value = MagicMock()

        from mlflow.genai.scorers.adk import get_scorer

        scorer = get_scorer("Hallucinations", model="gemini-2.5-flash")
        assert scorer.name == "Hallucinations"
