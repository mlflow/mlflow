"""Tests for the InspectAIScorer and get_scorer public API.

Covers:
- Happy-path: normal scorer output flows through to Feedback correctly.
- Negative: missing package and runtime scorer exceptions surface as Feedback.error.
- Terminal-state semantics: skip, not-run, and scorer-level error are each
  distinguishable via the ``terminal_state`` metadata key.
- Input mapping: MLflow eval inputs are correctly converted to Inspect AI payloads.
- Serialization: scorer kwargs are preserved through model_dump.
- Callable resolution: adapter correctly locates scorer callables.
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.inspect_ai import InspectAIScorer, get_scorer
from mlflow.genai.scorers.inspect_ai.registry import get_task_callable
from mlflow.genai.scorers.inspect_ai.utils import map_scorer_inputs_to_inspectai_payload
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY


class DummyInspectAIModule:
    def score(self, metric_name: str, payload: dict, **kwargs):
        return {"value": "yes", "score": 0.9, "reason": "Looks good", "extra": kwargs}


@pytest.fixture(autouse=True)
def mock_inspectai_module(monkeypatch):
    dummy = DummyInspectAIModule()
    monkeypatch.setitem(__import__("sys").modules, "inspectai", dummy)
    monkeypatch.setitem(__import__("sys").modules, "inspect_ai", dummy)
    yield


def test_inspect_ai_scorer_calls_inspectai_score():
    """Normal scorer output is translated into a Feedback with value, rationale, and source set."""
    scorer = get_scorer("TestMetric", model="openai:/gpt-4")

    with patch("mlflow.genai.scorers.inspect_ai.wrapper._ensure_inspectai_installed") as mock_installed:
        mock_installed.return_value = DummyInspectAIModule()
        result = scorer(inputs="input", outputs="output", expectations={"label": "yes"})

    assert isinstance(result, Feedback)
    assert result.name == "TestMetric"
    assert result.value == "yes"
    assert result.rationale == "Looks good"
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "inspect_ai"
    assert result.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert result.source.source_id == "openai:/gpt-4"



def test_inspect_ai_scorer_handles_missing_inspectai():
    """If the inspectai package is not installed, the scorer returns Feedback with error set."""
    scorer = get_scorer("TestMetric")
    with patch("mlflow.genai.scorers.inspect_ai.wrapper._ensure_inspectai_installed", side_effect=MlflowException("Missing")):
        result = scorer(inputs="input", outputs="output")

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "inspect_ai"



def test_scorer_runtime_exception_becomes_feedback_error():
    """A scorer crash must surface as Feedback.error — not disappear from the aggregate."""
    scorer = get_scorer("TestMetric", model="openai:/gpt-4")

    with patch("mlflow.genai.scorers.inspect_ai.wrapper._ensure_inspectai_installed") as mock_installed, \
         patch("mlflow.genai.scorers.inspect_ai.adapter.invoke_task_callable",
               side_effect=RuntimeError("scorer crashed internally")):
        mock_installed.return_value = DummyInspectAIModule()
        result = scorer(inputs="input", outputs="output")

    assert isinstance(result, Feedback)
    assert result.error is not None, "scorer exception must be recorded as Feedback.error"
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "inspect_ai"



def test_scorer_skip_state_is_distinguishable():
    """A scorer-produced skip verdict must set terminal_state=skip, not collapse to None score."""
    from mlflow.genai.scorers.inspect_ai.wrapper import _normalize_inspectai_result

    result = _normalize_inspectai_result(
        {"status": "skip", "reason": "case excluded from run"}, name="TestMetric"
    )

    assert isinstance(result, Feedback)
    assert result.value is None
    assert result.metadata["terminal_state"] == "skip"
    assert result.rationale == "case excluded from run"


def test_scorer_skipped_variant_state():
    """Both 'skip' and 'skipped' status strings map to terminal_state='skip'."""
    from mlflow.genai.scorers.inspect_ai.wrapper import _normalize_inspectai_result

    result = _normalize_inspectai_result({"status": "skipped"}, name="TestMetric")
    assert result.metadata["terminal_state"] == "skip"


def test_scorer_none_result_is_not_run():
    """None result must be recorded as not_run, not as a scored failure."""
    from mlflow.genai.scorers.inspect_ai.wrapper import _normalize_inspectai_result

    result = _normalize_inspectai_result(None, name="TestMetric")

    assert isinstance(result, Feedback)
    assert result.value is None
    assert result.metadata["terminal_state"] == "not_run"


def test_scorer_error_status_becomes_error_terminal_state():
    """A scorer-level error dict is recorded with terminal_state='error' and the reason preserved."""
    from mlflow.genai.scorers.inspect_ai.wrapper import _normalize_inspectai_result

    result = _normalize_inspectai_result(
        {"status": "error", "reason": "LLM judge API returned 500"}, name="TestMetric"
    )

    assert result.metadata["terminal_state"] == "error"
    assert result.rationale == "LLM judge API returned 500"



def test_map_scorer_inputs_to_inspectai_payload_returns_expected_structure():
    """Inputs, outputs, and expectations are mapped to the expected payload keys."""
    payload = map_scorer_inputs_to_inspectai_payload(
        metric_name="TestMetric",
        inputs="input",
        outputs="output",
        expectations={"expected_output": "expected"},
    )

    assert payload["metric_name"] == "TestMetric"
    assert payload["input"] == "input"
    assert payload["output"] == "output"
    assert payload["expectations"] == {"expected_output": "expected"}



def test_inspect_ai_scorer_serialization_includes_kwargs():
    """Extra scorer_kwargs are preserved in model_dump so the scorer can be reconstructed."""
    scorer = get_scorer("TestMetric", model="openai:/gpt-4", model_kwargs={"temperature": 0.0}, custom_flag=True)
    dump = scorer.model_dump()

    assert dump["third_party_scorer_data"]["kwargs"]["custom_flag"] is True



def test_inspect_ai_task_callable_resolution():
    """The registry correctly resolves a callable from the mocked Inspect AI module."""
    with patch("mlflow.genai.scorers.inspect_ai.adapter._import_inspectai_module") as mock_import:
        dummy = DummyInspectAIModule()
        mock_import.return_value = dummy
        task = get_task_callable("score")
        assert callable(task)
