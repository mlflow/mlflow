from unittest.mock import Mock, patch

import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import SpanType
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import ScorerKind
from mlflow.genai.scorers.google_adk import (
    ResponseMatch,
    ToolTrajectory,
    get_scorer,
)

_PATCH_PREFIX = "mlflow.genai.scorers.google_adk"


@pytest.fixture(autouse=True)
def _mock_adk_installed():
    with patch(f"{_PATCH_PREFIX}.check_adk_installed"):
        yield


def _make_eval_result(overall_score):
    """Build a mock EvaluationResult matching the ADK shape."""
    mock_result = Mock()
    mock_result.overall_score = overall_score
    mock_result.per_invocation_results = []
    return mock_result


def _mock_trajectory_evaluator(overall_score):
    mock_evaluator = Mock()
    mock_evaluator.evaluate_invocations.return_value = _make_eval_result(overall_score)
    return mock_evaluator


def _mock_rouge_evaluator(overall_score):
    mock_evaluator = Mock()
    mock_evaluator.evaluate_invocations.return_value = _make_eval_result(overall_score)
    return mock_evaluator


def _mock_invocations():
    """Return a pair of (actual_invocation, expected_invocation) mocks."""
    return (Mock(), Mock())


# ---------------------------------------------------------------------------
# ToolTrajectory scorer tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected_value"),
    [
        (1.0, CategoricalRating.YES),
        (0.0, CategoricalRating.NO),
    ],
    ids=["pass", "fail"],
)
def test_tool_trajectory_scorer(score, expected_value):
    with (
        patch(
            f"{_PATCH_PREFIX}._create_trajectory_evaluator",
            return_value=_mock_trajectory_evaluator(score),
        ),
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ToolTrajectory(match_type="EXACT", threshold=0.5)
        result = scorer(
            inputs="Book a flight",
            outputs="Flight booked",
            expectations={
                "expected_tool_calls": [
                    {"name": "search_flights", "args": {"dest": "Paris"}},
                ],
            },
        )

    assert isinstance(result, Feedback)
    assert result.name == "ToolTrajectory"
    assert result.value == expected_value
    assert result.metadata["score"] == score
    assert result.metadata["threshold"] == 0.5
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "google_adk"
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="google_adk/ToolTrajectory",
    )


def test_tool_trajectory_missing_expectations():
    with patch(
        f"{_PATCH_PREFIX}._create_trajectory_evaluator",
        return_value=_mock_trajectory_evaluator(0.0),
    ):
        scorer = ToolTrajectory()
        result = scorer(
            inputs="test",
            outputs="test",
            expectations=None,
        )

    assert isinstance(result, Feedback)
    assert result.name == "ToolTrajectory"
    assert result.error is not None
    assert "expected_tool_calls" in str(result.error)
    assert result.metadata == {FRAMEWORK_METADATA_KEY: "google_adk"}
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="google_adk/ToolTrajectory",
    )


def test_tool_trajectory_missing_expected_tool_calls_key():
    with patch(
        f"{_PATCH_PREFIX}._create_trajectory_evaluator",
        return_value=_mock_trajectory_evaluator(0.0),
    ):
        scorer = ToolTrajectory()
        result = scorer(
            inputs="test",
            outputs="test",
            expectations={"some_other_key": "value"},
        )

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "expected_tool_calls" in str(result.error)
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="google_adk/ToolTrajectory",
    )


def test_tool_trajectory_error_handling():
    mock_evaluator = Mock()
    mock_evaluator.evaluate_invocations.side_effect = RuntimeError("ADK internal error")

    with (
        patch(f"{_PATCH_PREFIX}._create_trajectory_evaluator", return_value=mock_evaluator),
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ToolTrajectory()
        result = scorer(
            inputs="test",
            outputs="test",
            expectations={
                "expected_tool_calls": [{"name": "tool_a", "args": {}}],
            },
        )

    assert isinstance(result, Feedback)
    assert result.name == "ToolTrajectory"
    assert result.error is not None
    assert "ADK internal error" in str(result.error)
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="google_adk/ToolTrajectory",
    )
    assert result.metadata == {FRAMEWORK_METADATA_KEY: "google_adk"}


@pytest.mark.parametrize(
    "match_type",
    ["EXACT", "IN_ORDER", "ANY_ORDER"],
)
def test_tool_trajectory_match_types(match_type):
    with (
        patch(
            f"{_PATCH_PREFIX}._create_trajectory_evaluator",
            return_value=_mock_trajectory_evaluator(1.0),
        ) as mock_create,
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ToolTrajectory(match_type=match_type, threshold=0.5)
        result = scorer(
            inputs="test",
            outputs="result",
            expectations={
                "expected_tool_calls": [{"name": "tool_a", "args": {}}],
            },
        )

    assert result.value == CategoricalRating.YES
    mock_create.assert_called_once_with(0.5, match_type)


def test_tool_trajectory_evaluator_called():
    mock_evaluator = _mock_trajectory_evaluator(1.0)

    with (
        patch(f"{_PATCH_PREFIX}._create_trajectory_evaluator", return_value=mock_evaluator),
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ToolTrajectory()
        scorer(
            inputs="test",
            outputs="result",
            expectations={
                "expected_tool_calls": [{"name": "tool_a", "args": {}}],
            },
        )

    mock_evaluator.evaluate_invocations.assert_called_once()


# ---------------------------------------------------------------------------
# ResponseMatch scorer tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected_value"),
    [
        (0.85, CategoricalRating.YES),
        (0.2, CategoricalRating.NO),
    ],
    ids=["pass", "fail"],
)
def test_response_match_scorer(score, expected_value):
    with (
        patch(
            f"{_PATCH_PREFIX}._create_rouge_evaluator",
            return_value=_mock_rouge_evaluator(score),
        ),
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ResponseMatch(threshold=0.5)
        result = scorer(
            outputs="MLflow is an ML platform.",
            expectations={"expected_response": "MLflow is an ML lifecycle platform."},
        )

    assert isinstance(result, Feedback)
    assert result.name == "ResponseMatch"
    assert result.value == expected_value
    assert result.metadata["score"] == score
    assert result.metadata["threshold"] == 0.5
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "google_adk"
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="google_adk/ResponseMatch",
    )


def test_response_match_missing_reference():
    with patch(
        f"{_PATCH_PREFIX}._create_rouge_evaluator",
        return_value=_mock_rouge_evaluator(0.0),
    ):
        scorer = ResponseMatch(threshold=0.5)
        result = scorer(
            outputs="Some output",
            expectations=None,
        )

    assert isinstance(result, Feedback)
    assert result.name == "ResponseMatch"
    assert result.error is not None
    assert "expected_response" in str(result.error)
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="google_adk/ResponseMatch",
    )


def test_response_match_uses_reference_key():
    with (
        patch(
            f"{_PATCH_PREFIX}._create_rouge_evaluator",
            return_value=_mock_rouge_evaluator(0.9),
        ),
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ResponseMatch(threshold=0.5)
        result = scorer(
            outputs="MLflow platform",
            expectations={"reference": "MLflow is a platform"},
        )

    assert result.value == CategoricalRating.YES
    assert result.metadata["score"] == 0.9


def test_response_match_error_handling():
    mock_evaluator = Mock()
    mock_evaluator.evaluate_invocations.side_effect = RuntimeError("ROUGE failed")

    with (
        patch(f"{_PATCH_PREFIX}._create_rouge_evaluator", return_value=mock_evaluator),
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ResponseMatch(threshold=0.5)
        result = scorer(
            outputs="output text",
            expectations={"expected_response": "reference text"},
        )

    assert isinstance(result, Feedback)
    assert result.name == "ResponseMatch"
    assert result.error is not None
    assert "ROUGE failed" in str(result.error)
    assert result.source == AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="google_adk/ResponseMatch",
    )
    assert result.metadata == {FRAMEWORK_METADATA_KEY: "google_adk"}


def test_response_match_evaluator_called():
    mock_evaluator = _mock_rouge_evaluator(0.8)

    with (
        patch(f"{_PATCH_PREFIX}._create_rouge_evaluator", return_value=mock_evaluator),
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = ResponseMatch(threshold=0.5)
        scorer(
            outputs="output text",
            expectations={"expected_response": "reference text"},
        )

    mock_evaluator.evaluate_invocations.assert_called_once()


# ---------------------------------------------------------------------------
# get_scorer factory tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "expected_class"),
    [
        ("ToolTrajectory", ToolTrajectory),
        ("ResponseMatch", ResponseMatch),
    ],
)
def test_get_scorer_returns_correct_class(metric_name, expected_class):
    with (
        patch(f"{_PATCH_PREFIX}._create_trajectory_evaluator"),
        patch(f"{_PATCH_PREFIX}._create_rouge_evaluator"),
    ):
        scorer = get_scorer(metric_name, threshold=0.7)

    assert isinstance(scorer, expected_class)


def test_get_scorer_unknown_metric():
    with pytest.raises(Exception, match="Unknown Google ADK metric"):
        get_scorer("NonExistentMetric")


def test_get_scorer_passes_kwargs():
    with (
        patch(
            f"{_PATCH_PREFIX}._create_trajectory_evaluator",
            return_value=_mock_trajectory_evaluator(1.0),
        ) as mock_create,
        patch(f"{_PATCH_PREFIX}.map_scorer_inputs_to_invocation", return_value=_mock_invocations()),
    ):
        scorer = get_scorer("ToolTrajectory", match_type="ANY_ORDER", threshold=0.8)
        result = scorer(
            inputs="test",
            outputs="result",
            expectations={
                "expected_tool_calls": [{"name": "tool", "args": {}}],
            },
        )

    assert result.metadata["threshold"] == 0.8
    mock_create.assert_called_once_with(0.8, "ANY_ORDER")


# ---------------------------------------------------------------------------
# Scorer properties and registration
# ---------------------------------------------------------------------------


def test_scorer_kind_is_third_party():
    with patch(f"{_PATCH_PREFIX}._create_trajectory_evaluator"):
        scorer = ToolTrajectory()

    assert scorer.kind == ScorerKind.THIRD_PARTY


def test_scorer_registration_not_supported():
    with patch(f"{_PATCH_PREFIX}._create_trajectory_evaluator"):
        scorer = ToolTrajectory()

    with pytest.raises(Exception, match="not supported for third-party scorers"):
        scorer.register()

    with pytest.raises(Exception, match="not supported for third-party scorers"):
        scorer.start()

    with pytest.raises(Exception, match="not supported for third-party scorers"):
        scorer.update()

    with pytest.raises(Exception, match="not supported for third-party scorers"):
        scorer.stop()

    with pytest.raises(Exception, match="not supported for third-party scorers"):
        scorer.align()


# ---------------------------------------------------------------------------
# Source ID verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scorer_factory", "expected_source_id"),
    [
        (
            lambda: ToolTrajectory(),
            "google_adk/ToolTrajectory",
        ),
        (
            lambda: ResponseMatch(),
            "google_adk/ResponseMatch",
        ),
    ],
    ids=["ToolTrajectory", "ResponseMatch"],
)
def test_scorer_source_id(scorer_factory, expected_source_id):
    with (
        patch(f"{_PATCH_PREFIX}._create_trajectory_evaluator"),
        patch(f"{_PATCH_PREFIX}._create_rouge_evaluator"),
    ):
        scorer = scorer_factory()

    assert scorer.name in expected_source_id


# ---------------------------------------------------------------------------
# Trace integration
# ---------------------------------------------------------------------------


def _create_test_trace(inputs=None, outputs=None):
    with mlflow.start_span() as span:
        if inputs is not None:
            span.set_inputs(inputs)
        if outputs is not None:
            span.set_outputs(outputs)
    return mlflow.get_trace(span.trace_id)


def _create_tool_trace(inputs, outputs, tool_calls):
    """Build a trace containing one or more TOOL spans.

    Each entry in ``tool_calls`` is a ``(name, args, result)`` triple.
    """
    with mlflow.start_span(name="root") as root:
        root.set_inputs(inputs)
        for name, args, result in tool_calls:
            with mlflow.start_span(name=name, span_type=SpanType.TOOL) as tool:
                tool.set_inputs(args)
                tool.set_outputs(result)
        root.set_outputs(outputs)
    return mlflow.get_trace(root.trace_id)


def test_response_match_with_trace():
    trace = _create_test_trace(
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is a platform for ML."},
    )

    with patch(
        f"{_PATCH_PREFIX}._create_rouge_evaluator",
        return_value=_mock_rouge_evaluator(0.75),
    ):
        scorer = ResponseMatch(threshold=0.5)
        result = scorer(
            trace=trace,
            expectations={"expected_response": "MLflow is an ML platform."},
        )

    assert isinstance(result, Feedback)
    assert result.name == "ResponseMatch"
    assert result.value == CategoricalRating.YES
    assert result.metadata["score"] == 0.75


def test_tool_trajectory_with_trace_extracts_actual_calls():
    """Regression test for the bug where actual tool calls were never
    extracted from the trace: the scorer silently reported 0.0 for any
    real agent run that didn't duplicate tool calls under
    ``expectations["actual_tool_calls"]``.
    """
    trace = _create_tool_trace(
        inputs={"query": "Book a flight to Paris"},
        outputs="Booked flight AA123",
        tool_calls=[
            ("search_flights", {"destination": "Paris"}, {"flights": ["AA123"]}),
            ("book_flight", {"flight_id": "AA123"}, {"status": "confirmed"}),
        ],
    )

    captured = {}

    def _fake_evaluate(*, actual_invocations, expected_invocations):
        captured["actual"] = actual_invocations
        captured["expected"] = expected_invocations
        return _make_eval_result(1.0)

    mock_evaluator = Mock()
    mock_evaluator.evaluate_invocations = _fake_evaluate

    with patch(
        f"{_PATCH_PREFIX}._create_trajectory_evaluator",
        return_value=mock_evaluator,
    ):
        scorer = ToolTrajectory(match_type="EXACT", threshold=0.5)
        result = scorer(
            trace=trace,
            expectations={
                "expected_tool_calls": [
                    {"name": "search_flights", "args": {"destination": "Paris"}},
                    {"name": "book_flight", "args": {"flight_id": "AA123"}},
                ],
            },
        )

    # Scorer result
    assert result.value == CategoricalRating.YES
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "google_adk"

    # Actual tool calls must be populated from the trace, not left empty
    actual_inv = captured["actual"][0]
    assert actual_inv.intermediate_data is not None
    actual_tool_names = [t.name for t in actual_inv.intermediate_data.tool_uses]
    assert actual_tool_names == ["search_flights", "book_flight"]


def test_tool_trajectory_expectation_override_beats_trace():
    """Explicit ``expectations["actual_tool_calls"]`` should override trace
    extraction so callers can inject synthetic data when needed.
    """
    trace = _create_tool_trace(
        inputs={"q": "x"},
        outputs="y",
        tool_calls=[("trace_tool", {"from": "trace"}, {"ok": True})],
    )

    captured = {}

    def _fake_evaluate(*, actual_invocations, expected_invocations):
        captured["actual"] = actual_invocations
        return _make_eval_result(1.0)

    mock_evaluator = Mock()
    mock_evaluator.evaluate_invocations = _fake_evaluate

    with patch(
        f"{_PATCH_PREFIX}._create_trajectory_evaluator",
        return_value=mock_evaluator,
    ):
        scorer = ToolTrajectory(match_type="EXACT")
        scorer(
            trace=trace,
            expectations={
                "expected_tool_calls": [{"name": "override_tool", "args": {}}],
                "actual_tool_calls": [{"name": "override_tool", "args": {}}],
            },
        )

    actual_inv = captured["actual"][0]
    assert actual_inv.intermediate_data is not None
    names = [t.name for t in actual_inv.intermediate_data.tool_uses]
    assert names == ["override_tool"]  # Not "trace_tool"


# ---------------------------------------------------------------------------
# map_scorer_inputs_to_invocation helper tests (require google-adk installed)
# ---------------------------------------------------------------------------


def test_map_scorer_inputs_to_invocation_basic():
    from mlflow.genai.scorers.google_adk.utils import map_scorer_inputs_to_invocation

    actual, expected = map_scorer_inputs_to_invocation(
        inputs="hello",
        outputs="world",
        expectations={
            "expected_tool_calls": [{"name": "greet", "args": {"who": "world"}}],
            "expected_response": "hello world",
        },
    )

    assert actual.user_content.parts[0].text == "hello"
    assert actual.final_response.parts[0].text == "world"
    assert expected.intermediate_data.tool_uses[0].name == "greet"
    assert expected.intermediate_data.tool_uses[0].args == {"who": "world"}
    assert expected.final_response.parts[0].text == "hello world"


def test_map_scorer_inputs_to_invocation_no_expectations():
    from mlflow.genai.scorers.google_adk.utils import map_scorer_inputs_to_invocation

    actual, expected = map_scorer_inputs_to_invocation(
        inputs="test input",
        outputs="test output",
    )

    assert actual.user_content.parts[0].text == "test input"
    assert actual.final_response.parts[0].text == "test output"
    assert expected.intermediate_data is None
    assert expected.final_response is None


def test_map_scorer_inputs_to_invocation_uses_context_fallback():
    from mlflow.genai.scorers.google_adk.utils import map_scorer_inputs_to_invocation

    _, expected = map_scorer_inputs_to_invocation(
        inputs="q",
        outputs="a",
        expectations={"context": "reference from context key"},
    )

    assert expected.final_response.parts[0].text == "reference from context key"


def test_extract_actual_tool_calls_from_trace():
    from mlflow.genai.scorers.google_adk.utils import _extract_actual_tool_calls

    trace = _create_tool_trace(
        inputs={"q": "weather and stock"},
        outputs="done",
        tool_calls=[
            ("get_weather", {"city": "NYC"}, {"temp": 72}),
            ("get_stock", {"symbol": "AAPL"}, {"price": 150}),
        ],
    )

    result = _extract_actual_tool_calls(expectations=None, trace=trace)
    assert [c["name"] for c in result] == ["get_weather", "get_stock"]
    assert result[0]["args"] == {"city": "NYC"}
    assert result[1]["args"] == {"symbol": "AAPL"}


def test_extract_actual_tool_calls_expectation_override():
    from mlflow.genai.scorers.google_adk.utils import _extract_actual_tool_calls

    trace = _create_tool_trace(
        inputs={"q": "x"},
        outputs="y",
        tool_calls=[("from_trace", {}, {})],
    )
    result = _extract_actual_tool_calls(
        expectations={"actual_tool_calls": [{"name": "explicit", "args": {}}]},
        trace=trace,
    )
    assert [c["name"] for c in result] == ["explicit"]


def test_extract_actual_tool_calls_no_trace_no_expectation():
    from mlflow.genai.scorers.google_adk.utils import _extract_actual_tool_calls

    assert _extract_actual_tool_calls(expectations=None, trace=None) == []
    assert _extract_actual_tool_calls(expectations={}, trace=None) == []


def test_map_scorer_inputs_to_invocation_reads_trace_tool_calls():
    from mlflow.genai.scorers.google_adk.utils import map_scorer_inputs_to_invocation

    trace = _create_tool_trace(
        inputs={"q": "Book a flight"},
        outputs="Booked",
        tool_calls=[("book_flight", {"dest": "Paris"}, {"ok": True})],
    )

    actual, _ = map_scorer_inputs_to_invocation(
        trace=trace,
        expectations={"expected_tool_calls": [{"name": "book_flight", "args": {"dest": "Paris"}}]},
    )
    assert actual.intermediate_data is not None
    names = [t.name for t in actual.intermediate_data.tool_uses]
    assert names == ["book_flight"]
