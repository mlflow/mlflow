from unittest import mock

import pytest

import mlflow
from mlflow.entities.assessment import (
    AssessmentError,
    AssessmentSource,
    Expectation,
    ExpectationValue,
    Feedback,
    FeedbackValue,
)
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException


@pytest.fixture
def store():
    mock_store = mock.MagicMock()
    with mock.patch("mlflow.tracing.client._get_store", return_value=mock_store):
        yield mock_store


_HUMAN_ASSESSMENT_SOURCE = AssessmentSource(
    source_type=AssessmentSourceType.HUMAN,
    source_id="bob@example.com",
)

_LLM_ASSESSMENT_SOURCE = AssessmentSource(
    source_type=AssessmentSourceType.LLM_JUDGE,
    source_id="gpt-4o-mini",
)


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_expectation(store, legacy_api):
    if legacy_api:
        mlflow.log_expectation(
            trace_id="1234",
            name="expected_answer",
            value="MLflow",
            source=_HUMAN_ASSESSMENT_SOURCE,
            metadata={"key": "value"},
        )
    else:
        feedback = Expectation(
            name="expected_answer",
            value="MLflow",
            source=_HUMAN_ASSESSMENT_SOURCE,
            metadata={"key": "value"},
        )
        mlflow.log_assessment(trace_id="1234", assessment=feedback)

    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.trace_id == "1234"
    assert assessment.name == "expected_answer"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source == _HUMAN_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation.value == "MLflow"
    assert assessment.feedback is None
    assert assessment.rationale is None
    assert assessment.metadata == {"key": "value"}


def test_log_expectation_invalid_parameters():
    with pytest.raises(MlflowException, match=r"The `value` field must be specified."):
        Expectation(
            name="expected_answer",
            value=None,
            source=_HUMAN_ASSESSMENT_SOURCE,
        )


def test_update_expectation(store):
    assessment = Expectation(name="expected_answer", value="MLflow")
    mlflow.update_assessment(
        assessment_id="1234",
        trace_id="tr-1234",
        assessment=assessment,
    )

    assert store.update_assessment.call_count == 1
    call_args = store.update_assessment.call_args[1]
    assert call_args["trace_id"] == "tr-1234"
    assert call_args["assessment_id"] == "1234"
    assert call_args["name"] == "expected_answer"
    assert call_args["expectation"] == ExpectationValue(value="MLflow")
    assert call_args["feedback"] is None
    assert call_args["rationale"] is None
    assert call_args["metadata"] is None


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_feedback(store, legacy_api):
    if legacy_api:
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source=_LLM_ASSESSMENT_SOURCE,
            rationale="This answer is very faithful.",
            metadata={"model": "gpt-4o-mini"},
        )
    else:
        feedback = Feedback(
            name="faithfulness",
            value=1.0,
            source=_LLM_ASSESSMENT_SOURCE,
            rationale="This answer is very faithful.",
            metadata={"model": "gpt-4o-mini"},
        )
        mlflow.log_assessment(trace_id="1234", assessment=feedback)

    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.trace_id == "1234"
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source == _LLM_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.feedback.value == 1.0
    assert assessment.feedback.error is None
    assert assessment.expectation is None
    assert assessment.rationale == "This answer is very faithful."
    assert assessment.metadata == {"model": "gpt-4o-mini"}


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_feedback_with_error(store, legacy_api):
    if legacy_api:
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=_LLM_ASSESSMENT_SOURCE,
            error=AssessmentError(
                error_code="RATE_LIMIT_EXCEEDED",
                error_message="Rate limit for the judge exceeded.",
            ),
        )
    else:
        feedback = Feedback(
            name="faithfulness",
            value=None,
            source=_LLM_ASSESSMENT_SOURCE,
            error=AssessmentError(
                error_code="RATE_LIMIT_EXCEEDED",
                error_message="Rate limit for the judge exceeded.",
            ),
        )
        mlflow.log_assessment(trace_id="1234", assessment=feedback)

    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source == _LLM_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation is None
    assert assessment.feedback.value is None
    assert assessment.feedback.error.error_code == "RATE_LIMIT_EXCEEDED"
    assert assessment.feedback.error.error_message == "Rate limit for the judge exceeded."
    assert assessment.rationale is None


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_feedback_with_exception_object(store, legacy_api):
    """Test that log_feedback correctly accepts Exception objects."""
    test_exception = ValueError("Test exception message")

    if legacy_api:
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=_LLM_ASSESSMENT_SOURCE,
            error=test_exception,
        )
    else:
        feedback = Feedback(
            name="faithfulness",
            value=None,
            source=_LLM_ASSESSMENT_SOURCE,
            error=test_exception,
        )
        mlflow.log_assessment(trace_id="1234", assessment=feedback)

    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source == _LLM_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation is None
    assert assessment.feedback.value is None
    # Exception should be converted to AssessmentError
    assert assessment.feedback.error.error_code == "ValueError"
    assert assessment.feedback.error.error_message == "Test exception message"
    assert assessment.feedback.error.stack_trace is not None
    assert assessment.rationale is None


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_feedback_with_value_and_error(store, legacy_api):
    if legacy_api:
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=_LLM_ASSESSMENT_SOURCE,
            value=0.5,
            error=AssessmentError(
                error_code="RATE_LIMIT_EXCEEDED",
                error_message="Rate limit for the judge exceeded.",
            ),
        )
    else:
        feedback = Feedback(
            name="faithfulness",
            value=0.5,
            source=_LLM_ASSESSMENT_SOURCE,
            error=AssessmentError(
                error_code="RATE_LIMIT_EXCEEDED",
                error_message="Rate limit for the judge exceeded.",
            ),
        )
        mlflow.log_assessment(trace_id="1234", assessment=feedback)

    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source == _LLM_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation is None
    assert assessment.feedback.value == 0.5
    assert assessment.feedback.error.error_code == "RATE_LIMIT_EXCEEDED"
    assert assessment.feedback.error.error_message == "Rate limit for the judge exceeded."
    assert assessment.rationale is None


def test_log_feedback_invalid_parameters():
    with pytest.raises(MlflowException, match=r"Either `value` or `error` must be provided."):
        Feedback(
            trace_id="1234",
            name="faithfulness",
            source=_LLM_ASSESSMENT_SOURCE,
        )

    # Test with a non-AssessmentSource object that is not None
    with pytest.raises(MlflowException, match=r"`source` must be an instance of"):
        Feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source="invalid_source_type",
        )


def test_update_feedback(store):
    feedback = Feedback(
        name="faithfulness",
        value=1.0,
        rationale="This answer is very faithful.",
        metadata={"model": "gpt-4o-mini"},
    )
    mlflow.update_assessment(
        assessment_id="1234",
        trace_id="tr-1234",
        assessment=feedback,
    )

    assert store.update_assessment.call_count == 1
    call_args = store.update_assessment.call_args[1]
    assert call_args["trace_id"] == "tr-1234"
    assert call_args["assessment_id"] == "1234"
    assert call_args["name"] == "faithfulness"
    assert call_args["expectation"] is None
    assert call_args["feedback"] == FeedbackValue(value=1.0)
    assert call_args["rationale"] == "This answer is very faithful."
    assert call_args["metadata"] == {"model": "gpt-4o-mini"}


def test_override_feedback(store):
    # Mock the store's get_assessment method to return a feedback
    old_feedback = Feedback(
        trace_id="tr-321",
        name="faithfulness",
        value=0.5,
        source=_LLM_ASSESSMENT_SOURCE,
        rationale="Original feedback",
        metadata={"model": "gpt-3.5"},
    )
    old_feedback.assessment_id = "a-1234"
    store.get_assessment.return_value = old_feedback

    mlflow.override_feedback(
        trace_id="tr-321",
        assessment_id="a-1234",
        value=1.0,
        source=_LLM_ASSESSMENT_SOURCE,
        rationale="This answer is very faithful.",
        metadata={"model": "gpt-4o-mini"},
    )

    # assert that the old feedback is fetched
    store.get_assessment.assert_called_once_with("tr-321", "a-1234")

    # assert that the new feedback is created
    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.trace_id == "tr-321"
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "tr-321"
    assert assessment.span_id is None
    assert assessment.source == _LLM_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.feedback.value == 1.0
    assert assessment.feedback.error is None
    assert assessment.expectation is None
    assert assessment.rationale == "This answer is very faithful."
    assert assessment.metadata == {"model": "gpt-4o-mini"}
    assert assessment.overrides == "a-1234"


def test_delete_assessment(store):
    mlflow.delete_assessment(trace_id="tr-1234", assessment_id="1234")

    assert store.delete_assessment.call_count == 1
    call_args = store.delete_assessment.call_args[1]
    assert call_args["assessment_id"] == "1234"
    assert call_args["trace_id"] == "tr-1234"


def test_search_traces_with_assessments(store):
    # Create a trace info with an assessment
    assessment = Feedback(
        trace_id="test",
        name="test",
        value="test",
        source=AssessmentSource(source_id="test", source_type=AssessmentSourceType.HUMAN),
        create_time_ms=0,
        last_update_time_ms=0,
    )

    trace_info = TraceInfo(
        trace_id="test",
        trace_location=TraceLocation.from_experiment_id("test"),
        request_time=0,
        execution_duration=0,
        state=TraceState.OK,
        tags={"mlflow.artifactLocation": "test"},
        assessments=[assessment],  # Include the assessment here
    )

    # Mock the search_traces to return our trace_info
    store.search_traces.return_value = ([trace_info, trace_info], None)

    # Now when search_traces is called, it should use our trace_info with the assessment
    with mock.patch(
        "mlflow.tracing.client.TracingClient._download_trace_data", return_value=TraceData()
    ):
        res = mlflow.search_traces(
            experiment_ids=["0"],
            max_results=2,
            return_type="list",
        )

    # Verify the results
    assert len(res) == 2
    for trace in res:
        assert trace.info.assessments is not None
        assert len(trace.info.assessments) == 1
        assert trace.info.assessments[0].trace_id == "test"
        assert trace.info.assessments[0].name == "test"

    # Verify the search_traces was called
    assert store.search_traces.call_count == 1

    # We no longer expect get_trace_info to be called
    assert store.get_trace_info.call_count == 0


def test_log_feedback_default_source(store):
    # Test that the default CODE source is used when no source is provided
    feedback = Feedback(
        trace_id="1234",
        name="faithfulness",
        value=1.0,
    )
    mlflow.log_assessment(trace_id="1234", assessment=feedback)

    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.source.source_type == AssessmentSourceType.CODE
    assert assessment.source.source_id == "default"
    assert assessment.feedback.value == 1.0


def test_log_expectation_default_source(store):
    # Test that the default HUMAN source is used when no source is provided
    expectation = Expectation(
        trace_id="1234",
        name="expected_answer",
        value="MLflow",
    )
    mlflow.log_assessment(trace_id="1234", assessment=expectation)

    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.name == "expected_answer"
    assert assessment.trace_id == "1234"
    assert assessment.source.source_type == AssessmentSourceType.HUMAN
    assert assessment.source.source_id == "default"
    assert assessment.expectation.value == "MLflow"


def test_log_feedback_and_exception_blocks_positional_args():
    with pytest.raises(TypeError, match=r"log_feedback\(\) takes 0 positional"):
        mlflow.log_feedback("tr-1234", "faithfulness", 1.0)

    with pytest.raises(TypeError, match=r"log_expectation\(\) takes 0 positional"):
        mlflow.log_expectation("tr-1234", "expected_answer", "MLflow")


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_assessment_on_in_progress_trace(store, legacy_api):
    @mlflow.trace
    def func(x: int, y: int) -> int:
        trace_id = mlflow.get_active_trace_id()
        if legacy_api:
            mlflow.log_assessment(trace_id, Feedback(name="feedback", value=1.0))
            mlflow.log_assessment(trace_id, Expectation(name="expectation", value="MLflow"))
            mlflow.log_assessment("other_trace_id", Feedback(name="other", value=2.0))
        else:
            mlflow.log_feedback(trace_id=trace_id, name="feedback", value=1.0)
            mlflow.log_expectation(trace_id=trace_id, name="expectation", value="MLflow")
            mlflow.log_feedback(trace_id="other_trace_id", name="other", value=2.0)
        return x + y

    assert func(1, 2) == 3

    mlflow.flush_trace_async_logging()

    # Two assessments should be logged as a part of StartTraceV3 call
    store.start_trace.assert_called_once()
    trace_info = store.start_trace.call_args[1]["trace_info"]
    assert trace_info.request_id == mlflow.get_last_active_trace_id()
    assert len(trace_info.assessments) == 2
    assert trace_info.assessments[0].name == "feedback"
    assert trace_info.assessments[0].feedback.value == 1.0
    assert trace_info.assessments[1].name == "expectation"
    assert trace_info.assessments[1].expectation.value == "MLflow"

    # CreateAssessment should be called for the assessment on the other trace (both V2 and V3)
    store.create_assessment.assert_called_once()
    call_args = store.create_assessment.call_args
    assessment = call_args[0][0]

    assert assessment.trace_id == "other_trace_id"
    assert assessment.name == "other"
    assert assessment.feedback.value == 2.0


@pytest.mark.asyncio
async def test_log_assessment_on_in_progress_trace_async(store):
    @mlflow.trace
    async def func(x: int, y: int) -> int:
        trace_id = mlflow.get_active_trace_id()
        mlflow.log_assessment(trace_id, Feedback(name="feedback", value=1.0))
        mlflow.log_assessment(trace_id, Expectation(name="expectation", value="MLflow"))
        return x + y

    assert (await func(1, 2)) == 3

    mlflow.flush_trace_async_logging()

    store.create_assessment.assert_not_called()

    store.start_trace.assert_called_once()
    trace_info = store.start_trace.call_args[1]["trace_info"]
    assert trace_info.request_id == mlflow.get_last_active_trace_id()
    assert len(trace_info.assessments) == 2
    assert trace_info.assessments[0].name == "feedback"
    assert trace_info.assessments[0].feedback.value == 1.0
    assert trace_info.assessments[1].name == "expectation"
    assert trace_info.assessments[1].expectation.value == "MLflow"


def test_log_assessment_on_in_progress_with_span_id(store):
    with mlflow.start_span(name="test_span") as span:
        # Only proceed if we have a real span (not NO_OP)
        if span.span_id is not None and span.trace_id != "MLFLOW_NO_OP_SPAN_TRACE_ID":
            mlflow.log_assessment(
                trace_id=span.trace_id,
                assessment=Feedback(name="feedback", value=1.0, span_id=span.span_id),
            )

    mlflow.flush_trace_async_logging()

    # Two assessments should be logged as a part of StartTraceV3 call
    store.start_trace.assert_called_once()
    trace_info = store.start_trace.call_args[1]["trace_info"]
    assert trace_info.request_id == mlflow.get_last_active_trace_id()
    assert len(trace_info.assessments) == 1
    assert trace_info.assessments[0].name == "feedback"
    assert trace_info.assessments[0].feedback.value == 1.0
    assert trace_info.assessments[0].span_id == span.span_id

    store.create_assessment.assert_not_called()


def test_log_assessment_on_in_progress_trace_works_when_tracing_is_disabled(store):
    # Calling log_assessment to an active trace should not fail when tracing is disabled.
    mlflow.tracing.disable()

    @mlflow.trace
    def func(x: int, y: int):
        trace_id = mlflow.get_active_trace_id()
        mlflow.log_assessment(trace_id=trace_id, assessment=Feedback(name="feedback", value=1.0))
        return x + y

    assert func(1, 2) == 3

    mlflow.flush_trace_async_logging()

    # Neither CreateAssessment nor StartTraceV3 should be called
    store.create_assessment.assert_not_called()
    store.start_trace.assert_not_called()


def test_get_assessment(store):
    """Test get_assessment calls store correctly"""
    mock_assessment = mock.Mock()
    store.get_assessment.return_value = mock_assessment

    result = mlflow.get_assessment("trace_123", "assessment_456")

    store.get_assessment.assert_called_once_with("trace_123", "assessment_456")
    assert result == mock_assessment


def test_assessment_end_to_end_workflow(store):
    """Test complete assessment workflow"""
    # Mock assessment for override test
    original_feedback = Feedback(
        name="quality", value=0.6, source=_LLM_ASSESSMENT_SOURCE, rationale="Original assessment"
    )
    original_feedback.assessment_id = "original_id"
    store.get_assessment.return_value = original_feedback

    # Test creating initial assessment
    initial_assessment = Feedback(
        name="quality",
        value=0.8,
        source=_HUMAN_ASSESSMENT_SOURCE,
        rationale="Good response quality",
    )

    mlflow.log_assessment(trace_id="tr-123", assessment=initial_assessment)

    # Verify creation
    assert store.create_assessment.call_count == 1
    call_args = store.create_assessment.call_args
    assert call_args[0][0].trace_id == "tr-123"
    assert call_args[0][0].value == 0.8

    # Test updating assessment
    updated_assessment = Feedback(
        name="quality", value=0.9, source=_HUMAN_ASSESSMENT_SOURCE, rationale="Updated assessment"
    )

    mlflow.update_assessment(
        trace_id="tr-123", assessment_id="test_id", assessment=updated_assessment
    )

    # Verify update
    assert store.update_assessment.call_count == 1
    update_args = store.update_assessment.call_args[1]
    assert update_args["trace_id"] == "tr-123"
    assert update_args["assessment_id"] == "test_id"
    assert update_args["feedback"].value == 0.9

    # Test override functionality
    mlflow.override_feedback(
        trace_id="tr-123",
        assessment_id="original_id",
        value=0.95,
        rationale="Human override of LLM assessment",
    )

    assert store.create_assessment.call_count == 2  # Initial + override
    override_call = store.create_assessment.call_args
    override_assessment = override_call[0][0]
    assert override_assessment.overrides == "original_id"
    assert override_assessment.value == 0.95

    mlflow.delete_assessment(trace_id="tr-123", assessment_id="test_id")

    assert store.delete_assessment.call_count == 1
    delete_args = store.delete_assessment.call_args[1]
    assert delete_args["trace_id"] == "tr-123"
    assert delete_args["assessment_id"] == "test_id"
