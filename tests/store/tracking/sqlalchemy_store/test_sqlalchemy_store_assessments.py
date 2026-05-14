import time
import uuid

import pytest

from mlflow.entities import (
    AssessmentSource,
    AssessmentSourceType,
    Expectation,
    Feedback,
    trace_location,
)
from mlflow.entities.assessment import ExpectationValue, FeedbackValue
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.utils.time import get_current_time_millis

pytestmark = pytest.mark.notrackingurimock


def test_create_and_get_assessment(store_and_trace_info):
    store, trace_info = store_and_trace_info

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="correctness",
        value=True,
        rationale="The response is correct and well-formatted",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="evaluator@company.com"
        ),
        metadata={"project": "test-project", "version": "1.0"},
        span_id="span-123",
    )

    created_feedback = store.create_assessment(feedback)
    assert created_feedback.assessment_id is not None
    assert created_feedback.assessment_id.startswith("a-")
    assert created_feedback.trace_id == trace_info.request_id
    assert created_feedback.create_time_ms is not None
    assert created_feedback.name == "correctness"
    assert created_feedback.value is True
    assert created_feedback.rationale == "The response is correct and well-formatted"
    assert created_feedback.metadata == {"project": "test-project", "version": "1.0"}
    assert created_feedback.span_id == "span-123"
    assert created_feedback.valid

    expectation = Expectation(
        trace_id=trace_info.request_id,
        name="expected_response",
        value="The capital of France is Paris.",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="annotator@company.com"
        ),
        metadata={"context": "geography-qa", "difficulty": "easy"},
        span_id="span-456",
    )

    created_expectation = store.create_assessment(expectation)
    assert created_expectation.assessment_id != created_feedback.assessment_id
    assert created_expectation.trace_id == trace_info.request_id
    assert created_expectation.value == "The capital of France is Paris."
    assert created_expectation.metadata == {"context": "geography-qa", "difficulty": "easy"}
    assert created_expectation.span_id == "span-456"
    assert created_expectation.valid

    retrieved_feedback = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved_feedback.name == "correctness"
    assert retrieved_feedback.value is True
    assert retrieved_feedback.rationale == "The response is correct and well-formatted"
    assert retrieved_feedback.metadata == {"project": "test-project", "version": "1.0"}
    assert retrieved_feedback.span_id == "span-123"
    assert retrieved_feedback.trace_id == trace_info.request_id
    assert retrieved_feedback.valid

    retrieved_expectation = store.get_assessment(
        trace_info.request_id, created_expectation.assessment_id
    )
    assert retrieved_expectation.value == "The capital of France is Paris."
    assert retrieved_expectation.metadata == {"context": "geography-qa", "difficulty": "easy"}
    assert retrieved_expectation.span_id == "span-456"
    assert retrieved_expectation.trace_id == trace_info.request_id
    assert retrieved_expectation.valid


def test_get_assessment_errors(store_and_trace_info):
    store, trace_info = store_and_trace_info

    with pytest.raises(MlflowException, match=r"Trace with (ID|request_id) 'fake_trace' not found"):
        store.get_assessment("fake_trace", "fake_assessment")

    with pytest.raises(
        MlflowException,
        match=r"Assessment with ID 'fake_assessment' not found for trace",
    ):
        store.get_assessment(trace_info.request_id, "fake_assessment")


def test_update_assessment_feedback(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="correctness",
        value=True,
        rationale="Original rationale",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="evaluator@company.com"
        ),
        metadata={"project": "test-project", "version": "1.0"},
        span_id="span-123",
    )

    created_feedback = store.create_assessment(original_feedback)
    original_id = created_feedback.assessment_id

    updated_feedback = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=original_id,
        name="correctness_updated",
        feedback=FeedbackValue(value=False),
        rationale="Updated rationale",
        metadata={"project": "test-project", "version": "2.0", "new_field": "added"},
    )

    assert updated_feedback.assessment_id == original_id
    assert updated_feedback.name == "correctness_updated"
    assert updated_feedback.value is False
    assert updated_feedback.rationale == "Updated rationale"
    assert updated_feedback.metadata == {
        "project": "test-project",
        "version": "2.0",
        "new_field": "added",
    }
    assert updated_feedback.span_id == "span-123"
    assert updated_feedback.source.source_id == "evaluator@company.com"
    assert updated_feedback.valid is True

    retrieved = store.get_assessment(trace_info.request_id, original_id)
    assert retrieved.value is False
    assert retrieved.name == "correctness_updated"
    assert retrieved.rationale == "Updated rationale"


def test_update_assessment_expectation(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_expectation = Expectation(
        trace_id=trace_info.request_id,
        name="expected_response",
        value="The capital of France is Paris.",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN, source_id="annotator@company.com"
        ),
        metadata={"context": "geography-qa"},
        span_id="span-456",
    )

    created_expectation = store.create_assessment(original_expectation)
    original_id = created_expectation.assessment_id

    updated_expectation = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=original_id,
        expectation=ExpectationValue(value="The capital and largest city of France is Paris."),
        metadata={"context": "geography-qa", "updated": "true"},
    )

    assert updated_expectation.assessment_id == original_id
    assert updated_expectation.name == "expected_response"
    assert updated_expectation.value == "The capital and largest city of France is Paris."
    assert updated_expectation.metadata == {"context": "geography-qa", "updated": "true"}
    assert updated_expectation.span_id == "span-456"
    assert updated_expectation.source.source_id == "annotator@company.com"


def test_update_assessment_partial_fields(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value=5,
        rationale="Original rationale",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        metadata={"scorer": "automated"},
    )

    created_feedback = store.create_assessment(original_feedback)
    original_id = created_feedback.assessment_id

    updated_feedback = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=original_id,
        rationale="Updated rationale only",
    )

    assert updated_feedback.assessment_id == original_id
    assert updated_feedback.name == "quality"
    assert updated_feedback.value == 5
    assert updated_feedback.rationale == "Updated rationale only"
    assert updated_feedback.metadata == {"scorer": "automated"}


def test_update_assessment_type_validation(store_and_trace_info):
    store, trace_info = store_and_trace_info

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="test_feedback",
        value="original",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )
    created_feedback = store.create_assessment(feedback)

    with pytest.raises(
        MlflowException, match=r"Cannot update expectation value on a Feedback assessment"
    ):
        store.update_assessment(
            trace_id=trace_info.request_id,
            assessment_id=created_feedback.assessment_id,
            expectation=ExpectationValue(value="This should fail"),
        )

    expectation = Expectation(
        trace_id=trace_info.request_id,
        name="test_expectation",
        value="original_expected",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    created_expectation = store.create_assessment(expectation)

    with pytest.raises(
        MlflowException, match=r"Cannot update feedback value on an Expectation assessment"
    ):
        store.update_assessment(
            trace_id=trace_info.request_id,
            assessment_id=created_expectation.assessment_id,
            feedback=FeedbackValue(value="This should fail"),
        )


def test_update_assessment_errors(store_and_trace_info):
    store, trace_info = store_and_trace_info

    with pytest.raises(MlflowException, match=r"Trace with (ID|request_id) 'fake_trace' not found"):
        store.update_assessment(
            trace_id="fake_trace", assessment_id="fake_assessment", rationale="This should fail"
        )

    with pytest.raises(
        MlflowException,
        match=r"Assessment with ID 'fake_assessment' not found for trace",
    ):
        store.update_assessment(
            trace_id=trace_info.request_id,
            assessment_id="fake_assessment",
            rationale="This should fail",
        )


def test_update_assessment_metadata_merging(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original = Feedback(
        trace_id=trace_info.request_id,
        name="test",
        value="original",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        metadata={"keep": "this", "override": "old_value", "remove_me": "will_stay"},
    )

    created = store.create_assessment(original)

    updated = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=created.assessment_id,
        metadata={"override": "new_value", "new_key": "new_value"},
    )

    expected_metadata = {
        "keep": "this",
        "override": "new_value",
        "remove_me": "will_stay",
        "new_key": "new_value",
    }
    assert updated.metadata == expected_metadata


def test_update_assessment_timestamps(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original = Feedback(
        trace_id=trace_info.request_id,
        name="test",
        value="original",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )

    created = store.create_assessment(original)
    original_create_time = created.create_time_ms
    original_update_time = created.last_update_time_ms

    time.sleep(0.001)

    updated = store.update_assessment(
        trace_id=trace_info.request_id,
        assessment_id=created.assessment_id,
        name="updated_name",
    )

    assert updated.create_time_ms == original_create_time
    assert updated.last_update_time_ms > original_update_time


def test_create_assessment_with_overrides(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value="poor",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
    )

    created_original = store.create_assessment(original_feedback)

    override_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value="excellent",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        overrides=created_original.assessment_id,
    )

    created_override = store.create_assessment(override_feedback)

    assert created_override.overrides == created_original.assessment_id
    assert created_override.value == "excellent"
    assert created_override.valid is True

    retrieved_original = store.get_assessment(trace_info.request_id, created_original.assessment_id)
    assert retrieved_original.valid is False
    assert retrieved_original.value == "poor"


def test_create_assessment_override_nonexistent(store_and_trace_info):
    store, trace_info = store_and_trace_info

    override_feedback = Feedback(
        trace_id=trace_info.request_id,
        name="quality",
        value="excellent",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        overrides="nonexistent-assessment-id",
    )

    with pytest.raises(
        MlflowException, match=r"Assessment with ID 'nonexistent-assessment-id' not found"
    ):
        store.create_assessment(override_feedback)


def test_delete_assessment_idempotent(store_and_trace_info):
    store, trace_info = store_and_trace_info

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="test",
        value="test_value",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )

    created_feedback = store.create_assessment(feedback)

    retrieved = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved.assessment_id == created_feedback.assessment_id

    store.delete_assessment(trace_info.request_id, created_feedback.assessment_id)

    with pytest.raises(
        MlflowException,
        match=rf"Assessment with ID '{created_feedback.assessment_id}' not found for trace",
    ):
        store.get_assessment(trace_info.request_id, created_feedback.assessment_id)

    store.delete_assessment(trace_info.request_id, created_feedback.assessment_id)
    store.delete_assessment(trace_info.request_id, "fake_assessment_id")


def test_delete_assessment_override_behavior(store_and_trace_info):
    store, trace_info = store_and_trace_info

    original = store.create_assessment(
        Feedback(
            trace_id=trace_info.request_id,
            name="original",
            value="original_value",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        ),
    )

    override = store.create_assessment(
        Feedback(
            trace_id=trace_info.request_id,
            name="override",
            value="override_value",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
            overrides=original.assessment_id,
        ),
    )

    assert store.get_assessment(trace_info.request_id, original.assessment_id).valid is False
    assert store.get_assessment(trace_info.request_id, override.assessment_id).valid is True

    store.delete_assessment(trace_info.request_id, override.assessment_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_assessment(trace_info.request_id, override.assessment_id)
    assert store.get_assessment(trace_info.request_id, original.assessment_id).valid is True


def test_assessment_with_run_id(store_and_trace_info):
    store, trace_info = store_and_trace_info

    run = store.create_run(
        experiment_id=trace_info.experiment_id,
        user_id="test_user",
        start_time=get_current_time_millis(),
        tags=[],
        run_name="test_run",
    )

    feedback = Feedback(
        trace_id=trace_info.request_id,
        name="run_feedback",
        value="excellent",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )
    feedback.run_id = run.info.run_id

    created_feedback = store.create_assessment(feedback)
    assert created_feedback.run_id == run.info.run_id

    retrieved_feedback = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved_feedback.run_id == run.info.run_id


def test_assessment_with_error(store_and_trace_info):
    store, trace_info = store_and_trace_info

    try:
        raise ValueError("Test error message")
    except ValueError as test_error:
        feedback = Feedback(
            trace_id=trace_info.request_id,
            name="error_feedback",
            value=None,
            error=test_error,
            source=AssessmentSource(source_type=AssessmentSourceType.CODE),
        )

    created_feedback = store.create_assessment(feedback)
    assert created_feedback.error.error_message == "Test error message"
    assert created_feedback.error.error_code == "ValueError"
    assert created_feedback.error.stack_trace is not None
    assert "ValueError: Test error message" in created_feedback.error.stack_trace
    assert "test_assessment_with_error" in created_feedback.error.stack_trace

    retrieved_feedback = store.get_assessment(trace_info.request_id, created_feedback.assessment_id)
    assert retrieved_feedback.error.error_message == "Test error message"
    assert retrieved_feedback.error.error_code == "ValueError"
    assert retrieved_feedback.error.stack_trace is not None
    assert "ValueError: Test error message" in retrieved_feedback.error.stack_trace
    assert created_feedback.error.stack_trace == retrieved_feedback.error.stack_trace


def test_start_trace_with_assessments_missing_trace_id(store):
    """
    Regression test for NOT NULL constraint on assessments.trace_id during trace export.

    During normal trace export (MlflowV3SpanExporter), two things happen:

    1. log_spans() is called incrementally as each span completes. Internally this calls
       start_trace(), creating the trace row in the DB.
    2. When the root span finishes, _log_trace() calls start_trace() again with the full
       TraceInfo — including any assessments attached to the trace.

    Because the trace row already exists from step 1, the second start_trace() hits an
    IntegrityError and falls back to session.merge(). Assessments created standalone
    (e.g. returned by custom metric functions) have trace_id=None by design. Without
    backfilling trace_id before the merge, SQLAlchemy updates the assessment row with
    trace_id=NULL, violating the NOT NULL constraint on assessments.trace_id.
    """
    exp_id = store.create_experiment("test_assessment_trace_id")
    timestamp_ms = get_current_time_millis()
    trace_id = f"tr-{uuid.uuid4()}"

    # Step 1: log_spans() creates the trace row as spans are exported incrementally.
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=0,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"cr-{uuid.uuid4()}",
            request_preview=None,
            response_preview=None,
        ),
    )

    # Assessment with trace_id=None, as returned by custom metric functions.
    assessment = Feedback(
        name="test_feedback",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1"),
        trace_id=None,
        value="good",
    )

    # Step 2: _log_trace() calls start_trace() with the full TraceInfo (including
    # assessments) after the root span finishes. The trace already exists from step 1,
    # so this hits the IntegrityError -> session.merge() path. Before the fix, this
    # raised sqlite3.IntegrityError because assessment.trace_id was None.
    result = store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
            request_time=timestamp_ms,
            execution_duration=100,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"cr-{uuid.uuid4()}",
            request_preview="request",
            response_preview="response",
            assessments=[assessment],
        ),
    )

    assert len(result.assessments) == 1
    assert result.assessments[0].trace_id == trace_id
    assert result.assessments[0].name == "test_feedback"
