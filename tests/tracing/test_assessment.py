import os
import sys

import pytest

import mlflow
from mlflow.entities.assessment import (
    AssessmentError,
    AssessmentSource,
    Expectation,
    Feedback,
)
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException

_HUMAN_ASSESSMENT_SOURCE = AssessmentSource(
    source_type=AssessmentSourceType.HUMAN,
    source_id="bob@example.com",
)

_LLM_ASSESSMENT_SOURCE = AssessmentSource(
    source_type=AssessmentSourceType.LLM_JUDGE,
    source_id="gpt-4o-mini",
)


@pytest.fixture
def trace_id():
    with mlflow.start_span(name="test_span") as span:
        pass

    return span.trace_id


@pytest.fixture(params=["file", "sqlalchemy"], autouse=True)
def tracking_uri(request, tmp_path):
    """Set an MLflow Tracking URI with different type of backend."""
    if "MLFLOW_SKINNY" in os.environ and request.param == "sqlalchemy":
        pytest.skip("SQLAlchemy store is not available in skinny.")

    original_tracking_uri = mlflow.get_tracking_uri()

    if request.param == "file":
        tracking_uri = tmp_path.joinpath("file").as_uri()
    elif request.param == "sqlalchemy":
        path = tmp_path.joinpath("sqlalchemy.db").as_uri()
        tracking_uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[
            len("file://") :
        ]

    # NB: MLflow tracer does not handle the change of tracking URI well,
    # so we need to reset the tracer to switch the tracking URI during testing.
    mlflow.tracing.disable()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.tracing.enable()

    yield tracking_uri

    # Reset tracking URI
    mlflow.set_tracking_uri(original_tracking_uri)


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_expectation(trace_id, legacy_api):
    if legacy_api:
        mlflow.log_expectation(
            trace_id=trace_id,
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
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert isinstance(assessment, Expectation)
    assert assessment.trace_id == trace_id
    assert assessment.name == "expected_answer"
    assert assessment.value == "MLflow"
    assert assessment.trace_id == trace_id
    assert assessment.span_id is None
    assert assessment.source == _HUMAN_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation.value == "MLflow"
    assert assessment.rationale is None
    assert assessment.metadata == {"key": "value"}


def test_log_expectation_invalid_parameters():
    with pytest.raises(MlflowException, match=r"The `value` field must be specified."):
        Expectation(
            name="expected_answer",
            value=None,
            source=_HUMAN_ASSESSMENT_SOURCE,
        )


def test_update_expectation(trace_id):
    assessment_id = mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_answer",
        value="MLflow",
    ).assessment_id

    updated_assessment = Expectation(
        name="expected_answer",
        value="Spark",
        metadata={"reason": "human override"},
    )

    mlflow.update_assessment(
        assessment_id=assessment_id,
        trace_id=trace_id,
        assessment=updated_assessment,
    )

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.trace_id == trace_id
    assert assessment.name == "expected_answer"
    assert assessment.expectation.value == "Spark"
    assert assessment.feedback is None
    assert assessment.rationale is None
    assert assessment.metadata == {"reason": "human override"}


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_feedback(trace_id, legacy_api):
    if legacy_api:
        mlflow.log_feedback(
            trace_id=trace_id,
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
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert isinstance(assessment, Feedback)
    assert assessment.trace_id == trace_id
    assert assessment.name == "faithfulness"
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
def test_log_feedback_with_error(trace_id, legacy_api):
    if legacy_api:
        mlflow.log_feedback(
            trace_id=trace_id,
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
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == trace_id
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
def test_log_feedback_with_exception_object(trace_id, legacy_api):
    """Test that log_feedback correctly accepts Exception objects."""
    test_exception = ValueError("Test exception message")

    if legacy_api:
        mlflow.log_feedback(
            trace_id=trace_id,
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
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == trace_id
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
def test_log_feedback_with_value_and_error(trace_id, legacy_api):
    if legacy_api:
        mlflow.log_feedback(
            trace_id=trace_id,
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
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == trace_id
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


def test_update_feedback(trace_id):
    assessment_id = mlflow.log_feedback(
        trace_id=trace_id,
        name="faithfulness",
        value=1.0,
        rationale="This answer is very faithful.",
        metadata={"model": "gpt-4o-mini"},
    ).assessment_id

    updated_feedback = Feedback(
        name="faithfulness",
        value=0,
        rationale="This answer is not faithful.",
        metadata={"reason": "human override"},
    )
    mlflow.update_assessment(
        assessment_id=assessment_id,
        trace_id=trace_id,
        assessment=updated_feedback,
    )

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == trace_id
    assert assessment.feedback.value == 0
    assert assessment.feedback.error is None
    assert assessment.rationale == "This answer is not faithful."
    assert assessment.metadata == {
        "model": "gpt-4o-mini",
        "reason": "human override",
    }


def test_override_feedback(trace_id):
    assessment_id = mlflow.log_feedback(
        trace_id=trace_id,
        name="faithfulness",
        value=0.5,
        source=_LLM_ASSESSMENT_SOURCE,
        rationale="Original feedback",
        metadata={"model": "gpt-3.5"},
    ).assessment_id

    new_assessment_id = mlflow.override_feedback(
        trace_id=trace_id,
        assessment_id=assessment_id,
        value=1.0,
        source=_LLM_ASSESSMENT_SOURCE,
        rationale="This answer is very faithful.",
        metadata={"model": "gpt-4o-mini"},
    ).assessment_id

    # New assessment should have the same trace_id as the original assessment
    assessment = mlflow.get_assessment(trace_id, new_assessment_id)
    assert assessment.trace_id == trace_id
    assert assessment.name == "faithfulness"
    assert assessment.span_id is None
    assert assessment.source == _LLM_ASSESSMENT_SOURCE
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.value == 1.0
    assert assessment.error is None
    assert assessment.rationale == "This answer is very faithful."
    assert assessment.metadata == {"model": "gpt-4o-mini"}
    assert assessment.overrides == assessment_id
    assert assessment.valid is True

    # Original assessment should be invalidated
    original_assessment = mlflow.get_assessment(trace_id, assessment_id)
    assert original_assessment.valid is False
    assert original_assessment.feedback.value == 0.5


def test_delete_assessment(trace_id):
    assessment_id = mlflow.log_feedback(
        trace_id=trace_id,
        name="faithfulness",
        value=1.0,
    ).assessment_id

    mlflow.delete_assessment(trace_id=trace_id, assessment_id=assessment_id)

    with pytest.raises(MlflowException, match=r"Assessment with ID"):
        assert mlflow.get_assessment(trace_id, assessment_id) is None

    # Assessment should be deleted from the trace
    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 0


def test_log_feedback_default_source(trace_id):
    # Test that the default CODE source is used when no source is provided
    feedback = Feedback(
        trace_id=trace_id,
        name="faithfulness",
        value=1.0,
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == trace_id
    assert assessment.source.source_type == AssessmentSourceType.CODE
    assert assessment.source.source_id == "default"
    assert assessment.feedback.value == 1.0


def test_log_expectation_default_source(trace_id):
    # Test that the default HUMAN source is used when no source is provided
    expectation = Expectation(
        trace_id=trace_id,
        name="expected_answer",
        value="MLflow",
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=expectation)

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.name == "expected_answer"
    assert assessment.trace_id == trace_id
    assert assessment.source.source_type == AssessmentSourceType.HUMAN
    assert assessment.source.source_id == "default"
    assert assessment.expectation.value == "MLflow"


def test_log_feedback_and_exception_blocks_positional_args():
    with pytest.raises(TypeError, match=r"log_feedback\(\) takes 0 positional"):
        mlflow.log_feedback("tr-1234", "faithfulness", 1.0)

    with pytest.raises(TypeError, match=r"log_expectation\(\) takes 0 positional"):
        mlflow.log_expectation("tr-1234", "expected_answer", "MLflow")


@pytest.mark.parametrize("legacy_api", [True, False])
def test_log_assessment_on_in_progress_trace(trace_id, legacy_api):
    @mlflow.trace
    def func(x: int, y: int) -> int:
        active_trace_id = mlflow.get_active_trace_id()
        if legacy_api:
            mlflow.log_assessment(active_trace_id, Feedback(name="feedback", value=1.0))
            mlflow.log_assessment(active_trace_id, Expectation(name="expectation", value="MLflow"))
            mlflow.log_assessment(trace_id, Feedback(name="other", value=2.0))
        else:
            mlflow.log_feedback(trace_id=active_trace_id, name="feedback", value=1.0)
            mlflow.log_expectation(trace_id=active_trace_id, name="expectation", value="MLflow")
            mlflow.log_feedback(trace_id=trace_id, name="other", value=2.0)
        return x + y

    assert func(1, 2) == 3

    # Two assessments should be logged as a part of StartTraceV3 call
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.info.assessments) == 2
    assessments = {a.name: a for a in trace.info.assessments}
    assert assessments["feedback"].value == 1.0
    assert assessments["expectation"].value == "MLflow"

    # Assessment on the other trace
    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assert trace.info.assessments[0].name == "other"
    assert trace.info.assessments[0].feedback.value == 2.0


@pytest.mark.asyncio
async def test_log_assessment_on_in_progress_trace_async():
    @mlflow.trace
    async def func(x: int, y: int) -> int:
        trace_id = mlflow.get_active_trace_id()
        mlflow.log_assessment(trace_id, Feedback(name="feedback", value=1.0))
        mlflow.log_assessment(trace_id, Expectation(name="expectation", value="MLflow"))
        return x + y

    assert (await func(1, 2)) == 3

    mlflow.flush_trace_async_logging()

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_info = trace.info
    assert len(trace_info.assessments) == 2
    assessments = {a.name: a for a in trace_info.assessments}
    assert assessments["feedback"].feedback.value == 1.0
    assert assessments["expectation"].expectation.value == "MLflow"


def test_log_assessment_on_in_progress_with_span_id():
    with mlflow.start_span(name="test_span") as span:
        # Only proceed if we have a real span (not NO_OP)
        if span.span_id is not None and span.trace_id != "MLFLOW_NO_OP_SPAN_TRACE_ID":
            mlflow.log_assessment(
                trace_id=span.trace_id,
                assessment=Feedback(name="feedback", value=1.0, span_id=span.span_id),
            )

    mlflow.flush_trace_async_logging()

    # Two assessments should be logged as a part of StartTraceV3 call
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_info = trace.info
    assert len(trace_info.assessments) == 1
    assert trace_info.assessments[0].name == "feedback"
    assert trace_info.assessments[0].feedback.value == 1.0
    assert trace_info.assessments[0].span_id == span.span_id


def test_log_assessment_on_in_progress_trace_works_when_tracing_is_disabled():
    # Calling log_assessment to an active trace should not fail when tracing is disabled.
    mlflow.tracing.disable()

    @mlflow.trace
    def func(x: int, y: int):
        trace_id = mlflow.get_active_trace_id()
        mlflow.log_assessment(trace_id=trace_id, assessment=Feedback(name="feedback", value=1.0))
        return x + y

    assert func(1, 2) == 3

    mlflow.flush_trace_async_logging()


def test_get_assessment(trace_id):
    """Test get_assessment calls store correctly"""
    assessment_id = mlflow.log_feedback(
        trace_id=trace_id,
        name="faithfulness",
        value=1.0,
    ).assessment_id

    result = mlflow.get_assessment(trace_id, assessment_id)

    assert isinstance(result, Feedback)
    assert result.name == "faithfulness"
    assert result.trace_id == trace_id
    assert result.value == 1.0
    assert result.error is None
    assert result.source.source_type == AssessmentSourceType.CODE
    assert result.source.source_id == "default"
    assert result.create_time_ms is not None
    assert result.last_update_time_ms is not None
    assert result.valid is True
    assert result.overrides is None


def test_search_traces_with_assessments():
    # Create traces with assessments
    with mlflow.start_span(name="trace_1") as span_1:
        mlflow.log_feedback(
            trace_id=span_1.trace_id,
            name="feedback_1",
            value=1.0,
        )
        mlflow.log_expectation(
            trace_id=span_1.trace_id,
            name="expectation_1",
            value="test",
            source=AssessmentSource(source_id="test", source_type=AssessmentSourceType.LLM_JUDGE),
        )
        with mlflow.start_span(name="child") as span_1_child:
            mlflow.log_feedback(
                trace_id=span_1_child.trace_id,
                name="feedback_2",
                value=1.0,
                span_id=span_1_child.span_id,
            )

    with mlflow.start_span(name="trace_2") as span_2:
        mlflow.log_feedback(
            trace_id=span_2.trace_id,
            name="feedback_3",
            value=1.0,
        )

    traces = mlflow.search_traces(
        experiment_ids=["0"],
        max_results=2,
        return_type="list",
        order_by=["timestamp_ms"],
    )
    # Verify the results
    assert len(traces) == 2
    assert len(traces[0].info.assessments) == 3

    assessments = {a.name: a for a in traces[0].info.assessments}
    assert assessments["feedback_1"].trace_id == span_1.trace_id
    assert assessments["feedback_1"].name == "feedback_1"
    assert assessments["feedback_1"].value == 1.0
    assert assessments["expectation_1"].trace_id == span_1.trace_id
    assert assessments["expectation_1"].name == "expectation_1"
    assert assessments["expectation_1"].value == "test"
    assert assessments["feedback_2"].trace_id == span_1_child.trace_id
    assert assessments["feedback_2"].name == "feedback_2"
    assert assessments["feedback_2"].value == 1.0

    assert len(traces[1].info.assessments) == 1
    assessment = traces[1].info.assessments[0]
    assert assessment.trace_id == span_2.trace_id
    assert assessment.name == "feedback_3"
    assert assessment.value == 1.0


@pytest.mark.parametrize("source_type", ["AI_JUDGE", AssessmentSourceType.AI_JUDGE])
def test_log_feedback_ai_judge_deprecation_warning(trace_id, source_type):
    with pytest.warns(FutureWarning, match="AI_JUDGE is deprecated. Use LLM_JUDGE instead."):
        ai_judge_source = AssessmentSource(source_type=source_type, source_id="gpt-4")

    mlflow.log_feedback(
        trace_id=trace_id,
        name="quality",
        value=0.8,
        source=ai_judge_source,
        rationale="AI evaluation",
    )

    trace = mlflow.get_trace(trace_id)
    assert len(trace.info.assessments) == 1
    assessment = trace.info.assessments[0]
    assert assessment.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert assessment.source.source_id == "gpt-4"
    assert assessment.name == "quality"
    assert assessment.feedback.value == 0.8
    assert assessment.rationale == "AI evaluation"
