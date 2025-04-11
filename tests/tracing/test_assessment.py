from unittest import mock

import pytest

import mlflow
from mlflow.entities.assessment import Assessment, AssessmentError, Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.fluent import TRACE_BUFFER


# TODO: This test mocks out the tracking client and only test if the fluent API implementation
# passes the correct arguments to the low-level client. Once the OSS backend is implemented,
# we should also test the end-to-end assessment CRUD functionality.
@pytest.fixture
def store():
    mock_store = mock.MagicMock()
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        mock_get_store.return_value = mock_store
        yield mock_store


def test_log_expectation(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.log_expectation(
        trace_id="1234",
        name="expected_answer",
        value="MLflow",
        source=AssessmentSourceType.HUMAN,
        metadata={"key": "value"},
    )

    assert store.create_assessment.call_count == 1
    assessment = store.create_assessment.call_args[0][0]
    assert assessment.name == "expected_answer"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source.source_type == "HUMAN"
    assert assessment.source.source_id is None
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation.value == "MLflow"
    assert assessment.feedback is None
    assert assessment.rationale is None
    assert assessment.metadata == {"key": "value"}
    assert assessment.error is None


def test_log_expectation_invalid_parameters():
    with pytest.raises(MlflowException, match=r"Expectation value cannot be None."):
        mlflow.log_expectation(
            trace_id="1234",
            name="expected_answer",
            value=None,
            source=AssessmentSourceType.HUMAN,
        )

    with pytest.raises(MlflowException, match=r"`source` must be provided."):
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source=None,
        )

    with pytest.raises(MlflowException, match=r"Invalid assessment source type"):
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source="INVALID_SOURCE_TYPE",
        )


def test_update_expectation(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.update_expectation(
        assessment_id="1234",
        trace_id="tr-1234",
        value="MLflow",
    )

    assert store.update_assessment.call_count == 1
    call_args = store.update_assessment.call_args[1]
    assert call_args["trace_id"] == "tr-1234"
    assert call_args["assessment_id"] == "1234"
    assert call_args["name"] is None
    assert call_args["expectation"] == Expectation(value="MLflow")
    assert call_args["feedback"] is None
    assert call_args["rationale"] is None
    assert call_args["metadata"] is None


def test_log_feedback(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.log_feedback(
        trace_id="1234",
        name="faithfulness",
        value=1.0,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="faithfulness-judge",
        ),
        rationale="This answer is very faithful.",
        metadata={"model": "gpt-4o-mini"},
    )

    assert store.create_assessment.call_count == 1
    assessment = store.create_assessment.call_args[0][0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source.source_type == "LLM_JUDGE"
    assert assessment.source.source_id == "faithfulness-judge"
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.feedback.value == 1.0
    assert assessment.expectation is None
    assert assessment.rationale == "This answer is very faithful."
    assert assessment.metadata == {"model": "gpt-4o-mini"}
    assert assessment.error is None


def test_log_feedback_with_error(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.log_feedback(
        trace_id="1234",
        name="faithfulness",
        source=AssessmentSourceType.LLM_JUDGE,
        error=AssessmentError(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Rate limit for the judge exceeded.",
        ),
    )

    assert store.create_assessment.call_count == 1
    assessment = store.create_assessment.call_args[0][0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source.source_type == "LLM_JUDGE"
    assert assessment.source.source_id is None
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation is None
    assert assessment.feedback.value is None
    assert assessment.rationale is None
    assert assessment.error.error_code == "RATE_LIMIT_EXCEEDED"
    assert assessment.error.error_message == "Rate limit for the judge exceeded."


def test_log_feedback_with_value_and_error(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.log_feedback(
        trace_id="1234",
        name="faithfulness",
        source=AssessmentSourceType.LLM_JUDGE,
        value=0.5,
        error=AssessmentError(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Rate limit for the judge exceeded.",
        ),
    )

    assert store.create_assessment.call_count == 1
    assessment = store.create_assessment.call_args[0][0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source.source_type == "LLM_JUDGE"
    assert assessment.source.source_id is None
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.expectation is None
    assert assessment.feedback == Feedback(value=0.5)
    assert assessment.rationale is None
    assert assessment.error.error_code == "RATE_LIMIT_EXCEEDED"
    assert assessment.error.error_message == "Rate limit for the judge exceeded."


def test_log_feedback_invalid_parameters():
    with pytest.raises(MlflowException, match=r"Either `value` or `error` must be provided."):
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=AssessmentSourceType.LLM_JUDGE,
        )

    with pytest.raises(MlflowException, match=r"`source` must be provided."):
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source=None,
        )


def test_update_feedback(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.update_feedback(
        assessment_id="1234",
        trace_id="tr-1234",
        value=1.0,
        rationale="This answer is very faithful.",
        metadata={"model": "gpt-4o-mini"},
    )

    assert store.update_assessment.call_count == 1
    call_args = store.update_assessment.call_args[1]
    assert call_args["trace_id"] == "tr-1234"
    assert call_args["assessment_id"] == "1234"
    assert call_args["name"] is None
    assert call_args["expectation"] is None
    assert call_args["feedback"] == Feedback(value=1.0)
    assert call_args["rationale"] == "This answer is very faithful."
    assert call_args["metadata"] == {"model": "gpt-4o-mini"}


def test_delete_expectation(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.delete_expectation(trace_id="tr-1234", assessment_id="1234")

    assert store.delete_assessment.call_count == 1
    call_args = store.delete_assessment.call_args[1]
    assert call_args["assessment_id"] == "1234"
    assert call_args["trace_id"] == "tr-1234"


def test_delete_feedback(store):
    mlflow.set_tracking_uri("databricks")

    mlflow.delete_feedback(trace_id="tr-5678", assessment_id="5678")

    assert store.delete_assessment.call_count == 1
    call_args = store.delete_assessment.call_args[1]
    assert call_args["assessment_id"] == "5678"
    assert call_args["trace_id"] == "tr-5678"


def test_assessment_apis_only_available_in_databricks():
    with pytest.raises(MlflowException, match=r"This API is currently only available"):
        mlflow.log_expectation(
            trace_id="1234",
            name="expected_answer",
            value="MLflow",
            source=AssessmentSourceType.HUMAN,
        )

    with pytest.raises(MlflowException, match=r"This API is currently only available"):
        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source=AssessmentSourceType.LLM_JUDGE,
        )

    with pytest.raises(MlflowException, match=r"This API is currently only available"):
        mlflow.update_expectation(trace_id="1234", assessment_id="1234", value=1.0)

    with pytest.raises(MlflowException, match=r"This API is currently only available"):
        mlflow.update_feedback(trace_id="1234", assessment_id="1234", value=1.0)

    with pytest.raises(MlflowException, match=r"This API is currently only available"):
        mlflow.delete_expectation(trace_id="1234", assessment_id="1234")

    with pytest.raises(MlflowException, match=r"This API is currently only available"):
        mlflow.delete_feedback(trace_id="1234", assessment_id="1234")


## Test that the fluent API functions correctly update the trace buffer


@pytest.fixture
def trace_obj():
    return Trace.from_dict(
        {
            "info": {
                "request_id": "1234567812345678abcdabcdabcdabcd",
                "experiment_id": "0",
                "timestamp_ms": 123456789,
                "execution_time_ms": 123,
                "status": "OK",
                "request_metadata": {
                    "mlflow.traceInputs": "Hello world!",
                    "mlflow.traceOutputs": "Hello!",
                    TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
                },
                "tags": {},
                "assessments": [],
            },
            "data": {
                "request": "Hello world!",
                "response": "Hello!",
                "spans": [
                    {
                        "name": "predict",
                        "context": {
                            "trace_id": "1234567812345678abcdabcdabcdabcd",
                            "span_id": "0xabcdabcdabcdabcd",
                        },
                        "parent_id": None,
                        "start_time": 0,
                        "end_time": 123456789,
                        "status_code": "OK",
                        "status_message": "",
                        "attributes": {
                            "mlflow.traceRequestId": "1234567812345678abcdabcdabcdabcd",
                            "mlflow.spanType": '"UNKNOWN"',
                            "mlflow.spanFunctionName": '"predict"',
                            "mlflow.spanInputs": "Hello world!",
                            "mlflow.spanOutputs": "Hello!",
                        },
                        "events": [],
                    },
                ],
            },
        }
    )


@pytest.fixture
def buffered_trace_id(trace_obj):
    TRACE_BUFFER[trace_obj.info.request_id] = trace_obj
    yield trace_obj.info.request_id
    del TRACE_BUFFER[trace_obj.info.request_id]


def test_log_expectation_updates_buffer(store, buffered_trace_id):
    mlflow.set_tracking_uri("databricks")
    store.create_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        expectation=Expectation("answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.log_expectation(
        trace_id=buffered_trace_id,
        name="expected_answer",
        value="answer",
        source=AssessmentSourceType.HUMAN,
    )

    retrieved_trace = mlflow.get_trace(buffered_trace_id)
    assert retrieved_trace.info.assessments[0].name == "expected_answer"


def test_update_expectation_updates_buffer(store, buffered_trace_id):
    mlflow.set_tracking_uri("databricks")
    store.create_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        expectation=Expectation("answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.log_expectation(
        trace_id=buffered_trace_id,
        name="expected_answer",
        value="answer",
        source=AssessmentSourceType.HUMAN,
    )

    store.update_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        expectation=Expectation("Updated answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.update_expectation(
        trace_id=buffered_trace_id,
        assessment_id="1234",
        name="expected_answer",
        value="answer",
    )

    retrieved_trace = mlflow.get_trace(buffered_trace_id)
    assert retrieved_trace.info.assessments[0].expectation.value == "Updated answer"


def test_delete_expectation_updates_buffer(store, buffered_trace_id):
    mlflow.set_tracking_uri("databricks")
    store.create_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        expectation=Expectation("answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.log_expectation(
        trace_id=buffered_trace_id,
        name="expected_answer",
        value="answer",
        source=AssessmentSourceType.HUMAN,
    )

    mlflow.delete_expectation(
        trace_id=buffered_trace_id,
        assessment_id="1234",
    )

    retrieved_trace = mlflow.get_trace(buffered_trace_id)
    assert len(retrieved_trace.info.assessments) == 0


def test_log_feedback_updates_buffer(store, buffered_trace_id):
    mlflow.set_tracking_uri("databricks")
    store.create_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        feedback=Feedback("answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.log_feedback(
        trace_id=buffered_trace_id,
        name="expected_answer",
        value="answer",
        source=AssessmentSourceType.HUMAN,
    )

    retrieved_trace = mlflow.get_trace(buffered_trace_id)
    assert retrieved_trace.info.assessments[0].name == "expected_answer"


def test_update_feedback_updates_buffer(store, buffered_trace_id):
    mlflow.set_tracking_uri("databricks")
    store.create_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        feedback=Feedback("answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.log_feedback(
        trace_id=buffered_trace_id,
        name="expected_answer",
        value="answer",
        source=AssessmentSourceType.HUMAN,
    )

    store.update_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        feedback=Feedback("Updated answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.update_feedback(
        trace_id=buffered_trace_id,
        assessment_id="1234",
        name="expected_answer",
        value="answer",
    )

    retrieved_trace = mlflow.get_trace(buffered_trace_id)
    assert retrieved_trace.info.assessments[0].feedback.value == "Updated answer"


def test_delete_feedback_updates_buffer(store, buffered_trace_id):
    mlflow.set_tracking_uri("databricks")
    store.create_assessment.return_value = Assessment(
        trace_id=buffered_trace_id,
        _assessment_id="1234",
        name="expected_answer",
        feedback=Feedback("answer"),
        source=AssessmentSourceType.HUMAN,
        create_time_ms=0,
        last_update_time_ms=0,
    )

    mlflow.log_feedback(
        trace_id=buffered_trace_id,
        name="expected_answer",
        value="answer",
        source=AssessmentSourceType.HUMAN,
    )

    mlflow.delete_feedback(
        trace_id=buffered_trace_id,
        assessment_id="1234",
    )

    retrieved_trace = mlflow.get_trace(buffered_trace_id)
    assert len(retrieved_trace.info.assessments) == 0
