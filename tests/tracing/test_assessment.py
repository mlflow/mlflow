from unittest import mock

import pytest

from mlflow.entities.assessment import AssessmentError, Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.tracing.assessment import log_expectation, log_feedback


# TODO: This test mocks out the tracking client and only test if the fluent API implementation
# passes the correct arguments to the low-level client. Once the OSS backend is implemented,
# we should also test the end-to-end assessment CRUD functionality.
@pytest.fixture
def mock_tracking_client():
    mock_client = mock.MagicMock()
    with mock.patch("mlflow.tracking.client.TrackingServiceClient", return_value=mock_client):
        yield mock_client


def test_log_expectation(mock_tracking_client):
    log_expectation(
        trace_id="1234",
        name="expected_answer",
        value="MLflow",
        source=AssessmentSourceType.HUMAN,
        metadata={"key": "value"},
    )

    assert mock_tracking_client.create_assessment.call_count == 1
    assessment = mock_tracking_client.create_assessment.call_args[0][0]
    assert assessment.name == "expected_answer"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source.source_type == "HUMAN"
    assert assessment.source.source_id is None
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert isinstance(assessment.value, Expectation)
    assert assessment.value.value == "MLflow"
    assert assessment.rationale is None
    assert assessment.metadata == {"key": "value"}
    assert assessment.error is None


def test_log_expectation_invalid_parameters():
    with pytest.raises(MlflowException, match=r"Expectation value cannot be None."):
        log_expectation(
            trace_id="1234",
            name="expected_answer",
            value=None,
            source=AssessmentSourceType.HUMAN,
        )

    with pytest.raises(MlflowException, match=r"`source` must be provided."):
        log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source=None,
        )

    with pytest.raises(MlflowException, match=r"Invalid assessment source type"):
        log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source="INVALID_SOURCE_TYPE",
        )


def test_log_feedback(mock_tracking_client):
    log_feedback(
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

    assert mock_tracking_client.create_assessment.call_count == 1
    assessment = mock_tracking_client.create_assessment.call_args[0][0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source.source_type == "LLM_JUDGE"
    assert assessment.source.source_id == "faithfulness-judge"
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert isinstance(assessment.value, Feedback)
    assert assessment.value.value == 1.0
    assert assessment.rationale == "This answer is very faithful."
    assert assessment.metadata == {"model": "gpt-4o-mini"}
    assert assessment.error is None


def test_log_feedback_with_error(mock_tracking_client):
    log_feedback(
        trace_id="1234",
        name="faithfulness",
        source=AssessmentSourceType.LLM_JUDGE,
        error=AssessmentError(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Rate limit for the judge exceeded.",
        ),
    )

    assert mock_tracking_client.create_assessment.call_count == 1
    assessment = mock_tracking_client.create_assessment.call_args[0][0]
    assert assessment.name == "faithfulness"
    assert assessment.trace_id == "1234"
    assert assessment.span_id is None
    assert assessment.source.source_type == "LLM_JUDGE"
    assert assessment.source.source_id is None
    assert assessment.create_time_ms is not None
    assert assessment.last_update_time_ms is not None
    assert assessment.value is None
    assert assessment.rationale is None
    assert assessment.error.error_code == "RATE_LIMIT_EXCEEDED"
    assert assessment.error.error_message == "Rate limit for the judge exceeded."


def test_log_feedback_invalid_parameters():
    with pytest.raises(MlflowException, match=r"Either `value` or `error` must be specified."):
        log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=AssessmentSourceType.LLM_JUDGE,
        )

    with pytest.raises(MlflowException, match=r"Only one of `value` or `error` should be "):
        log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=AssessmentSourceType.LLM_JUDGE,
            value=1.0,
            error=AssessmentError(
                error_code="RATE_LIMIT_EXCEEDED",
                error_message="Rate limit for the judge exceeded.",
            ),
        )

    with pytest.raises(MlflowException, match=r"`source` must be provided."):
        log_feedback(
            trace_id="1234",
            name="faithfulness",
            value=1.0,
            source=None,
        )
