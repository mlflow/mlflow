import time

import pytest

from mlflow.entities.assessment import (
    Assessment,
    AssessmentError,
    AssessmentSource,
    Expectation,
    Feedback,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import Assessment as ProtoAssessment


def test_assessment_creation():
    default_params = {
        "trace_id": "trace_id",
        "name": "relevance",
        "source": AssessmentSource(source_type="HUMAN", source_id="user_1"),
        "create_time_ms": 123456789,
        "last_update_time_ms": 123456789,
        "expectation": None,
        "feedback": Feedback(0.9),
        "rationale": "Rationale text",
        "metadata": {"key1": "value1"},
        "error": None,
        "span_id": "span_id",
        "_assessment_id": "assessment_id",
    }

    assessment = Assessment(**default_params)
    for key, value in default_params.items():
        assert getattr(assessment, key) == value

    assessment_with_error = Assessment(
        **{
            **default_params,
            "feedback": Feedback(None),
            "error": AssessmentError(error_code="E001", error_message="An error occurred."),
        }
    )
    assert assessment_with_error.error.error_code == "E001"
    assert assessment_with_error.error.error_message == "An error occurred."


def test_assessment_equality():
    source_1 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_2 = AssessmentSource(source_type="HUMAN", source_id="user_1")
    source_3 = AssessmentSource(source_type="LLM_JUDGE", source_id="llm_1")

    common_args = {
        "trace_id": "trace_id",
        "name": "relevance",
        "create_time_ms": 123456789,
        "last_update_time_ms": 123456789,
    }

    # Valid assessments
    assessment_1 = Assessment(
        source=source_1,
        feedback=Feedback(0.9),
        **common_args,
    )
    assessment_2 = Assessment(
        source=source_2,
        feedback=Feedback(0.9),
        **common_args,
    )
    assessment_3 = Assessment(
        source=source_1,
        feedback=Feedback(0.8),
        **common_args,
    )
    assessment_4 = Assessment(
        source=source_3,
        feedback=Feedback(0.9),
        **common_args,
    )
    assessment_5 = Assessment(
        source=source_1,
        feedback=Feedback(None),
        error=AssessmentError(
            error_code="E002",
            error_message="A different error occurred.",
        ),
        **common_args,
    )
    assessment_6 = Assessment(
        source=source_1,
        feedback=Feedback(None),
        error=AssessmentError(
            error_code="E001",
            error_message="Another error message.",
        ),
        **common_args,
    )

    # Same evaluation_id, name, source, timestamp, and numeric_value
    assert assessment_1 == assessment_2
    assert assessment_1 != assessment_3  # Different numeric_value
    assert assessment_1 != assessment_4  # Different source
    assert assessment_1 != assessment_5  # One has numeric_value, other has error_code
    assert assessment_5 != assessment_6  # Different error_code


def test_assessment_value_validation():
    common_args = {
        "trace_id": "trace_id",
        "name": "relevance",
        "source": AssessmentSource(source_type="HUMAN", source_id="user_1"),
        "create_time_ms": 123456789,
        "last_update_time_ms": 123456789,
    }

    # Valid cases
    Assessment(expectation=Expectation("MLflow"), **common_args)
    Assessment(feedback=Feedback("This is correct."), **common_args)
    Assessment(feedback=Feedback(None), error=AssessmentError(error_code="E001"), **common_args)
    Assessment(
        feedback=Feedback("This is correct."),
        error=AssessmentError(error_code="E001"),
        **common_args,
    )

    # Invalid case: no value specified
    with pytest.raises(MlflowException, match=r"Exactly one of"):
        Assessment(**common_args)

    # Invalid case: both feedback and expectation specified
    with pytest.raises(MlflowException, match=r"Exactly one of"):
        Assessment(
            expectation=Expectation("MLflow"),
            feedback=Feedback("This is correct."),
            **common_args,
        )

    # Invalid case: Expectation with an error
    with pytest.raises(MlflowException, match=r"Expectations cannot have"):
        Assessment(
            expectation=Expectation("MLflow"),
            error=AssessmentError(error_code="E001"),
            **common_args,
        )

    # Invalid case: All three are set
    with pytest.raises(MlflowException, match=r"Exactly one of"):
        Assessment(
            expectation=Expectation("MLflow"),
            feedback=Feedback("This is correct."),
            error=AssessmentError(error_code="E001"),
            **common_args,
        )


@pytest.mark.parametrize(
    ("expectation", "feedback", "error"),
    [
        (Expectation("MLflow"), None, None),
        (None, Feedback("This is correct."), None),
        (None, Feedback(None), AssessmentError(error_code="E001")),
        (
            None,
            Feedback(None),
            AssessmentError(error_code="E001", error_message="An error occurred."),
        ),
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        AssessmentSource(source_type="HUMAN", source_id="user_1"),
        AssessmentSource(source_type="CODE"),
    ],
)
@pytest.mark.parametrize(
    "metadata",
    [
        {"key1": "value1"},
        None,
    ],
)
def test_assessment_conversion(expectation, feedback, error, source, metadata):
    timestamp_ms = int(time.time() * 1000)
    assessment = Assessment(
        trace_id="trace_id",
        name="relevance",
        source=source,
        create_time_ms=timestamp_ms,
        last_update_time_ms=timestamp_ms,
        expectation=expectation,
        feedback=feedback,
        rationale="Rationale text",
        metadata=metadata,
        error=error,
        span_id="span_id",
    )

    proto = assessment.to_proto()

    assert isinstance(proto, ProtoAssessment)

    result = Assessment.from_proto(proto)
    assert result == assessment

    dict = assessment.to_dictionary()
    assert dict["assessment_id"] == assessment._assessment_id
    assert dict["trace_id"] == assessment.trace_id
    assert dict["name"] == assessment.name
    assert dict["source"] == {
        "source_type": source.source_type,
        "source_id": source.source_id,
    }
    assert dict["create_time_ms"] == assessment.create_time_ms
    assert dict["last_update_time_ms"] == assessment.last_update_time_ms
    assert dict["rationale"] == assessment.rationale
    assert dict["metadata"] == metadata

    if expectation:
        assert dict["expectation"] == {"value": expectation.value}

    if feedback:
        assert dict["feedback"] == feedback.to_dictionary()

    if error:
        assert dict["error"] == {
            "error_code": error.error_code,
            "error_message": error.error_message,
        }
