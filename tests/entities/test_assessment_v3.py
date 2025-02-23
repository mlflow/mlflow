import time

import pytest

from mlflow.entities.assessment_v3 import (
    AssessmentV3,
    AssessmentError,
    AssessmentSourceV3,
    Expectation,
    Feedback,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import Assessment as ProtoAssessment


def test_assessment_creation():
    default_params = {
        "trace_id": "trace_id",
        "name": "relevance",
        "source": AssessmentSourceV3(source_type="HUMAN", source_id="user_1"),
        "create_time_ms": 123456789,
        "last_update_time_ms": 123456789,
        "value": Feedback(0.9),
        "rationale": "Rationale text",
        "metadata": {"key1": "value1"},
        "error": None,
        "span_id": "span_id",
        "_assessment_id": "assessment_id",
    }

    assessment = AssessmentV3(**default_params)
    for key, value in default_params.items():
        assert getattr(assessment, key) == value

    assessment_with_error = AssessmentV3(
        **{
            **default_params,
            "value": None,
            "error": AssessmentError(error_code="E001", error_message="An error occurred."),
        }
    )
    assert assessment_with_error.error.error_code == "E001"
    assert assessment_with_error.error.error_message == "An error occurred."


def test_assessment_equality():
    source_1 = AssessmentSourceV3(source_type="HUMAN", source_id="user_1")
    source_2 = AssessmentSourceV3(source_type="HUMAN", source_id="user_1")
    source_3 = AssessmentSourceV3(source_type="LLM_JUDGE", source_id="llm_1")

    common_args = {
        "trace_id": "trace_id",
        "name": "relevance",
        "create_time_ms": 123456789,
        "last_update_time_ms": 123456789,
    }

    # Valid assessments
    assessment_1 = AssessmentV3(
        source=source_1,
        value=Feedback(0.9),
        **common_args,
    )
    assessment_2 = AssessmentV3(
        source=source_2,
        value=Feedback(0.9),
        **common_args,
    )
    assessment_3 = AssessmentV3(
        source=source_1,
        value=Feedback(0.8),
        **common_args,
    )
    assessment_4 = AssessmentV3(
        source=source_3,
        value=Feedback(0.9),
        **common_args,
    )
    assessment_5 = AssessmentV3(
        source=source_1,
        error=AssessmentError(
            error_code="E002",
            error_message="A different error occurred.",
        ),
        **common_args,
    )
    assessment_6 = AssessmentV3(
        source=source_1,
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
        "source": AssessmentSourceV3(source_type="HUMAN", source_id="user_1"),
        "create_time_ms": 123456789,
        "last_update_time_ms": 123456789,
    }

    # Valid cases
    AssessmentV3(value=Expectation("MLflow"), **common_args)
    AssessmentV3(value=Feedback("This is correct."), **common_args)

    # Invalid case: invalid value type
    with pytest.raises(MlflowException, match=r"Value must be an instance of "):
        AssessmentV3(value="Invalid value type", **common_args)

    # Invalid case: no value specified
    with pytest.raises(MlflowException, match=r"Either `value` or `error` must be specified"):
        AssessmentV3(**common_args)

    # Invalid case: both value and error specified
    with pytest.raises(MlflowException, match=r"Only one of `value` or `error` should be"):
        AssessmentV3(
            value=Expectation("MLflow"),
            error=AssessmentError(error_code="E001", error_message="An error occurred."),
            **common_args,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (Expectation("MLflow"), None),
        (Feedback("This is correct."), None),
        (None, AssessmentError(error_code="E001")),
        (None, AssessmentError(error_code="E001", error_message="An error occurred.")),
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        AssessmentSourceV3(source_type="HUMAN", source_id="user_1"),
        AssessmentSourceV3(source_type="CODE"),
    ],
)
@pytest.mark.parametrize(
    "metadata",
    [
        {"key1": "value1"},
        {
            "key2": 1,
            "key3": 2.0,
            "key4": True,
            "key5": [1, 2, 3],
            "key6": {"key7": "value7"},
        },
        None,
    ],
)
def test_assessment_proto_conversion(value, error, source, metadata):
    timestamp_ms = int(time.time() * 1000)
    assessment = AssessmentV3(
        trace_id="trace_id",
        name="relevance",
        source=source,
        create_time_ms=timestamp_ms,
        last_update_time_ms=timestamp_ms,
        value=value,
        rationale="Rationale text",
        metadata=metadata,
        error=error,
        span_id="span_id",
    )

    proto = assessment.to_proto()

    assert isinstance(proto, ProtoAssessment)

    result = AssessmentV3.from_proto(proto)
    assert result == assessment
