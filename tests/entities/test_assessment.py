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
from mlflow.protos.assessments_pb2 import Assessment as ProtoAssessment
from mlflow.utils.proto_json_utils import proto_timestamp_to_milliseconds


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
        "span_id": "span_id",
        "assessment_id": "assessment_id",
    }

    assessment = Assessment(**default_params)
    for key, value in default_params.items():
        assert getattr(assessment, key) == value

    assessment_with_error = Assessment(
        **{
            **default_params,
            "feedback": Feedback(
                None, AssessmentError(error_code="E001", error_message="An error occurred.")
            ),
        }
    )
    assert assessment_with_error.feedback.error.error_code == "E001"
    assert assessment_with_error.feedback.error.error_message == "An error occurred."

    # Both feedback value and error can be set. For example, a default fallback value can
    # be set when LLM judge fails to provide a value.
    assessment_with_value_and_error = Assessment(
        **{
            **default_params,
            "feedback": Feedback(value=1, error=AssessmentError(error_code="E001")),
        }
    )
    assert assessment_with_value_and_error.feedback.value == 1
    assert assessment_with_value_and_error.feedback.error.error_code == "E001"

    # Backward compatibility. "error" was previously in the Assessment class.
    assessment_legacy_error = Assessment(
        **{
            **default_params,
            "error": AssessmentError(error_code="E001", error_message="An error occurred."),
            "feedback": Feedback(None),
        }
    )
    assert assessment_legacy_error.feedback.error.error_code == "E001"
    assert assessment_legacy_error.feedback.error.error_message == "An error occurred."


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
        feedback=Feedback(
            None,
            AssessmentError(
                error_code="E002",
                error_message="A different error occurred.",
            ),
        ),
        **common_args,
    )
    assessment_6 = Assessment(
        source=source_1,
        feedback=Feedback(
            None,
            AssessmentError(
                error_code="E001",
                error_message="Another error message.",
            ),
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
    Assessment(feedback=Feedback(None, error=AssessmentError(error_code="E001")), **common_args)
    Assessment(
        feedback=Feedback("This is correct.", AssessmentError(error_code="E001")),
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

    # Invalid case: All three are set
    with pytest.raises(MlflowException, match=r"Exactly one of"):
        Assessment(
            expectation=Expectation("MLflow"),
            feedback=Feedback("This is correct.", AssessmentError(error_code="E001")),
            **common_args,
        )


@pytest.mark.parametrize(
    ("expectation", "feedback"),
    [
        (Expectation("MLflow"), None),
        (None, Feedback("This is correct.")),
        (None, Feedback(None, AssessmentError(error_code="E001"))),
        (
            None,
            Feedback(None, AssessmentError(error_code="E001", error_message="An error occurred.")),
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
def test_assessment_conversion(expectation, feedback, source, metadata):
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
        span_id="span_id",
    )

    proto = assessment.to_proto()

    assert isinstance(proto, ProtoAssessment)

    result = Assessment.from_proto(proto)
    assert result == assessment

    dict = assessment.to_dictionary()
    assert dict.get("assessment_id") == assessment.assessment_id
    assert dict["trace_id"] == assessment.trace_id
    assert dict["assessment_name"] == assessment.name
    assert dict["source"].get("source_type") == source.source_type
    assert dict["source"].get("source_id") == source.source_id
    assert proto_timestamp_to_milliseconds(dict["create_time"]) == assessment.create_time_ms
    assert (
        proto_timestamp_to_milliseconds(dict["last_update_time"]) == assessment.last_update_time_ms
    )
    assert dict.get("rationale") == assessment.rationale
    assert dict.get("metadata") == metadata

    if expectation:
        assert dict["expectation"] == {"value": expectation.value}

    if feedback:
        assert dict["feedback"] == feedback.to_dictionary()
