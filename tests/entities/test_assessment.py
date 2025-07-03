import json
import time
from unittest.mock import patch

import pytest

from mlflow.entities.assessment import (
    Assessment,
    AssessmentError,
    AssessmentSource,
    Expectation,
    ExpectationValue,
    Feedback,
    FeedbackValue,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.assessments_pb2 import Assessment as ProtoAssessment
from mlflow.protos.assessments_pb2 import Expectation as ProtoExpectation
from mlflow.protos.assessments_pb2 import Feedback as ProtoFeedback
from mlflow.utils.proto_json_utils import proto_timestamp_to_milliseconds


def test_assessment_creation():
    default_params = {
        "trace_id": "trace_id",
        "name": "relevance",
        "source": AssessmentSource(source_type="HUMAN", source_id="user_1"),
        "create_time_ms": 123456789,
        "last_update_time_ms": 123456789,
        "expectation": None,
        "feedback": FeedbackValue(0.9),
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
            "feedback": FeedbackValue(
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
            "feedback": FeedbackValue(value=1, error=AssessmentError(error_code="E001")),
        }
    )
    assert assessment_with_value_and_error.feedback.value == 1
    assert assessment_with_value_and_error.feedback.error.error_code == "E001"

    # Backward compatibility. "error" was previously in the Assessment class.
    assessment_legacy_error = Assessment(
        **{
            **default_params,
            "error": AssessmentError(error_code="E001", error_message="An error occurred."),
            "feedback": FeedbackValue(None),
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
        feedback=FeedbackValue(0.9),
        **common_args,
    )
    assessment_2 = Assessment(
        source=source_2,
        feedback=FeedbackValue(0.9),
        **common_args,
    )
    assessment_3 = Assessment(
        source=source_1,
        feedback=FeedbackValue(0.8),
        **common_args,
    )
    assessment_4 = Assessment(
        source=source_3,
        feedback=FeedbackValue(0.9),
        **common_args,
    )
    assessment_5 = Assessment(
        source=source_1,
        feedback=FeedbackValue(
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
        feedback=FeedbackValue(
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
    Assessment(expectation=ExpectationValue("MLflow"), **common_args)
    Assessment(feedback=FeedbackValue("This is correct."), **common_args)
    Assessment(
        feedback=FeedbackValue(None, error=AssessmentError(error_code="E001")), **common_args
    )
    Assessment(
        feedback=FeedbackValue("This is correct.", AssessmentError(error_code="E001")),
        **common_args,
    )

    # Invalid case: no value specified
    with pytest.raises(MlflowException, match=r"Exactly one of"):
        Assessment(**common_args)

    # Invalid case: both feedback and expectation specified
    with pytest.raises(MlflowException, match=r"Exactly one of"):
        Assessment(
            expectation=ExpectationValue("MLflow"),
            feedback=FeedbackValue("This is correct."),
            **common_args,
        )

    # Invalid case: All three are set
    with pytest.raises(MlflowException, match=r"Exactly one of"):
        Assessment(
            expectation=ExpectationValue("MLflow"),
            feedback=FeedbackValue("This is correct.", AssessmentError(error_code="E001")),
            **common_args,
        )


@pytest.mark.parametrize(
    ("expectation", "feedback"),
    [
        (ExpectationValue("MLflow"), None),
        (ExpectationValue({"complex": {"expectation": ["structure"]}}), None),
        (None, FeedbackValue("This is correct.")),
        (None, FeedbackValue(None, AssessmentError(error_code="E001"))),
        (
            None,
            FeedbackValue(
                None, AssessmentError(error_code="E001", error_message="An error occurred.")
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        AssessmentSource(source_type="HUMAN", source_id="user_1"),
        AssessmentSource(source_type="CODE", source_id="code.py"),
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
    if expectation:
        assessment = Expectation(
            trace_id="trace_id",
            name="relevance",
            source=source,
            create_time_ms=timestamp_ms,
            last_update_time_ms=timestamp_ms,
            value=expectation.value,
            metadata=metadata,
            span_id="span_id",
        )
    elif feedback:
        assessment = Feedback(
            trace_id="trace_id",
            name="relevance",
            source=source,
            create_time_ms=timestamp_ms,
            last_update_time_ms=timestamp_ms,
            value=feedback.value,
            error=feedback.error,
            rationale="Rationale text",
            metadata=metadata,
            span_id="span_id",
        )
    else:
        raise ValueError("Either expectation or feedback must be provided")

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
        if isinstance(expectation.value, str):
            assert dict["expectation"] == {"value": expectation.value}
        else:
            assert dict["expectation"] == {
                "serialized_value": {
                    "value": json.dumps(expectation.value),
                    "serialization_format": "JSON_FORMAT",
                }
            }

    if feedback:
        assert dict["feedback"] == feedback.to_dictionary()


@pytest.mark.parametrize(
    "value",
    [
        "MLflow",  # string
        42,  # integer
        3.14,  # float
        True,  # boolean
    ],
    ids=["string", "integer", "float", "boolean"],
)
def test_expectation_proto_dict_conversion(value):
    expectation = ExpectationValue(value)
    proto = expectation.to_proto()
    assert isinstance(proto, ProtoExpectation)

    result = ExpectationValue.from_proto(proto)
    assert result.value == expectation.value

    expectation_dict = expectation.to_dictionary()
    result = ExpectationValue.from_dictionary(expectation_dict)
    assert result.value == result.value


@pytest.mark.parametrize(
    "value",
    [
        {"key": "value"},
        ["a", "b", "c"],
        {"nested": {"dict": {"with": ["mixed", "types", 1, 2.0, True]}}},
        [1, "two", 3.0, False, {"mixed": "list"}],
        [{"complex": "structure"}, [1, 2, 3], {"with": ["nested", "arrays"]}],
    ],
)
def test_expectation_value_serialization(value):
    expectation = ExpectationValue(value)
    proto = expectation.to_proto()

    assert proto.HasField("serialized_value")
    assert proto.serialized_value.serialization_format == "JSON_FORMAT"

    result = ExpectationValue.from_proto(proto)
    assert result.value == expectation.value

    expectation_dict = expectation.to_dictionary()
    result = ExpectationValue.from_dictionary(expectation_dict)
    assert result.value == result.value


def test_expectation_invalid_values():
    class CustomObject:
        pass

    with pytest.raises(MlflowException, match="Expectation value must be JSON-serializable"):
        ExpectationValue(CustomObject()).to_proto()

    # Test invalid serialization format
    proto = ProtoExpectation()
    proto.serialized_value.serialization_format = "INVALID_FORMAT"
    proto.serialized_value.value = '{"key": "value"}'

    with pytest.raises(MlflowException, match="Unknown serialization format"):
        ExpectationValue.from_proto(proto)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0.9, None),
        (None, AssessmentError(error_code="E001", error_message="An error occurred.")),
        (
            "Error message",
            AssessmentError(error_code="E002", error_message="Another error occurred."),
        ),
    ],
)
def test_feedback_value_proto_dict_conversion(value, error):
    feedback = FeedbackValue(value, error)
    proto = feedback.to_proto()
    assert isinstance(proto, ProtoFeedback)

    result = FeedbackValue.from_proto(proto)
    assert result.value == result.value
    assert result.error == result.error

    feedback_dict = feedback.to_dictionary()
    result = FeedbackValue.from_dictionary(feedback_dict)
    assert result.value == result.value
    assert result.error == result.error


@pytest.mark.parametrize("stack_trace_length", [500, 2000])
def test_feedback_from_exception(stack_trace_length):
    err = None
    try:
        raise ValueError("An error occurred.")
    except ValueError as e:
        err = e

    # Mock traceback.format_tb to simulate long stack trace
    with patch(
        "mlflow.entities.assessment.get_stacktrace",
        return_value="A" * (stack_trace_length - 9) + "last line",
    ):
        feedback = Feedback(error=err)
    assert feedback.error.error_code == "ValueError"
    assert feedback.error.error_message == "An error occurred."
    assert feedback.error.stack_trace is not None

    # Feedback should expose error_code and error_message for backward compatibility
    assert feedback.error_code == "ValueError"
    assert feedback.error_message == "An error occurred."

    proto = feedback.to_proto()
    assert len(proto.feedback.error.stack_trace) == min(stack_trace_length, 1000)
    assert proto.feedback.error.stack_trace.endswith("last line")
    if stack_trace_length > 1000:
        assert proto.feedback.error.stack_trace.startswith("[Stack trace is truncated]\n...\n")

    recovered = Feedback.from_proto(feedback.to_proto())
    assert feedback.error.error_code == recovered.error.error_code
    assert feedback.error.error_message == recovered.error.error_message


def test_assessment_value_assignment():
    feedback = Feedback(name="relevance", value=1.0)
    assert feedback.value == 1.0

    feedback.value = 0.9
    assert feedback.value == 0.9

    expectation = Expectation(name="expected_value", value=1.0)
    assert expectation.value == 1.0

    expectation.value = 0.9
    assert expectation.value == 0.9
