from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSourceV3
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import Assessment as ProtoAssessment
from mlflow.protos.service_pb2 import Expectation as ProtoExpectation
from mlflow.protos.service_pb2 import Feedback as ProtoFeedback
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import parse_pb_value, set_pb_value

# Assessment value should be one of the following types:
# - float
# - int
# - str
# - bool
# - list of values of the same types as above
# - dict with string keys and values of the same types as above
PbValueType = Union[float, int, str, bool]
AssessmentValueType = Union[PbValueType, dict[str, PbValueType], list[PbValueType]]


@experimental
@dataclass
class AssessmentV3(_MlflowObject):
    """
    Assessment object associated with a trace.

    Assessment are an abstraction for annotating two different types of labels on traces:

    - Expectations: A label that represents the expected value for a particular operation.
        For example, an expected answer for a user question from a chatbot.
    - Feedback: A label that represents the feedback on the quality of the operation.
        Feedback can come from different sources, such as human judges, heuristic scorers,
        or LLM-as-a-Judge.

    To create an assessment with these labels, use the :py:func:`mlflow.log_expectation`
    or :py:func:`mlflow.log_feedback` functions. Do **not** create an assessment object
    directly using the constructor.

    Args:
        trace_id: The ID of the trace associated with the assessment.
        name: The name of the assessment.
        source: The source of the assessment.
        create_time_ms: The creation time of the assessment in milliseconds.
        last_update_time_ms: The last update time of the assessment in milliseconds.
        value: The value of the assessment. It can be an expectation or feedback.
        rationale: The rationale / justification for the assessment.
        metadata: The metadata associated with the assessment.
        error: An error object representing any issues during generating the assessment.
            Either `value` or `error` should be specified, not both.
        span_id: The ID of the span associated with the assessment, if the assessment should
            be associated with a particular span in the trace.
        _assessment_id: The ID of the assessment. This must be generated in the backend.
    """

    trace_id: str
    name: str
    source: AssessmentSourceV3
    create_time_ms: int
    last_update_time_ms: int
    value: Optional[Union[Expectation, Feedback]] = None
    rationale: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    error: Optional[AssessmentError] = None
    span_id: Optional[str] = None
    # NB: The assessment ID should always be generated in the backend. The CreateAssessment
    #   backend API asks for an incomplete Assessment object without an ID and returns a
    #   complete one with assessment_id, so the ID is Optional in the constructor here.
    _assessment_id: Optional[str] = None

    @property
    def assessment_id(self) -> str:
        if self._assessment_id is None:
            raise ValueError(
                "Assessment ID is not set. The assessment object might not be "
                "properly created. Please use the `mlflow.log_expectation` or "
                "the `mlflow.log_feedback` API to create an assessment."
            )
        return self._assessment_id

    def __post_init__(self):
        if self.value is not None and self.error is not None:
            raise MlflowException.invalid_parameter_value(
                "Only one of `value` or `error` should be specified, not both.",
            )

        if self.value is None and self.error is None:
            raise MlflowException.invalid_parameter_value(
                "Either `value` or `error` must be specified.",
            )

        if not isinstance(self.value, (Expectation, Feedback)) and self.value is not None:
            raise MlflowException.invalid_parameter_value(
                f"Value must be an instance of Expectation or Feedback. Got {type(self.value)}.",
            )

    def to_proto(self):
        assessment = ProtoAssessment()
        assessment.assessment_name = self.name
        assessment.trace_id = self.trace_id

        assessment.source.CopyFrom(self.source.to_proto())

        # Convert time in milliseconds to protobuf Timestamp
        assessment.create_time.FromMilliseconds(self.create_time_ms)
        assessment.last_update_time.FromMilliseconds(self.last_update_time_ms)

        if self.span_id is not None:
            assessment.span_id = self.span_id
        if self.rationale is not None:
            assessment.rationale = self.rationale
        if self._assessment_id is not None:
            assessment.assessment_id = self.assessment_id
        if self.error is not None:
            assessment.error.CopyFrom(self.error.to_proto())

        if isinstance(self.value, Expectation):
            set_pb_value(assessment.expectation.value, self.value.value)
        elif isinstance(self.value, Feedback):
            set_pb_value(assessment.feedback.value, self.value.value)

        # The metadata values are google.protobuf.Value and does not support
        # assignment like metadata[key] = value.
        if self.metadata:
            for key, value in self.metadata.items():
                set_pb_value(assessment.metadata[key], value)

        return assessment

    @classmethod
    def from_proto(cls, proto):
        if proto.WhichOneof("value") == "expectation":
            value = Expectation(parse_pb_value(proto.expectation.value))
        elif proto.WhichOneof("value") == "feedback":
            value = Feedback(parse_pb_value(proto.feedback.value))
        else:
            value = None

        error = AssessmentError.from_proto(proto.error) if proto.error.error_code else None
        metadata = {key: parse_pb_value(proto.metadata[key]) for key in proto.metadata}

        return cls(
            _assessment_id=proto.assessment_id or None,
            trace_id=proto.trace_id,
            name=proto.assessment_name,
            source=AssessmentSourceV3.from_proto(proto.source),
            create_time_ms=proto.create_time.ToMilliseconds(),
            last_update_time_ms=proto.last_update_time.ToMilliseconds(),
            value=value,
            rationale=proto.rationale or None,
            metadata=metadata or None,
            error=error,
            span_id=proto.span_id or None,
        )


@experimental
@dataclass
class Expectation(_MlflowObject):
    """Represents an expectation value in an assessment."""

    value: AssessmentValueType

    def to_proto(self):
        expectation = ProtoExpectation()
        expectation.value = self.value
        return expectation


@experimental
@dataclass
class Feedback(_MlflowObject):
    """Represents a feedback value in an assessment."""

    value: AssessmentValueType

    def to_proto(self):
        feedback = ProtoFeedback()
        feedback.value = self.value
        return feedback
