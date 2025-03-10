from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType  # noqa: F401
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
class Assessment(_MlflowObject):
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
        expectation: The expectation value of the assessment.
        feedback: The feedback value of the assessment. Only one of `expectation`, `feedback`
            or `error` should be specified.
        rationale: The rationale / justification for the assessment.
        metadata: The metadata associated with the assessment.
        error: An error object representing any issues during generating the assessment.
            If this is set, the assessment should not contain `expectation` or `feedback`.
        span_id: The ID of the span associated with the assessment, if the assessment should
            be associated with a particular span in the trace.
        _assessment_id: The ID of the assessment. This must be generated in the backend.
    """

    trace_id: str
    name: str
    source: AssessmentSource
    create_time_ms: int
    last_update_time_ms: int
    expectation: Optional[Expectation] = None
    feedback: Optional[Feedback] = None
    rationale: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
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
        if (self.expectation is not None) + (self.feedback is not None) != 1:
            raise MlflowException.invalid_parameter_value(
                "Exactly one of `expectation` or `feedback` should be specified.",
            )
        if (self.expectation is not None) and self.error is not None:
            raise MlflowException.invalid_parameter_value(
                "Expectations cannot have `error` specified.",
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

        if self.expectation is not None:
            set_pb_value(assessment.expectation.value, self.expectation.value)
        elif self.feedback is not None:
            set_pb_value(assessment.feedback.value, self.feedback.value)

        if self.metadata:
            assessment.metadata.update(self.metadata)

        return assessment

    @classmethod
    def from_proto(cls, proto):
        if proto.WhichOneof("value") == "expectation":
            expectation = Expectation(parse_pb_value(proto.expectation.value))
            feedback = None
        elif proto.WhichOneof("value") == "feedback":
            expectation = None
            feedback = Feedback(parse_pb_value(proto.feedback.value))
        else:
            expectation = None
            feedback = None

        error = AssessmentError.from_proto(proto.error) if proto.error.error_code else None
        metadata = proto.metadata

        return cls(
            _assessment_id=proto.assessment_id or None,
            trace_id=proto.trace_id,
            name=proto.assessment_name,
            source=AssessmentSource.from_proto(proto.source),
            create_time_ms=proto.create_time.ToMilliseconds(),
            last_update_time_ms=proto.last_update_time.ToMilliseconds(),
            expectation=expectation,
            feedback=feedback,
            rationale=proto.rationale or None,
            metadata=metadata or None,
            error=error,
            span_id=proto.span_id or None,
        )

    def to_dictionary(self):
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "source": self.source.to_dictionary(),
            "create_time_ms": self.create_time_ms,
            "last_update_time_ms": self.last_update_time_ms,
            "expectation": self.expectation,
            "feedback": self.feedback,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "error": self.error.to_dictionary() if self.error else None,
            "span_id": self.span_id,
            "_assessment_id": self._assessment_id,
        }


@experimental
@dataclass
class Expectation(_MlflowObject):
    """Represents an expectation value in an assessment."""

    value: AssessmentValueType

    def to_proto(self):
        expectation = ProtoExpectation()
        expectation.value = self.value
        return expectation

    def to_dictionary(self):
        return {"value": self.value}


@experimental
@dataclass
class Feedback(_MlflowObject):
    """Represents a feedback value in an assessment."""

    value: AssessmentValueType

    def to_proto(self):
        feedback = ProtoFeedback()
        feedback.value = self.value
        return feedback

    def to_dictionary(self):
        return {"value": self.value}
