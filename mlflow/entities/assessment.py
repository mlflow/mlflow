from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Union

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType  # noqa: F401
from mlflow.exceptions import MlflowException
from mlflow.protos.assessments_pb2 import Assessment as ProtoAssessment
from mlflow.protos.assessments_pb2 import Expectation as ProtoExpectation
from mlflow.protos.assessments_pb2 import Feedback as ProtoFeedback
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
    An abstraction for annotating a trace. An Assessment should be one of the following types:

    - Expectations: A label that represents the expected value for a particular operation.
        For example, an expected answer for a user question from a chatbot.
    - Feedback: A label that represents the feedback on the quality of the operation.
        Feedback can come from different sources, such as human judges, heuristic scorers,
        or LLM-as-a-Judge.

    You can log an assessment to a trace using the :py:func:`mlflow.log_expectation` or
    :py:func:`mlflow.log_feedback` functions.

    Args:
        name: The name of the assessment.
        source: The source of the assessment.
        trace_id: The ID of the trace associated with the assessment. If unset, the assessment
            is not associated with any trace yet.
        expectation: The expectation value of the assessment.
        feedback: The feedback value of the assessment.  Only one of `expectation` or `feedback`
            should be specified.
        rationale: The rationale / justification for the assessment.
        metadata: The metadata associated with the assessment.
        span_id: The ID of the span associated with the assessment, if the assessment should
            be associated with a particular span in the trace.
        create_time_ms: The creation time of the assessment in milliseconds. If unset, the
            current time is used.
        last_update_time_ms: The last update time of the assessment in milliseconds.
            If unset, the current time is used.
        assessment_id: The ID of the assessment. This must be generated in the backend.
    """

    name: str
    source: AssessmentSource
    # NB: The trace ID is optional because the assessment object itself may be created
    #   standalone. For example, a custom metric function returns an assessment object
    #   without a trace ID. That said, the trace ID is required when logging the
    #   assessment to a trace in the backend eventually.
    #   https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/custom-metrics#-metric-decorator
    trace_id: Optional[str] = None
    expectation: Optional[Expectation] = None
    feedback: Optional[Feedback] = None
    rationale: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    span_id: Optional[str] = None
    create_time_ms: Optional[int] = None
    last_update_time_ms: Optional[int] = None
    # NB: The assessment ID should always be generated in the backend. The CreateAssessment
    #   backend API asks for an incomplete Assessment object without an ID and returns a
    #   complete one with assessment_id, so the ID is Optional in the constructor here.
    assessment_id: Optional[str] = None
    # Deprecated, use `error` in Feedback instead. Just kept for backward compatibility
    # and will be removed in the 3.0.0 release.
    error: Optional[AssessmentError] = None

    def __post_init__(self):
        if (self.expectation is not None) + (self.feedback is not None) != 1:
            raise MlflowException.invalid_parameter_value(
                "Exactly one of `expectation` or `feedback` should be specified.",
            )

        # Populate the error field to the feedback object
        if self.error is not None:
            if self.expectation is not None:
                raise MlflowException.invalid_parameter_value(
                    "Cannot set `error` when `expectation` is specified.",
                )
            if self.feedback is None:
                raise MlflowException.invalid_parameter_value(
                    "Cannot set `error` when `feedback` is not specified.",
                )
            self.feedback.error = self.error

        # Set timestamp if not provided
        current_time = int(time.time() * 1000)  # milliseconds
        if self.create_time_ms is None:
            self.create_time_ms = current_time
        if self.last_update_time_ms is None:
            self.last_update_time_ms = current_time

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
        if self.assessment_id is not None:
            assessment.assessment_id = self.assessment_id

        if self.expectation is not None:
            set_pb_value(assessment.expectation.value, self.expectation.value)
        elif self.feedback is not None:
            assessment.feedback.CopyFrom(self.feedback.to_proto())

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
            feedback = Feedback.from_proto(proto.feedback)
        else:
            expectation = None
            feedback = None

        # Convert ScalarMapContainer to a normal Python dict
        metadata = dict(proto.metadata) if proto.metadata else None

        return cls(
            assessment_id=proto.assessment_id or None,
            trace_id=proto.trace_id,
            name=proto.assessment_name,
            source=AssessmentSource.from_proto(proto.source),
            create_time_ms=proto.create_time.ToMilliseconds(),
            last_update_time_ms=proto.last_update_time.ToMilliseconds(),
            expectation=expectation,
            feedback=feedback,
            rationale=proto.rationale or None,
            metadata=metadata,
            span_id=proto.span_id or None,
        )

    def to_dictionary(self):
        return {
            "assessment_id": self.assessment_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "source": self.source.to_dictionary(),
            "create_time_ms": self.create_time_ms,
            "last_update_time_ms": self.last_update_time_ms,
            "expectation": self.expectation.to_dictionary() if self.expectation else None,
            "feedback": self.feedback.to_dictionary() if self.feedback else None,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "span_id": self.span_id,
        }


@experimental
@dataclass
class Expectation(_MlflowObject):
    """
    Represents an expectation about the output of an operation, such as the expected response
    that a generative AI application should provide to a particular user query.
    """

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
    """
    Represents feedback about the output of an operation. For example, if the response from a
    generative AI application to a particular user query is correct, then a human or LLM judge
    may provide feedback with the value ``"correct"``.
    """

    value: AssessmentValueType
    error: Optional[AssessmentError] = None

    def to_proto(self):
        return ProtoFeedback(
            value=ParseDict(self.value, Value(), ignore_unknown_fields=True),
            error=self.error.to_proto() if self.error else None,
        )

    @classmethod
    def from_proto(self, proto):
        return Feedback(
            value=MessageToDict(proto.value),
            error=AssessmentError.from_proto(proto.error) if proto.HasField("error") else None,
        )

    def to_dictionary(self):
        d = {"value": self.value}
        if self.error:
            d["error"] = self.error.to_dictionary()
        return d
