from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.protos.assessments_pb2 import Assessment as ProtoAssessment
from mlflow.protos.assessments_pb2 import Expectation as ProtoExpectation
from mlflow.protos.assessments_pb2 import Feedback as ProtoFeedback
from mlflow.utils.annotations import experimental
from mlflow.utils.exception_utils import get_stacktrace
from mlflow.utils.proto_json_utils import proto_timestamp_to_milliseconds

# Feedback value should be one of the following types:
# - float
# - int
# - str
# - bool
# - list of values of the same types as above
# - dict with string keys and values of the same types as above
PbValueType = float | int | str | bool
FeedbackValueType = PbValueType | dict[str, PbValueType] | list[PbValueType]


@experimental(version="2.21.0")
@dataclass
class Assessment(_MlflowObject):
    """
    Base class for assessments that can be attached to a trace.
    An Assessment should be one of the following types:

    - Expectations: A label that represents the expected value for a particular operation.
        For example, an expected answer for a user question from a chatbot.
    - Feedback: A label that represents the feedback on the quality of the operation.
        Feedback can come from different sources, such as human judges, heuristic scorers,
        or LLM-as-a-Judge.
    """

    name: str
    source: AssessmentSource
    # NB: The trace ID is optional because the assessment object itself may be created
    #   standalone. For example, a custom metric function returns an assessment object
    #   without a trace ID. That said, the trace ID is required when logging the
    #   assessment to a trace in the backend eventually.
    #   https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/custom-metrics#-metric-decorator
    trace_id: str | None = None
    run_id: str | None = None
    rationale: str | None = None
    metadata: dict[str, str] | None = None
    span_id: str | None = None
    create_time_ms: int | None = None
    last_update_time_ms: int | None = None
    # NB: The assessment ID should always be generated in the backend. The CreateAssessment
    #   backend API asks for an incomplete Assessment object without an ID and returns a
    #   complete one with assessment_id, so the ID is Optional in the constructor here.
    assessment_id: str | None = None
    # Deprecated, use `error` in Feedback instead. Just kept for backward compatibility
    # and will be removed in the 3.0.0 release.
    error: AssessmentError | None = None
    # Should only be used internally. To create an assessment with an expectation or feedback,
    # use the`Expectation` or `Feedback` classes instead.
    expectation: ExpectationValue | None = None
    feedback: FeedbackValue | None = None
    # The ID of the assessment which this assessment overrides.
    overrides: str | None = None
    # Whether this assessment is valid (i.e. has not been overridden).
    # This should not be set by the user, it is automatically set by the backend.
    valid: bool | None = None

    def __post_init__(self):
        from mlflow.tracing.constant import AssessmentMetadataKey

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

        if not isinstance(self.source, AssessmentSource):
            raise MlflowException.invalid_parameter_value(
                "`source` must be an instance of `AssessmentSource`. "
                f"Got {type(self.source)} instead."
            )
        # Extract and set run_id from metadata but don't modify the proto representation
        if (
            self.run_id is None
            and self.metadata
            and AssessmentMetadataKey.SOURCE_RUN_ID in self.metadata
        ):
            self.run_id = self.metadata[AssessmentMetadataKey.SOURCE_RUN_ID]

    def to_proto(self):
        assessment = ProtoAssessment()
        assessment.assessment_name = self.name
        assessment.trace_id = self.trace_id or ""

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
            assessment.expectation.CopyFrom(self.expectation.to_proto())
        elif self.feedback is not None:
            assessment.feedback.CopyFrom(self.feedback.to_proto())

        if self.metadata:
            for key, value in self.metadata.items():
                assessment.metadata[key] = str(value)
        if self.overrides:
            assessment.overrides = self.overrides
        if self.valid is not None:
            assessment.valid = self.valid

        return assessment

    @classmethod
    def from_proto(cls, proto):
        if proto.WhichOneof("value") == "expectation":
            return Expectation.from_proto(proto)
        elif proto.WhichOneof("value") == "feedback":
            return Feedback.from_proto(proto)
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unknown assessment type: {proto.WhichOneof('value')}"
            )

    def to_dictionary(self):
        # Note that MessageToDict excludes None fields. For example, if assessment_id is None,
        # it won't be included in the resulting dictionary.
        return MessageToDict(self.to_proto(), preserving_proto_field_name=True)

    @classmethod
    def from_dictionary(cls, d: dict[str, Any]) -> "Assessment":
        if d.get("expectation"):
            return Expectation.from_dictionary(d)
        elif d.get("feedback"):
            return Feedback.from_dictionary(d)
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unknown assessment type: {d.get('assessment_name')}"
            )


DEFAULT_FEEDBACK_NAME = "feedback"


@experimental(version="3.0.0")
@dataclass
class Feedback(Assessment):
    """
    Represents feedback about the output of an operation. For example, if the response from a
    generative AI application to a particular user query is correct, then a human or LLM judge
    may provide feedback with the value ``"correct"``.

    Args:
        name: The name of the assessment. If not provided, the default name "feedback" is used.
        value: The feedback value. This can be one of the following types:
            - float
            - int
            - str
            - bool
            - list of values of the same types as above
            - dict with string keys and values of the same types as above
        error: An optional error associated with the feedback. This is used to indicate
            that the feedback is not valid or cannot be processed. Accepts an exception
            object, or an :py:class:`~mlflow.entities.Expectation` object.
        rationale: The rationale / justification for the feedback.
        source: The source of the assessment. If not provided, the default source is CODE.
        trace_id: The ID of the trace associated with the assessment. If unset, the assessment
            is not associated with any trace yet.
            should be specified.
        metadata: The metadata associated with the assessment.
        span_id: The ID of the span associated with the assessment, if the assessment should
            be associated with a particular span in the trace.
        create_time_ms: The creation time of the assessment in milliseconds. If unset, the
            current time is used.
        last_update_time_ms: The last update time of the assessment in milliseconds.
            If unset, the current time is used.

    Example:

        .. code-block:: python

            from mlflow.entities import AssessmentSource, Feedback

            feedback = Feedback(
                name="correctness",
                value=True,
                rationale="The response is correct.",
                source=AssessmentSource(
                    source_type="HUMAN",
                    source_id="john@example.com",
                ),
                metadata={"project": "my-project"},
            )
    """

    def __init__(
        self,
        name: str = DEFAULT_FEEDBACK_NAME,
        value: FeedbackValueType | None = None,
        error: Exception | AssessmentError | None = None,
        source: AssessmentSource | None = None,
        trace_id: str | None = None,
        metadata: dict[str, str] | None = None,
        span_id: str | None = None,
        create_time_ms: int | None = None,
        last_update_time_ms: int | None = None,
        rationale: str | None = None,
        overrides: str | None = None,
        valid: bool = True,
    ):
        if value is None and error is None:
            raise MlflowException.invalid_parameter_value(
                "Either `value` or `error` must be provided.",
            )

        # Default to CODE source if not provided
        if source is None:
            source = AssessmentSource(source_type=AssessmentSourceType.CODE)

        if isinstance(error, Exception):
            error = AssessmentError(
                error_message=str(error),
                error_code=error.__class__.__name__,
                stack_trace=get_stacktrace(error),
            )

        super().__init__(
            name=name,
            source=source,
            trace_id=trace_id,
            metadata=metadata,
            span_id=span_id,
            create_time_ms=create_time_ms,
            last_update_time_ms=last_update_time_ms,
            feedback=FeedbackValue(value=value, error=error),
            rationale=rationale,
            overrides=overrides,
            valid=valid,
        )
        self.error = error

    @property
    def value(self) -> FeedbackValueType:
        return self.feedback.value

    @value.setter
    def value(self, value: FeedbackValueType):
        self.feedback.value = value

    @classmethod
    def from_proto(cls, proto):
        # Convert ScalarMapContainer to a normal Python dict
        metadata = dict(proto.metadata) if proto.metadata else None
        feedback_value = FeedbackValue.from_proto(proto.feedback)
        feedback = cls(
            trace_id=proto.trace_id,
            name=proto.assessment_name,
            source=AssessmentSource.from_proto(proto.source),
            create_time_ms=proto.create_time.ToMilliseconds(),
            last_update_time_ms=proto.last_update_time.ToMilliseconds(),
            value=feedback_value.value,
            error=feedback_value.error,
            rationale=proto.rationale or None,
            metadata=metadata,
            span_id=proto.span_id or None,
            overrides=proto.overrides or None,
            valid=proto.valid,
        )
        feedback.assessment_id = proto.assessment_id or None
        return feedback

    @classmethod
    def from_dictionary(cls, d: dict[str, Any]) -> "Feedback":
        feedback_value = d.get("feedback")

        if not feedback_value:
            raise MlflowException.invalid_parameter_value(
                "`feedback` must exist in the dictionary."
            )

        feedback_value = FeedbackValue.from_dictionary(feedback_value)

        feedback = cls(
            trace_id=d.get("trace_id"),
            name=d["assessment_name"],
            source=AssessmentSource.from_dictionary(d["source"]),
            create_time_ms=proto_timestamp_to_milliseconds(d["create_time"]),
            last_update_time_ms=proto_timestamp_to_milliseconds(d["last_update_time"]),
            value=feedback_value.value,
            error=feedback_value.error,
            rationale=d.get("rationale"),
            metadata=d.get("metadata"),
            span_id=d.get("span_id"),
            overrides=d.get("overrides"),
            valid=d.get("valid", True),
        )
        feedback.assessment_id = d.get("assessment_id") or None
        return feedback

    # Backward compatibility: The old assessment object had these fields at top level.
    @property
    def error_code(self) -> str | None:
        """The error code of the error that occurred when the feedback was created."""
        return self.feedback.error.error_code if self.feedback.error else None

    @property
    def error_message(self) -> str | None:
        """The error message of the error that occurred when the feedback was created."""
        return self.feedback.error.error_message if self.feedback.error else None


@experimental(version="2.21.0")
@dataclass
class Expectation(Assessment):
    """
    Represents an expectation about the output of an operation, such as the expected response
    that a generative AI application should provide to a particular user query.

    Args:
        name: The name of the assessment.
        value: The expected value of the operation. This can be any JSON-serializable value.
        source: The source of the assessment. If not provided, the default source is HUMAN.
        trace_id: The ID of the trace associated with the assessment. If unset, the assessment
            is not associated with any trace yet.
            should be specified.
        metadata: The metadata associated with the assessment.
        span_id: The ID of the span associated with the assessment, if the assessment should
            be associated with a particular span in the trace.
        create_time_ms: The creation time of the assessment in milliseconds. If unset, the
            current time is used.
        last_update_time_ms: The last update time of the assessment in milliseconds.
            If unset, the current time is used.

    Example:

        .. code-block:: python

            from mlflow.entities import AssessmentSource, Expectation

            expectation = Expectation(
                name="expected_response",
                value="The capital of France is Paris.",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id="john@example.com",
                ),
                metadata={"project": "my-project"},
            )
    """

    def __init__(
        self,
        name: str,
        value: Any,
        source: AssessmentSource | None = None,
        trace_id: str | None = None,
        metadata: dict[str, str] | None = None,
        span_id: str | None = None,
        create_time_ms: int | None = None,
        last_update_time_ms: int | None = None,
    ):
        if source is None:
            source = AssessmentSource(source_type=AssessmentSourceType.HUMAN)

        if value is None:
            raise MlflowException.invalid_parameter_value("The `value` field must be specified.")

        super().__init__(
            name=name,
            source=source,
            trace_id=trace_id,
            metadata=metadata,
            span_id=span_id,
            create_time_ms=create_time_ms,
            last_update_time_ms=last_update_time_ms,
            expectation=ExpectationValue(value=value),
        )

    @property
    def value(self) -> Any:
        return self.expectation.value

    @value.setter
    def value(self, value: Any):
        self.expectation.value = value

    @classmethod
    def from_proto(cls, proto) -> "Expectation":
        # Convert ScalarMapContainer to a normal Python dict
        metadata = dict(proto.metadata) if proto.metadata else None
        expectation_value = ExpectationValue.from_proto(proto.expectation)
        expectation = cls(
            trace_id=proto.trace_id,
            name=proto.assessment_name,
            source=AssessmentSource.from_proto(proto.source),
            create_time_ms=proto.create_time.ToMilliseconds(),
            last_update_time_ms=proto.last_update_time.ToMilliseconds(),
            value=expectation_value.value,
            metadata=metadata,
            span_id=proto.span_id or None,
        )
        expectation.assessment_id = proto.assessment_id or None
        return expectation

    @classmethod
    def from_dictionary(cls, d: dict[str, Any]) -> "Expectation":
        expectation_value = d.get("expectation")

        if not expectation_value:
            raise MlflowException.invalid_parameter_value(
                "`expectation` must exist in the dictionary."
            )

        expectation_value = ExpectationValue.from_dictionary(expectation_value)

        expectation = cls(
            trace_id=d.get("trace_id"),
            name=d["assessment_name"],
            source=AssessmentSource.from_dictionary(d["source"]),
            create_time_ms=proto_timestamp_to_milliseconds(d["create_time"]),
            last_update_time_ms=proto_timestamp_to_milliseconds(d["last_update_time"]),
            value=expectation_value.value,
            metadata=d.get("metadata"),
            span_id=d.get("span_id"),
        )
        expectation.assessment_id = d.get("assessment_id") or None
        return expectation


_JSON_SERIALIZATION_FORMAT = "JSON_FORMAT"


@experimental(version="3.0.0")
@dataclass
class ExpectationValue(_MlflowObject):
    """Represents an expectation value."""

    value: Any

    def to_proto(self):
        if self._need_serialization():
            try:
                serialized_value = json.dumps(self.value)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to serialize value {self.value} to JSON string. "
                    "Expectation value must be JSON-serializable."
                ) from e
            return ProtoExpectation(
                serialized_value=ProtoExpectation.SerializedValue(
                    serialization_format=_JSON_SERIALIZATION_FORMAT,
                    value=serialized_value,
                )
            )

        return ProtoExpectation(value=ParseDict(self.value, Value()))

    @classmethod
    def from_proto(cls, proto) -> "Expectation":
        if proto.HasField("serialized_value"):
            if proto.serialized_value.serialization_format != _JSON_SERIALIZATION_FORMAT:
                raise MlflowException.invalid_parameter_value(
                    f"Unknown serialization format: {proto.serialized_value.serialization_format}. "
                    "Only JSON_FORMAT is supported."
                )
            return cls(value=json.loads(proto.serialized_value.value))
        else:
            return cls(value=MessageToDict(proto.value))

    def to_dictionary(self):
        return MessageToDict(self.to_proto(), preserving_proto_field_name=True)

    @classmethod
    def from_dictionary(cls, d):
        if "value" in d:
            return cls(d["value"])
        elif "serialized_value" in d:
            return cls(value=json.loads(d["serialized_value"]["value"]))
        else:
            raise MlflowException.invalid_parameter_value(
                "Either 'value' or 'serialized_value' must be present in the dictionary "
                "representation of an Expectation."
            )

    def _need_serialization(self):
        # Values like None, lists, dicts, should be serialized as a JSON string
        return self.value is not None and not isinstance(self.value, (int, float, bool, str))


@experimental(version="2.21.0")
@dataclass
class FeedbackValue(_MlflowObject):
    """Represents a feedback value."""

    value: FeedbackValueType
    error: AssessmentError | None = None

    def to_proto(self):
        return ProtoFeedback(
            value=ParseDict(self.value, Value(), ignore_unknown_fields=True),
            error=self.error.to_proto() if self.error else None,
        )

    @classmethod
    def from_proto(cls, proto) -> "FeedbackValue":
        return FeedbackValue(
            value=MessageToDict(proto.value),
            error=AssessmentError.from_proto(proto.error) if proto.HasField("error") else None,
        )

    def to_dictionary(self):
        return MessageToDict(self.to_proto(), preserving_proto_field_name=True)

    @classmethod
    def from_dictionary(cls, d):
        return cls(
            value=d["value"],
            error=AssessmentError.from_dictionary(err) if (err := d.get("error")) else None,
        )
