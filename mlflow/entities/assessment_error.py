from dataclasses import dataclass
from typing import Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.assessments_pb2 import AssessmentError as ProtoAssessmentError
from mlflow.utils.annotations import experimental

_STACK_TRACE_TRUNCATION_PREFIX = "[Stack trace is truncated]\n...\n"
_STACK_TRACE_TRUNCATION_LENGTH = 1000


@experimental
@dataclass
class AssessmentError(_MlflowObject):
    """
    Error object representing any issues during generating the assessment.

    For example, if the LLM-as-a-Judge fails to generate an feedback, you can
    log an error with the error code and message as shown below:

    .. code-block:: python

        from mlflow.entities import AssessmentError

        error = AssessmentError(
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Rate limit for the judge exceeded.",
            stack_trace="...",
        )

        mlflow.log_feedback(
            trace_id="1234",
            name="faithfulness",
            source=AssessmentSourceType.LLM_JUDGE,
            error=error,
            # Skip setting value when an error is present
        )

    Args:
        error_code: The error code.
        error_message: The detailed error message. Optional.
        stack_trace: The stack trace of the error. Truncated to 1000 characters
            before being logged to MLflow. Optional.
    """

    error_code: str
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_proto(self):
        error = ProtoAssessmentError()
        error.error_code = self.error_code
        if self.error_message:
            error.error_message = self.error_message
        if self.stack_trace:
            if len(self.stack_trace) > _STACK_TRACE_TRUNCATION_LENGTH:
                trunc_len = _STACK_TRACE_TRUNCATION_LENGTH - len(_STACK_TRACE_TRUNCATION_PREFIX)
                error.stack_trace = _STACK_TRACE_TRUNCATION_PREFIX + self.stack_trace[-trunc_len:]
            else:
                error.stack_trace = self.stack_trace
        return error

    @classmethod
    def from_proto(cls, proto):
        return cls(
            error_code=proto.error_code,
            error_message=proto.error_message or None,
            stack_trace=proto.stack_trace or None,
        )

    def to_dictionary(self):
        return {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
        }

    @classmethod
    def from_dictionary(cls, error_dict):
        return cls(**error_dict)
