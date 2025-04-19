from dataclasses import dataclass
from typing import Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.assessments_pb2 import AssessmentError as ProtoAssessmentError
from mlflow.utils.annotations import experimental


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
    """

    error_code: str
    error_message: Optional[str] = None

    def to_proto(self):
        error = ProtoAssessmentError()
        error.error_code = self.error_code
        if self.error_message:
            error.error_message = self.error_message
        return error

    @classmethod
    def from_proto(cls, proto):
        return cls(
            error_code=proto.error_code,
            error_message=proto.error_message or None,
        )

    def to_dictionary(self):
        return {"error_code": self.error_code, "error_message": self.error_message}

    @classmethod
    def from_dictionary(cls, error_dict):
        return cls(**error_dict)
