from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class Assessment(_MlflowObject):
    """
    Assessment data associated with an evaluation.
    """

    def __init__(
        self,
        evaluation_id: str,
        name: str,
        source: AssessmentSource,
        timestamp: int,
        boolean_value: Optional[bool] = None,
        numeric_value: Optional[float] = None,
        string_value: Optional[str] = None,
        rationale: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Construct a new mlflow.entities.Assessment instance.

        Args:
            evaluation_id: The ID of the evaluation with which the assessment is associated.
            name: The name of the assessment.
            source: The source of the assessment (AssessmentSource instance).
            timestamp: The timestamp when the assessment was given.
            boolean_value: The boolean assessment value, if applicable.
            numeric_value: The numeric assessment value, if applicable.
            string_value: The string assessment value, if applicable.
            rationale: The rationale / justification for the value.
            metadata: Additional metadata for the assessment, e.g. the index of the chunk in the
                      retrieved documents that the assessment applies to.
            error_code: An error code representing any issues encountered during the assessment.
            error_message: A descriptive error message representing any issues encountered during
                the assessment.
        """
        self._evaluation_id = evaluation_id
        self._name = name
        self._source = source
        self._timestamp = timestamp
        self._boolean_value = boolean_value
        self._numeric_value = numeric_value
        self._string_value = string_value
        self._rationale = rationale
        self._metadata = metadata or {}
        self._error_code = error_code
        self._error_message = error_message

        if error_message is not None and (
            boolean_value is not None or numeric_value is not None or string_value is not None
        ):
            raise MlflowException(
                "error_message cannot be specified when boolean_value, numeric_value, "
                "or string_value is specified.",
                INVALID_PARAMETER_VALUE,
            )

        if (self._boolean_value, self._string_value, self._numeric_value, self._error_code).count(
            None
        ) != 3:
            raise MlflowException(
                "Exactly one of boolean_value, numeric_value, string_value, or error_code must be "
                "specified for an assessment.",
                INVALID_PARAMETER_VALUE,
            )

    @property
    def evaluation_id(self) -> str:
        """The evaluation ID."""
        return self._evaluation_id

    @property
    def name(self) -> str:
        """The name of the assessment."""
        return self._name

    @property
    def timestamp(self) -> int:
        """The timestamp of the assessment."""
        return self._timestamp

    @property
    def boolean_value(self) -> Optional[bool]:
        """The boolean assessment value."""
        return self._boolean_value

    @property
    def numeric_value(self) -> Optional[float]:
        """The numeric assessment value."""
        return self._numeric_value

    @property
    def string_value(self) -> Optional[str]:
        """The string assessment value."""
        return self._string_value

    @property
    def rationale(self) -> Optional[str]:
        """The rationale / justification for the assessment."""
        return self._rationale

    @property
    def source(self) -> AssessmentSource:
        """The source of the assessment."""
        return self._source

    @property
    def metadata(self) -> Dict[str, Any]:
        """The metadata associated with the assessment."""
        return self._metadata

    @property
    def error_code(self) -> Optional[str]:
        """The error code."""
        return self._error_code

    @property
    def error_message(self) -> Optional[str]:
        """The error message."""
        return self._error_message

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self.to_dictionary() == __o.to_dictionary()
        return False

    def to_dictionary(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "name": self.name,
            "source": self.source.to_dictionary(),
            "timestamp": self.timestamp,
            "boolean_value": self.boolean_value,
            "numeric_value": self.numeric_value,
            "string_value": self.string_value,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dictionary(cls, assessment_dict: Dict[str, Any]) -> "Assessment":
        """
        Create an Assessment object from a dictionary.

        Args:
            assessment_dict (dict): Dictionary containing assessment information.

        Returns:
            Assessment: The Assessment object created from the dictionary.
        """
        return cls(
            evaluation_id=assessment_dict["evaluation_id"],
            name=assessment_dict["name"],
            source=AssessmentSource.from_dictionary(assessment_dict["source"]),
            timestamp=assessment_dict["timestamp"],
            boolean_value=assessment_dict.get("boolean_value"),
            numeric_value=assessment_dict.get("numeric_value"),
            string_value=assessment_dict.get("string_value"),
            rationale=assessment_dict.get("rationale"),
            metadata=assessment_dict.get("metadata"),
            error_code=assessment_dict.get("error_code"),
            error_message=assessment_dict.get("error_message"),
        )
