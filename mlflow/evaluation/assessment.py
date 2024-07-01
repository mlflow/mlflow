import numbers
import time
from typing import Any, Dict, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import Assessment as AssessmentEntity
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class Assessment(_MlflowObject):
    """
    Assessment data associated with an evaluation result.
    """

    def __init__(
        self,
        name: str,
        source: AssessmentSource,
        value: Optional[Union[bool, float, str]] = None,
        rationale: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Construct a new Assessment instance.

        Args:
            name: The name of the piece of assessment.
            source: The source of the assessment (AssessmentSource instance).
            value: The value of the assessment. This can be a boolean, numeric, or string value.
            rationale: The rationale / justification for the value.
            metadata: Additional metadata for the assessment, e.g. the index of the chunk in the
                      retrieved documents that the assessment applies to.
            error_code: An error code representing any issues encountered during the assessment.
            error_message: A descriptive error message representing any issues encountered during
                the assessment.
        """
        if (value is None) == (error_code is None):
            raise MlflowException(
                "Exactly one of value or error_code must be specified for an assessment.",
                INVALID_PARAMETER_VALUE,
            )

        if value is not None and error_message is not None:
            raise MlflowException(
                "error_message cannot be specified when value is specified.",
                INVALID_PARAMETER_VALUE,
            )

        self._name = name
        self._source = source
        self._value = value
        self._rationale = rationale
        self._metadata = metadata or {}
        self._error_code = error_code
        self._error_message = error_message

        self._boolean_value = None
        self._numeric_value = None
        self._string_value = None
        if isinstance(value, bool):
            self._boolean_value = value
        elif isinstance(value, numbers.Number):
            self._numeric_value = float(value)
        elif value is not None:
            self._string_value = str(value)

    @property
    def name(self) -> str:
        """The name of the assessment."""
        return self._name

    @property
    def value(self) -> Union[bool, float, str]:
        """The assessment value."""
        return self._value

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
            "name": self.name,
            "source": self.source.to_dictionary(),
            "value": self.value,
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
            name=assessment_dict["name"],
            source=AssessmentSource.from_dictionary(assessment_dict["source"]),
            value=assessment_dict.get("value"),
            rationale=assessment_dict.get("rationale"),
            metadata=assessment_dict.get("metadata"),
            error_code=assessment_dict.get("error_code"),
            error_message=assessment_dict.get("error_message"),
        )

    def _to_entity(self, evaluation_id: str) -> AssessmentEntity:
        return AssessmentEntity(
            evaluation_id=evaluation_id,
            name=self._name,
            source=self._source,
            timestamp=int(time.time() * 1000),
            boolean_value=self._boolean_value,
            numeric_value=self._numeric_value,
            string_value=self._string_value,
            rationale=self._rationale,
            metadata=self._metadata,
            error_code=self._error_code,
            error_message=self._error_message,
        )
