"""
THE 'mlflow.evaluation` MODULE IS LEGACY AND WILL BE REMOVED SOON. PLEASE DO NOT USE THESE CLASSES
IN NEW CODE. INSTEAD, USE `mlflow/entities/assessment.py` FOR ASSESSMENT CLASSES.
"""

import numbers
import time
from typing import Any, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated


@deprecated(since="3.0.0", alternative="mlflow.entities.Assessment")
class AssessmentEntity(_MlflowObject):
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
        metadata: Optional[dict[str, str]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        span_id: Optional[str] = None,
    ):
        """Construct a new mlflow.evaluation.AssessmentEntity instance.

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
            span_id: The span ID of the span within the Trace, that the assessment is evaluating,
                     e.g. if you are evaluating a retrieval span for document recall, you can
                     specify the span ID of the retrieval span here. This field is only supported
                     by mlflow.log_feedback.
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
        self._span_id = span_id

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
    def metadata(self) -> dict[str, Any]:
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

    @property
    def span_id(self) -> Optional[str]:
        """The span ID of the span within the Trace, that the assessment is evaluating."""
        return self._span_id

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self.to_dictionary() == __o.to_dictionary()
        return False

    def to_dictionary(self) -> dict[str, Any]:
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
            "span_id": self.span_id,
        }

    @classmethod
    def from_dictionary(cls, assessment_dict: dict[str, Any]) -> "AssessmentEntity":
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
            span_id=assessment_dict.get("span_id"),
        )


@deprecated(since="3.0.0", alternative="mlflow.entities.Assessment")
class Assessment(_MlflowObject):
    """
    Assessment data associated with an evaluation result.

    Assessment is an enriched output from the evaluation that provides more context,
    such as the rationale, source, and metadata for the evaluation result.

    Example:

    .. code-block:: python

        from mlflow.evaluation import Assessment

        assessment = Assessment(
            name="answer_correctness",
            value=0.5,
            rationale="The answer is partially correct.",
        )
    """

    def __init__(
        self,
        name: str,
        source: Optional[AssessmentSource] = None,
        value: Optional[Union[bool, float, str]] = None,
        rationale: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
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
        self._source = source or AssessmentSource(
            source_type=AssessmentSourceType.SOURCE_TYPE_UNSPECIFIED,
            source_id="unknown",
        )
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
    def metadata(self) -> dict[str, Any]:
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

    def to_dictionary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source.to_dictionary() if self.source is not None else None,
            "value": self.value,
            "rationale": self.rationale,
            "metadata": self.metadata,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dictionary(cls, assessment_dict: dict[str, Any]) -> "Assessment":
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
        # We require that the source be specified for an assessment before sending it to the backend
        if self._source is None:
            raise MlflowException(
                message=(
                    f"Assessment source must be specified."
                    f"Got empty source for assessment with name {self._name}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
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
