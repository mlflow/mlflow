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
        value: Union[bool, float, str],
        rationale: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Construct a new Assessment instance.

        Args:
            name: The name of the piece of assessment.
            source: The source of the assessment (AssessmentSource instance).
            value: The value of the assessment. This can be a boolean, numeric, or string value.
            rationale: The rationale / justification for the value.
            metadata: Additional metadata for the assessment, e.g. the index of the chunk in the
                      retrieved documents that the assessment applies to.
        """
        self._name = name
        self._source = source
        self._value = value
        self._rationale = rationale
        self._metadata = metadata or {}

        self._boolean_value = None
        self._numeric_value = None
        self._string_value = None
        if isinstance(value, bool):
            self._boolean_value = value
        elif isinstance(value, float):
            self._numeric_value = value
        elif value is not None:
            self._string_value = str(value)
        else:
            raise MlflowException(
                "Assessment must specify a boolean, numeric, or string value.",
                INVALID_PARAMETER_VALUE,
            )

    @property
    def name(self) -> str:
        """Get the name of the assessment."""
        return self._name

    @property
    def value(self) -> Union[bool, float, str]:
        """Get the assessment value."""
        return self._value

    @property
    def rationale(self) -> Optional[str]:
        """Get the rationale / justification for the assessment."""
        return self._rationale

    @property
    def source(self) -> AssessmentSource:
        """Get the source of the assessment."""
        return self._source

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata associated with the assessment."""
        return self._metadata

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self.to_dictionary() == __o.to_dictionary()
        return False

    def to_dictionary(self) -> Dict[str, Any]:
        assessment_dict = {
            "name": self.name,
            "source": self.source.to_dictionary(),
            "value": self.value,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }
        # Remove keys with None values
        return {k: v for k, v in assessment_dict.items() if v is not None}

    @classmethod
    def from_dictionary(cls, assessment_dict: Dict[str, Any]) -> "Assessment":
        """
        Create a Assessment object from a dictionary.

        Args:
            assessment_dict (dict): Dictionary containing assessment information.

        Returns:
            Assessment: The Assessment object created from the dictionary.
        """
        name = assessment_dict["name"]
        source_dict = assessment_dict["source"]
        source = AssessmentSource.from_dictionary(source_dict)
        rationale = assessment_dict.get("rationale")
        metadata = assessment_dict.get("metadata")
        value = assessment_dict.get("value")
        return cls(
            name=name,
            source=source,
            value=value,
            rationale=rationale,
            metadata=metadata,
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
        )
