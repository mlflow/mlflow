from typing import Any, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


@experimental
class AssessmentSource(_MlflowObject):
    """
    Source of an assessment (human, LLM as a judge with GPT-4, etc).
    """

    def __init__(self, source_type: str, source_id: str, metadata: Optional[dict[str, Any]] = None):
        """Construct a new mlflow.entities.AssessmentSource instance.

        Args:
            source_type: The type of the assessment source (AssessmentSourceType).
            source_id: An identifier for the source, e.g. user ID or LLM judge ID.
            metadata: Additional metadata about the source, e.g. human-readable name, inlined LLM
                judge parameters, etc.
        """
        self._source_type = AssessmentSourceType._standardize(source_type)
        self._source_id = source_id
        self._metadata = metadata or {}

    @property
    def source_type(self) -> str:
        """The type of the assessment source."""
        return self._source_type

    @property
    def source_id(self) -> str:
        """The identifier for the source."""
        return self._source_id

    @property
    def metadata(self) -> dict[str, Any]:
        """The additional metadata about the source."""
        return self._metadata

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self.to_dictionary() == __o.to_dictionary()

        return False

    def to_dictionary(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dictionary(cls, source_dict: dict[str, Any]) -> "AssessmentSource":
        """
        Create a AssessmentSource object from a dictionary.

        Args:
            source_dict (dict): Dictionary containing assessment source information.

        Returns:
            AssessmentSource: The AssessmentSource object created from the dictionary.
        """
        source_type = source_dict["source_type"]
        source_id = source_dict["source_id"]
        metadata = source_dict.get("metadata")
        return cls(source_type=source_type, source_id=source_id, metadata=metadata)


@experimental
class AssessmentSourceType:
    AI_JUDGE = "AI_JUDGE"
    HUMAN = "HUMAN"
    CODE = "CODE"
    _SOURCE_TYPES = [AI_JUDGE, HUMAN, CODE]

    def __init__(self, source_type: str):
        self._source_type = AssessmentSourceType._parse(source_type)

    @staticmethod
    def _parse(source_type: str) -> str:
        source_type = source_type.upper()
        if source_type not in AssessmentSourceType._SOURCE_TYPES:
            raise MlflowException(
                message=(
                    f"Invalid assessment source type: {source_type}. "
                    f"Valid source types: {AssessmentSourceType._SOURCE_TYPES}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return source_type

    def __str__(self):
        return self._source_type

    @staticmethod
    def _standardize(source_type: str) -> str:
        return str(AssessmentSourceType(source_type))
