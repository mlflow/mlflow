from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MlflowObject


class FeedbackSource(_MlflowObject):
    """
    Source of the feedback (human, LLM as a judge with GPT-4, etc).
    """

    def __init__(self, source_type: str, source_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Construct a new mlflow.entities.FeedbackSource instance.

        Args:
            source_type: The type of the feedback source (FeedbackSourceType).
            source_id: An identifier for the source, e.g. Databricks user ID or LLM judge ID.
            metadata: Additional metadata about the source, e.g. human-readable name, inlined LLM
            judge parameters, etc.
        """
        self.source_type = source_type
        self.source_id = source_id
        self.metadata = metadata or {}

    def to_dictionary(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dictionary(cls, source_dict: Dict[str, Any]) -> "FeedbackSource":
        """
        Create a FeedbackSource object from a dictionary.

        Args:
            source_dict (dict): Dictionary containing feedback source information.

        Returns:
            FeedbackSource: The FeedbackSource object created from the dictionary.
        """
        source_type = source_dict["source_type"]
        source_id = source_dict["source_id"]
        metadata = source_dict.get("metadata")
        return cls(source_type=source_type, source_id=source_id, metadata=metadata)


class FeedbackSourceType:
    AI_JUDGE = "AI_JUDGE"
    HUMAN = "HUMAN"
