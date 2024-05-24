from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.feedback_source import FeedbackSource


class Feedback(_MlflowObject):
    """
    Feedback data associated with an evaluation result.
    """

    def __init__(
        self,
        evaluation_id: str,
        name: str,
        boolean_value: Optional[bool] = None,
        numeric_value: Optional[float] = None,
        string_value: Optional[str] = None,
        rationale: Optional[str] = None,
        source: Optional[FeedbackSource] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Construct a new mlflow.entities.Feedback instance.

        Args:
            evaluation_id: The ID of the evaluation result with which the feedback is associated.
            name: The name of the piece of feedback.
            boolean_value: The boolean feedback value, if applicable.
            numeric_value: The numeric feedback value, if applicable.
            string_value: The string feedback value, if applicable.
            rationale: The rationale / justification for the value.
            source: The source of the feedback (FeedbackSource instance).
            metadata: Additional metadata for the feedback, e.g. the index of the chunk in the
                      retrieved documents that the feedback applies to.
        """
        self.evaluation_id = evaluation_id
        self.name = name
        self.boolean_value = boolean_value
        self.numeric_value = numeric_value
        self.string_value = string_value
        self.rationale = rationale
        self.source = source
        self.metadata = metadata or {}

    def to_dictionary(self) -> Dict[str, Any]:
        feedback_dict = {
            "evaluation_id": self.evaluation_id,
            "name": self.name,
            "boolean_value": self.boolean_value,
            "numeric_value": self.numeric_value,
            "string_value": self.string_value,
            "rationale": self.rationale,
            "source": self.source.to_dictionary() if self.source else None,
            "metadata": self.metadata,
        }
        # Remove keys with None values
        return {k: v for k, v in feedback_dict.items() if v is not None}

    @classmethod
    def from_dictionary(cls, feedback_dict: Dict[str, Any]) -> "Feedback":
        """
        Create a Feedback object from a dictionary.

        Args:
            feedback_dict (dict): Dictionary containing feedback information.

        Returns:
            Feedback: The Feedback object created from the dictionary.
        """
        evaluation_id = feedback_dict["evaluation_id"]
        name = feedback_dict["name"]
        boolean_value = feedback_dict.get("boolean_value")
        numeric_value = feedback_dict.get("numeric_value")
        string_value = feedback_dict.get("string_value")
        rationale = feedback_dict.get("rationale")
        source_dict = feedback_dict.get("source")
        source = FeedbackSource.from_dictionary(source_dict) if source_dict else None
        metadata = feedback_dict.get("metadata")
        return cls(
            evaluation_id=evaluation_id,
            name=name,
            boolean_value=boolean_value,
            numeric_value=numeric_value,
            string_value=string_value,
            rationale=rationale,
            source=source,
            metadata=metadata,
        )
