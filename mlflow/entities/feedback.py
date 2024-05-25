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
        source: FeedbackSource,
        timestamp: int,
        boolean_value: Optional[bool] = None,
        numeric_value: Optional[float] = None,
        string_value: Optional[str] = None,
        rationale: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Construct a new mlflow.entities.Feedback instance.

        Args:
            evaluation_id: The ID of the evaluation result with which the feedback is associated.
            name: The name of the piece of feedback.
            source: The source of the feedback (FeedbackSource instance).
            timestamp: The timestamp when the feedback was given.
            boolean_value: The boolean feedback value, if applicable.
            numeric_value: The numeric feedback value, if applicable.
            string_value: The string feedback value, if applicable.
            rationale: The rationale / justification for the value.
            metadata: Additional metadata for the feedback, e.g. the index of the chunk in the
                      retrieved documents that the feedback applies to.
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

    @property
    def evaluation_id(self) -> str:
        """Get the evaluation ID."""
        return self._evaluation_id

    @property
    def name(self) -> str:
        """Get the name of the feedback."""
        return self._name

    @property
    def timestamp(self) -> int:
        """Get the timestamp of the feedback."""
        return self._timestamp

    @property
    def boolean_value(self) -> Optional[bool]:
        """Get the boolean feedback value."""
        return self._boolean_value

    @property
    def numeric_value(self) -> Optional[float]:
        """Get the numeric feedback value."""
        return self._numeric_value

    @property
    def string_value(self) -> Optional[str]:
        """Get the string feedback value."""
        return self._string_value

    @property
    def rationale(self) -> Optional[str]:
        """Get the rationale / justification for the feedback."""
        return self._rationale

    @property
    def source(self) -> FeedbackSource:
        """Get the source of the feedback."""
        return self._source

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata associated with the feedback."""
        return self._metadata

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self.to_dictionary() == __o.to_dictionary()
        return False

    def to_dictionary(self) -> Dict[str, Any]:
        feedback_dict = {
            "evaluation_id": self.evaluation_id,
            "name": self.name,
            "source": self.source.to_dictionary(),
            "timestamp": self.timestamp,
            "boolean_value": self.boolean_value,
            "numeric_value": self.numeric_value,
            "string_value": self.string_value,
            "rationale": self.rationale,
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
        source_dict = feedback_dict["source"]
        source = FeedbackSource.from_dictionary(source_dict)
        timestamp = feedback_dict["timestamp"]
        boolean_value = feedback_dict.get("boolean_value")
        numeric_value = feedback_dict.get("numeric_value")
        string_value = feedback_dict.get("string_value")
        rationale = feedback_dict.get("rationale")
        metadata = feedback_dict.get("metadata")
        return cls(
            evaluation_id=evaluation_id,
            name=name,
            source=source,
            timestamp=timestamp,
            boolean_value=boolean_value,
            numeric_value=numeric_value,
            string_value=string_value,
            rationale=rationale,
            metadata=metadata,
        )
