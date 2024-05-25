import time
from typing import Any, Dict, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.feedback import Feedback as FeedbackEntity
from mlflow.entities.feedback_source import FeedbackSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class Feedback(_MlflowObject):
    """
    Feedback data associated with an evaluation result.
    """

    def __init__(
        self,
        name: str,
        source: FeedbackSource,
        value: Union[bool, float, str],
        rationale: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Construct a new Feedback instance.

        Args:
            name: The name of the piece of feedback.
            source: The source of the feedback (FeedbackSource instance).
            value: The value of the feedback. This can be a boolean, numeric, or string value.
            rationale: The rationale / justification for the value.
            metadata: Additional metadata for the feedback, e.g. the index of the chunk in the
                      retrieved documents that the feedback applies to.
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
                "Feedback must specify a boolean, numeric, or string value.",
                INVALID_PARAMETER_VALUE,
            )

    @property
    def name(self) -> str:
        """Get the name of the feedback."""
        return self._name

    @property
    def value(self) -> Union[bool, float, str]:
        """Get the feedback value."""
        return self._value

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
            "name": self.name,
            "source": self.source.to_dictionary(),
            "value": self.value,
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
        name = feedback_dict["name"]
        source_dict = feedback_dict["source"]
        source = FeedbackSource.from_dictionary(source_dict)
        rationale = feedback_dict.get("rationale")
        metadata = feedback_dict.get("metadata")
        value = feedback_dict.get("value")
        return cls(
            name=name,
            source=source,
            value=value,
            rationale=rationale,
            metadata=metadata,
        )

    def _to_entity(self, evaluation_id: str) -> FeedbackEntity:
        return FeedbackEntity(
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
