from typing import Dict

from mlflow.entities._mlflow_object import _MlflowObject


class EvaluationTag(_MlflowObject):
    """Key-value tag associated with an evaluation."""

    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self):
        """String name of the tag."""
        return self._key

    @property
    def value(self):
        """String value of the tag."""
        return self._value

    def to_dictionary(self) -> Dict[str, str]:
        """
        Convert the EvaluationTag object to a dictionary.

        Returns:
            dict: The EvaluationTag object represented as a dictionary.
        """
        return {
            "key": self.key,
            "value": self.value,
        }

    @classmethod
    def from_dictionary(cls, evaluation_tag_dict: Dict[str, str]):
        """
        Create an EvaluationTag object from a dictionary.

        Args:
            evaluation_tag_dict (dict): Dictionary containing evaluation tag information.

        Returns:
            Evaluation: The EvaluationTag object created from the dictionary.
        """
        key = evaluation_tag_dict["key"]
        value = evaluation_tag_dict["value"]
        return cls(
            key=key,
            value=value,
        )
