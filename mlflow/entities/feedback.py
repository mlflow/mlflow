from mlflow.entities._mlflow_object import _MlflowObject


class Feedback(_MlflowObject):
    """
    Feedback data associated with an evaluation result.
    """

    def __init__(
        self,
        evaluation_id,
        name,
        boolean_value=None,
        numeric_value=None,
        string_value=None,
        rationale=None,
        source=None,
        metadata=None,
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

    def to_dictionary(self):
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
