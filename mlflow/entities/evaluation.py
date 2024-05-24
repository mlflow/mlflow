from typing import Any, Dict, List, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.feedback import Feedback
from mlflow.entities.metric import Metric


class Evaluation(_MlflowObject):
    """
    Evaluation result data.
    """

    def __init__(
        self,
        evaluation_id: str,
        run_id: str,
        inputs_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        request_id: Optional[str] = None,
        ground_truths: Optional[Dict[str, Any]] = None,
        feedback: Optional[List[Feedback]] = None,
        metrics: Optional[List[Metric]] = None,
    ):
        """
        Construct a new mlflow.entities.Evaluation instance.

        Args:
            evaluation_id: A unique identifier for the evaluation result.
            run_id: The ID of the MLflow Run containing the Evaluation.
            inputs_id: A unique identifier for the input names and values for evaluation.
            inputs: Input names and values for evaluation.
            outputs: Outputs obtained during inference.
            request_id: The ID of an MLflow Trace corresponding to the inputs and outputs.
            ground_truths: Expected values that the GenAI app should produce during inference.
            feedback: Feedback for the given row.
            metrics: Objective numerical metrics for the row, e.g., "number of input tokens",
                "number of output tokens".
        """
        self.evaluation_id = evaluation_id
        self.run_id = run_id
        self.inputs_id = inputs_id
        self.inputs = inputs
        self.outputs = outputs
        self.request_id = request_id
        self.ground_truths = ground_truths
        self.feedback = feedback
        self.metrics = metrics

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Convert the Evaluation object to a dictionary.

        Returns:
            dict: The Evaluation object represented as a dictionary.
        """
        evaluation_dict = {
            "evaluation_id": self.evaluation_id,
            "run_id": self.run_id,
            "inputs_id": self.inputs_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        if self.request_id:
            evaluation_dict["request_id"] = self.request_id
        if self.ground_truths:
            evaluation_dict["ground_truths"] = self.ground_truths
        if self.feedback:
            evaluation_dict["feedback"] = [fb.__dict__ for fb in self.feedback]
        if self.metrics:
            evaluation_dict["metrics"] = self.metrics
        return evaluation_dict
