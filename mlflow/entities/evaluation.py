from mlflow.entities._mlflow_object import _MlflowObject


class Evaluation(_MlflowObject):
    """
    Evaluation result data.
    """

    def __init__(
        self,
        evaluation_id,
        run_id,
        inputs_id,
        inputs,
        outputs,
        request_id=None,
        ground_truths=None,
        feedback=None,
    ):
        """Construct a new mlflow.entities.Evaluation instance.

        Args:
            evaluation_id: A unique identifier for the evaluation result.
            run_id: The ID of the MLflow Run containing the Evaluation.
            inputs_id: A unique identifier for the input names and values for evaluation.
            inputs: Input names and values for evaluation.
            outputs: Outputs obtained during inference.
            request_id: The ID of an MLflow Trace corresponding to the inputs and outputs.
            ground_truths: Expected values that the GenAI app should produce during inference.
            feedback: Feedback for the given row.
        """
        self.evaluation_id = evaluation_id
        self.run_id = run_id
        self.inputs_id = inputs_id
        self.inputs = inputs
        self.outputs = outputs
        self.request_id = request_id
        self.ground_truths = ground_truths or {}
        self.feedback = feedback or []

    def to_dictionary(self):
        return {
            "evaluation_id": self.evaluation_id,
            "run_id": self.run_id,
            "inputs_id": self.inputs_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "request_id": self.request_id,
            "ground_truths": self.ground_truths,
            "feedback": [fb.to_dictionary() for fb in self.feedback],
        }
