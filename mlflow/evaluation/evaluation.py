import hashlib
import json
from typing import Any, Dict, List, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.evaluation import Evaluation as EvaluationEntity
from mlflow.entities.metric import Metric
from mlflow.evaluation.assessment import Assessment
from mlflow.tracing.utils import TraceJSONEncoder


class Evaluation(_MlflowObject):
    """
    Evaluation result data.
    """

    def __init__(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        inputs_id: Optional[str] = None,
        request_id: Optional[str] = None,
        targets: Optional[Dict[str, Any]] = None,
        assessments: Optional[List[Assessment]] = None,
        metrics: Optional[List[Metric]] = None,
    ):
        """
        Construct a new Evaluation instance.

        Args:
            inputs_id: A unique identifier for the input names and values for evaluation.
            inputs: Input names and values for evaluation.
            outputs: Outputs obtained during inference.
            request_id: The ID of an MLflow Trace corresponding to the inputs and outputs.
            targets: Expected values that the model should produce during inference.
            assessments: Assessments for the given row.
            metrics: Objective numerical metrics for the row, e.g., "number of input tokens",
                "number of output tokens".
        """
        self._inputs_id = inputs_id or _generate_inputs_id(inputs)
        self._inputs = inputs
        self._outputs = outputs
        self._request_id = request_id
        self._targets = targets
        self._assessments = assessments
        self._metrics = metrics

    @property
    def inputs_id(self) -> str:
        """Get the inputs ID."""
        return self._inputs_id

    @property
    def inputs(self) -> Dict[str, Any]:
        """Get the inputs."""
        return self._inputs

    @property
    def outputs(self) -> Dict[str, Any]:
        """Get the outputs."""
        return self._outputs

    @property
    def request_id(self) -> Optional[str]:
        """Get the request ID."""
        return self._request_id

    @property
    def targets(self) -> Optional[Dict[str, Any]]:
        """Get the targets."""
        return self._targets

    @property
    def assessments(self) -> Optional[List[Assessment]]:
        """Get the assessments."""
        return self._assessments

    @property
    def metrics(self) -> Optional[List[Metric]]:
        """Get the metrics."""
        return self._metrics

    def __eq__(self, __o):
        if isinstance(__o, self.__class__):
            return self.to_dictionary() == __o.to_dictionary()

        return False

    def _to_entity(self, run_id: str, evaluation_id: str) -> EvaluationEntity:
        """
        Convert the Evaluation object to an EvaluationEntity object.

        Returns:
            EvaluationEntity: An EvaluationEntity object.
        """
        return EvaluationEntity(
            evaluation_id=evaluation_id,
            run_id=run_id,
            inputs_id=self.inputs_id,
            inputs=self.inputs,
            outputs=self.outputs,
            request_id=self.request_id,
            targets=self.targets,
            assessments=[assess._to_entity(evaluation_id) for assess in self.assessments]
            if self.assessments
            else None,
            metrics=self.metrics,
        )


def _generate_inputs_id(inputs: Dict[str, Any]) -> str:
    """
    Generates a unique identifier for the inputs.

    Args:
        inputs (Dict[str, Any]): Input fields used by the model to compute outputs.

    Returns:
        str: A unique identifier for the inputs.
    """
    inputs_json = json.dumps(inputs, sort_keys=True, cls=TraceJSONEncoder)
    return hashlib.sha256(inputs_json.encode("utf-8")).hexdigest()
