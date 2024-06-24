import hashlib
import json
from typing import Any, Dict, List, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.evaluation import Evaluation as EvaluationEntity
from mlflow.entities.evaluation_tag import EvaluationTag  # Assuming EvaluationTag is in this module
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
        outputs: Optional[Dict[str, Any]] = None,
        inputs_id: Optional[str] = None,
        request_id: Optional[str] = None,
        targets: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        assessments: Optional[List[Assessment]] = None,
        metrics: Optional[Union[Dict[str, float], List[Metric]]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Construct a new Evaluation instance.

        Args:
            inputs: Input names and values for evaluation.
            outputs: Outputs obtained during inference.
            inputs_id: A unique identifier for the input names and values for evaluation.
            request_id: The ID of an MLflow Trace corresponding to the inputs and outputs.
            targets: Expected values that the model should produce during inference.
            error_code: An error code representing any issues encountered during the evaluation.
            error_message: A descriptive error message representing any issues encountered during
                the evaluation.
            assessments: Assessments for the given row.
            metrics: Objective numerical metrics for the row, e.g., "number of input tokens",
                "number of output tokens".
            tags: Dictionary of tags associated with the evaluation.
        """
        if isinstance(metrics, dict):
            metrics = [
                Metric(key=key, value=value, timestamp=0, step=0) for key, value in metrics.items()
            ]
        if isinstance(tags, dict):
            tags = [EvaluationTag(key=key, value=value) for key, value in tags.items()]

        self._inputs_id = inputs_id or _generate_inputs_id(inputs)
        self._inputs = inputs
        self._outputs = outputs
        self._request_id = request_id
        self._targets = targets
        self._error_code = error_code
        self._error_message = error_message
        self._assessments = assessments
        self._metrics = metrics
        self._tags = tags

    @property
    def inputs_id(self) -> str:
        """Get the inputs ID."""
        return self._inputs_id

    @property
    def inputs(self) -> Dict[str, Any]:
        """Get the inputs."""
        return self._inputs

    @property
    def outputs(self) -> Optional[Dict[str, Any]]:
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
    def error_code(self) -> Optional[str]:
        """Get the error code."""
        return self._error_code

    @property
    def error_message(self) -> Optional[str]:
        """Get the error message."""
        return self._error_message

    @property
    def assessments(self) -> Optional[List[Assessment]]:
        """Get the assessments."""
        return self._assessments

    @property
    def metrics(self) -> Optional[List[Metric]]:
        """Get the metrics."""
        return self._metrics

    @property
    def tags(self) -> Optional[Dict[str, str]]:
        """Get the tags."""
        return self._tags

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
            error_code=self.error_code,
            error_message=self.error_message,
            assessments=[assess._to_entity(evaluation_id) for assess in self.assessments]
            if self.assessments
            else None,
            metrics=self.metrics,
            tags=self.tags,
        )

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Convert the Evaluation object to a dictionary.

        Returns:
            dict: The Evaluation object represented as a dictionary.
        """
        evaluation_dict = {
            "inputs_id": self.inputs_id,
            "inputs": self.inputs,
        }
        if self.outputs:
            evaluation_dict["outputs"] = self.outputs
        if self.request_id:
            evaluation_dict["request_id"] = self.request_id
        if self.targets:
            evaluation_dict["targets"] = self.targets
        if self.error_code:
            evaluation_dict["error_code"] = self.error_code
        if self.error_message:
            evaluation_dict["error_message"] = self.error_message
        if self.assessments:
            evaluation_dict["assessments"] = [assess.to_dictionary() for assess in self.assessments]
        if self.metrics:
            evaluation_dict["metrics"] = [metric.to_dictionary() for metric in self.metrics]
        if self.tags:
            evaluation_dict["tags"] = [tag.to_dictionary() for tag in self.tags]
        return evaluation_dict


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
