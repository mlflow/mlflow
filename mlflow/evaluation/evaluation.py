import hashlib
import json
from typing import Any, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.evaluation import Evaluation as EvaluationEntity
from mlflow.entities.evaluation_tag import EvaluationTag  # Assuming EvaluationTag is in this module
from mlflow.entities.metric import Metric
from mlflow.evaluation.assessment import Assessment
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.annotations import experimental


@experimental
class Evaluation(_MlflowObject):
    """
    Evaluation result data.
    """

    def __init__(
        self,
        inputs: dict[str, Any],
        outputs: Optional[dict[str, Any]] = None,
        inputs_id: Optional[str] = None,
        request_id: Optional[str] = None,
        targets: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        assessments: Optional[list[Assessment]] = None,
        metrics: Optional[Union[dict[str, float], list[Metric]]] = None,
        tags: Optional[dict[str, str]] = None,
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
            assessments: Assessments for the evaluation.
            metrics: Objective numerical metrics for the evaluation, e.g., "number of input tokens",
                "number of output tokens".
            tags: Dictionary of tags associated with the evaluation.
        """
        if isinstance(metrics, dict):
            metrics = [
                Metric(key=key, value=value, timestamp=0, step=0) for key, value in metrics.items()
            ]
        if isinstance(tags, dict):
            tags = [EvaluationTag(key=str(key), value=str(value)) for key, value in tags.items()]

        self._inputs = inputs
        self._outputs = outputs
        self._inputs_id = inputs_id or _generate_inputs_id(inputs)
        self._request_id = request_id
        self._targets = targets
        self._error_code = error_code
        self._error_message = error_message
        self._assessments = assessments
        self._metrics = metrics
        self._tags = tags

    @property
    def inputs_id(self) -> str:
        """The evaluation inputs ID."""
        return self._inputs_id

    @property
    def inputs(self) -> dict[str, Any]:
        """The evaluation inputs."""
        return self._inputs

    @property
    def outputs(self) -> Optional[dict[str, Any]]:
        """The evaluation outputs."""
        return self._outputs

    @property
    def request_id(self) -> Optional[str]:
        """The evaluation request ID."""
        return self._request_id

    @property
    def targets(self) -> Optional[dict[str, Any]]:
        """The evaluation targets."""
        return self._targets

    @property
    def error_code(self) -> Optional[str]:
        """The evaluation error code."""
        return self._error_code

    @property
    def error_message(self) -> Optional[str]:
        """The evaluation error message."""
        return self._error_message

    @property
    def assessments(self) -> Optional[list[Assessment]]:
        """The evaluation assessments."""
        return self._assessments

    @property
    def metrics(self) -> Optional[list[Metric]]:
        """The evaluation metrics."""
        return self._metrics

    @property
    def tags(self) -> Optional[dict[str, str]]:
        """The evaluation tags."""
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

    def to_dictionary(self) -> dict[str, Any]:
        """
        Convert the Evaluation object to a dictionary.

        Returns:
            dict: The Evaluation object represented as a dictionary.
        """
        evaluation_dict = {
            "inputs_id": self.inputs_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "request_id": self.request_id,
            "targets": self.targets,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "assessments": [assess.to_dictionary() for assess in self.assessments]
            if self.assessments
            else None,
            "metrics": [metric.to_dictionary() for metric in self.metrics]
            if self.metrics
            else None,
            "tags": [tag.to_dictionary() for tag in self.tags] if self.tags else None,
        }
        return {k: v for k, v in evaluation_dict.items() if v is not None}

    @classmethod
    def from_dictionary(cls, evaluation_dict: dict[str, Any]):
        """
        Create an Evaluation object from a dictionary.

        Args:
            evaluation_dict (dict): Dictionary containing evaluation information.

        Returns:
            Evaluation: The Evaluation object created from the dictionary.
        """
        assessments = None
        if "assessments" in evaluation_dict:
            assessments = [
                Assessment.from_dictionary(assess) for assess in evaluation_dict["assessments"]
            ]
        metrics = None
        if "metrics" in evaluation_dict:
            metrics = [Metric.from_dictionary(metric) for metric in evaluation_dict["metrics"]]
        tags = None
        if "tags" in evaluation_dict:
            tags = [EvaluationTag(tag["key"], tag["value"]) for tag in evaluation_dict["tags"]]
        return cls(
            inputs_id=evaluation_dict["inputs_id"],
            inputs=evaluation_dict["inputs"],
            outputs=evaluation_dict.get("outputs"),
            request_id=evaluation_dict.get("request_id"),
            targets=evaluation_dict.get("targets"),
            error_code=evaluation_dict.get("error_code"),
            error_message=evaluation_dict.get("error_message"),
            assessments=assessments,
            metrics=metrics,
            tags=tags,
        )


def _generate_inputs_id(inputs: dict[str, Any]) -> str:
    """
    Generates a unique identifier for the inputs.

    Args:
        inputs (Dict[str, Any]): Input fields used by the model to compute outputs.

    Returns:
        str: A unique identifier for the inputs.
    """
    inputs_json = json.dumps(inputs, sort_keys=True, cls=TraceJSONEncoder)
    return hashlib.sha256(inputs_json.encode("utf-8")).hexdigest()
