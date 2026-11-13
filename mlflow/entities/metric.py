from __future__ import annotations

import numbers
from decimal import Decimal
from typing import Any, cast

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import Metric as ProtoMetric
from mlflow.protos.service_pb2 import MetricWithRunId as ProtoMetricWithRunId


def _normalize_metric_field(name: str, value: Any) -> Any:
    """
    Normalize numeric timestamps/steps to plain ints so that the metric store can
    always persist and read them back. Fractional values used to be accepted and
    persisted verbatim, after which every read of the metric failed because the file
    store parses them back with int(). Non-numeric values are left untouched; they are
    rejected by ``_validate_metric`` at store write time.
    """
    # bool is an instance of numbers.Integral but is never a valid timestamp/step, and
    # _validate_metric cannot catch it because bool is a numbers.Number
    if isinstance(value, bool):
        raise MlflowException(
            f"Metric {name} must be an integer, but got {value!r} of type bool",
            INVALID_PARAMETER_VALUE,
        )
    if isinstance(value, numbers.Integral):
        return int(value)
    # Decimal is a numbers.Number but not a numbers.Real, so it needs its own branch;
    # str(Decimal("2.0")) cannot be parsed back by int()
    if isinstance(value, Decimal):
        if value.is_finite() and value == value.to_integral_value():
            return int(value)
        raise MlflowException(
            f"Metric {name} must be an integer, but got {value!r} of type Decimal",
            INVALID_PARAMETER_VALUE,
        )
    if isinstance(value, numbers.Real):
        as_float = float(value)
        # Integrations may emit integral floats (e.g. PyTorch Lightning logging epoch
        # steps as floats); normalize them instead of rejecting the value
        if as_float.is_integer():
            return int(as_float)
        raise MlflowException(
            f"Metric {name} must be an integer, but got {value!r} of type {type(value).__name__}",
            INVALID_PARAMETER_VALUE,
        )
    return value


class Metric(_MlflowObject):
    """
    Metric object.
    """

    def __init__(
        self,
        key: str,
        value: float,
        timestamp: int | float | Decimal,
        step: int | float | Decimal,
        model_id: str | None = None,
        dataset_name: str | None = None,
        dataset_digest: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if (dataset_name, dataset_digest).count(None) == 1:
            raise MlflowException(
                "Both dataset_name and dataset_digest must be provided if one is provided",
                INVALID_PARAMETER_VALUE,
            )

        self._key = key
        self._value = value
        self._timestamp: int = _normalize_metric_field("timestamp", timestamp)
        self._step: int = _normalize_metric_field("step", step)
        self._model_id = model_id
        self._dataset_name = dataset_name
        self._dataset_digest = dataset_digest
        self._run_id = run_id

    @property
    def key(self) -> str:
        """String key corresponding to the metric name."""
        return self._key

    @property
    def value(self) -> float:
        """Float value of the metric."""
        return self._value

    @property
    def timestamp(self) -> int:
        """Metric timestamp as an integer (milliseconds since the Unix epoch)."""
        return self._timestamp

    @property
    def step(self) -> int:
        """Integer metric step (x-coordinate)."""
        return self._step

    @property
    def model_id(self) -> str | None:
        """ID of the Model associated with the metric."""
        return self._model_id

    @property
    def dataset_name(self) -> str | None:
        """String. Name of the dataset associated with the metric."""
        return self._dataset_name

    @property
    def dataset_digest(self) -> str | None:
        """String. Digest of the dataset associated with the metric."""
        return self._dataset_digest

    @property
    def run_id(self) -> str | None:
        """String. Run ID associated with the metric."""
        return self._run_id

    def to_proto(self) -> ProtoMetric:
        metric = ProtoMetric()
        metric.key = self.key
        metric.value = self.value
        metric.timestamp = self.timestamp
        metric.step = self.step
        if self.model_id:
            metric.model_id = self.model_id
        if self.dataset_name:
            metric.dataset_name = self.dataset_name
        if self.dataset_digest:
            metric.dataset_digest = self.dataset_digest
        if self.run_id:
            metric.run_id = self.run_id
        return metric

    @classmethod
    def from_proto(cls, proto: ProtoMetric) -> Metric:
        return cls(
            proto.key,
            proto.value,
            proto.timestamp,
            proto.step,
            model_id=proto.model_id or None,
            dataset_name=proto.dataset_name or None,
            dataset_digest=proto.dataset_digest or None,
            run_id=proto.run_id or None,
        )

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self.__dict__ == __o.__dict__

        return False

    def __hash__(self) -> int:
        return hash((
            self._key,
            self._value,
            self._timestamp,
            self._step,
            self._model_id,
            self._dataset_name,
            self._dataset_digest,
            self._run_id,
        ))

    def to_dictionary(self) -> dict[str, Any]:
        """
        Convert the Metric object to a dictionary.

        Returns:
            dict: The Metric object represented as a dictionary.
        """
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "model_id": self.model_id,
            "dataset_name": self.dataset_name,
            "dataset_digest": self.dataset_digest,
            "run_id": self._run_id,
        }

    @classmethod
    def from_dictionary(cls, metric_dict: dict[str, Any]) -> Metric:
        """
        Create a Metric object from a dictionary.

        Args:
            metric_dict (dict): Dictionary containing metric information.

        Returns:
            Metric: The Metric object created from the dictionary.
        """
        required_keys = ["key", "value", "timestamp", "step"]
        if missing_keys := [key for key in required_keys if key not in metric_dict]:
            raise MlflowException(
                f"Missing required keys {missing_keys} in metric dictionary",
                INVALID_PARAMETER_VALUE,
            )

        return cls(**metric_dict)


class MetricWithRunId(Metric):
    def __init__(self, metric: Metric, run_id: str) -> None:
        super().__init__(
            key=metric.key,
            value=metric.value,
            timestamp=metric.timestamp,
            step=metric.step,
        )
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        # `_run_id` is always set to a string by this subclass's `__init__`, while the
        # attribute is inherited with an optional type from `Metric`, hence the cast.
        return cast(str, self._run_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "run_id": self.run_id,
        }

    def to_proto(self) -> ProtoMetricWithRunId:
        metric = ProtoMetricWithRunId()
        metric.key = self.key
        metric.value = self.value
        metric.timestamp = self.timestamp
        metric.step = self.step
        metric.run_id = self.run_id
        return metric
