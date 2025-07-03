from typing import Any, Optional, Union

import mlflow.protos.service_pb2 as pb2
from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.metric import Metric


class LoggedModel(_MlflowObject):
    """
    MLflow entity representing a Model logged to an MLflow Experiment.
    """

    def __init__(
        self,
        experiment_id: str,
        model_id: str,
        name: str,
        artifact_location: str,
        creation_timestamp: int,
        last_updated_timestamp: int,
        model_type: Optional[str] = None,
        source_run_id: Optional[str] = None,
        status: Union[LoggedModelStatus, int] = LoggedModelStatus.READY,
        status_message: Optional[str] = None,
        tags: Optional[Union[list[LoggedModelTag], dict[str, str]]] = None,
        params: Optional[Union[list[LoggedModelParameter], dict[str, str]]] = None,
        metrics: Optional[list[Metric]] = None,
    ):
        super().__init__()
        self._experiment_id: str = experiment_id
        self._model_id: str = model_id
        self._name: str = name
        self._artifact_location: str = artifact_location
        self._creation_time: int = creation_timestamp
        self._last_updated_timestamp: int = last_updated_timestamp
        self._model_type: Optional[str] = model_type
        self._source_run_id: Optional[str] = source_run_id
        self._status: LoggedModelStatus = (
            status if isinstance(status, LoggedModelStatus) else LoggedModelStatus.from_int(status)
        )
        self._status_message: Optional[str] = status_message
        self._tags: dict[str, str] = (
            {tag.key: tag.value for tag in (tags or [])} if isinstance(tags, list) else (tags or {})
        )
        self._params: dict[str, str] = (
            {param.key: param.value for param in (params or [])}
            if isinstance(params, list)
            else (params or {})
        )
        self._metrics: Optional[list[Metric]] = metrics
        self._model_uri = f"models:/{self.model_id}"

    def __repr__(self) -> str:
        return "LoggedModel({})".format(
            ", ".join(
                f"{k}={v!r}"
                for k, v in sorted(self, key=lambda x: x[0])
                if (
                    k
                    not in [
                        # These fields can be large and take up space on the notebook or terminal
                        "tags",
                        "params",
                        "metrics",
                    ]
                )
            )
        )

    @property
    def experiment_id(self) -> str:
        """String. Experiment ID associated with this Model."""
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, new_experiment_id: str):
        self._experiment_id = new_experiment_id

    @property
    def model_id(self) -> str:
        """String. Unique ID for this Model."""
        return self._model_id

    @model_id.setter
    def model_id(self, new_model_id: str):
        self._model_id = new_model_id

    @property
    def name(self) -> str:
        """String. Name for this Model."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def artifact_location(self) -> str:
        """String. Location of the model artifacts."""
        return self._artifact_location

    @artifact_location.setter
    def artifact_location(self, new_artifact_location: str):
        self._artifact_location = new_artifact_location

    @property
    def creation_timestamp(self) -> int:
        """Integer. Model creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_time

    @property
    def last_updated_timestamp(self) -> int:
        """Integer. Timestamp of last update for this Model (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp: int):
        self._last_updated_timestamp = updated_timestamp

    @property
    def model_type(self) -> Optional[str]:
        """String. Type of the model."""
        return self._model_type

    @model_type.setter
    def model_type(self, new_model_type: Optional[str]):
        self._model_type = new_model_type

    @property
    def source_run_id(self) -> Optional[str]:
        """String. MLflow run ID that generated this model."""
        return self._source_run_id

    @property
    def status(self) -> LoggedModelStatus:
        """String. Current status of this Model."""
        return self._status

    @status.setter
    def status(self, updated_status: str):
        self._status = updated_status

    @property
    def status_message(self) -> Optional[str]:
        """String. Descriptive message for error status conditions."""
        return self._status_message

    @property
    def tags(self) -> dict[str, str]:
        """Dictionary of tag key (string) -> tag value for this Model."""
        return self._tags

    @property
    def params(self) -> dict[str, str]:
        """Model parameters."""
        return self._params

    @property
    def metrics(self) -> Optional[list[Metric]]:
        """List of metrics associated with this Model."""
        return self._metrics

    @property
    def model_uri(self) -> str:
        """URI of the model."""
        return self._model_uri

    @metrics.setter
    def metrics(self, new_metrics: Optional[list[Metric]]):
        self._metrics = new_metrics

    @classmethod
    def _properties(cls) -> list[str]:
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper())

    def _add_tag(self, tag):
        self._tags[tag.key] = tag.value

    def to_dictionary(self) -> dict[str, Any]:
        model_dict = dict(self)
        model_dict["status"] = self.status.to_int()
        # Remove the model_uri field from the dictionary since it is a derived field
        del model_dict["model_uri"]
        return model_dict

    def to_proto(self):
        return pb2.LoggedModel(
            info=pb2.LoggedModelInfo(
                experiment_id=self.experiment_id,
                model_id=self.model_id,
                name=self.name,
                artifact_uri=self.artifact_location,
                creation_timestamp_ms=self.creation_timestamp,
                last_updated_timestamp_ms=self.last_updated_timestamp,
                model_type=self.model_type,
                source_run_id=self.source_run_id,
                status=self.status.to_proto(),
                tags=[pb2.LoggedModelTag(key=k, value=v) for k, v in self.tags.items()],
            ),
            data=pb2.LoggedModelData(
                params=[pb2.LoggedModelParameter(key=k, value=v) for (k, v) in self.params.items()],
                metrics=[m.to_proto() for m in self.metrics] if self.metrics else [],
            ),
        )

    @classmethod
    def from_proto(cls, proto):
        return cls(
            experiment_id=proto.info.experiment_id,
            model_id=proto.info.model_id,
            name=proto.info.name,
            artifact_location=proto.info.artifact_uri,
            creation_timestamp=proto.info.creation_timestamp_ms,
            last_updated_timestamp=proto.info.last_updated_timestamp_ms,
            model_type=proto.info.model_type,
            source_run_id=proto.info.source_run_id,
            status=LoggedModelStatus.from_proto(proto.info.status),
            status_message=proto.info.status_message,
            tags=[LoggedModelTag.from_proto(tag) for tag in proto.info.tags],
            params=[LoggedModelParameter.from_proto(param) for param in proto.data.params],
            metrics=[Metric.from_proto(metric) for metric in proto.data.metrics],
        )
