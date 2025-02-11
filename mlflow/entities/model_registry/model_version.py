from typing import Optional

from mlflow.entities.logged_model_parameter import LoggedModelParameter as ModelParam
from mlflow.entities.metric import Metric
from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_deployment_job_state import (
    ModelVersionDeploymentJobState,
)
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.protos.model_registry_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.model_registry_pb2 import ModelVersionTag as ProtoModelVersionTag


class ModelVersion(_ModelRegistryEntity):
    """
    MLflow entity for Model Version.
    """

    def __init__(
        self,
        name: str,
        version: str,
        creation_timestamp: int,
        last_updated_timestamp: Optional[int] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        current_stage: Optional[str] = None,
        source: Optional[str] = None,
        run_id: Optional[str] = None,
        status: str = ModelVersionStatus.to_string(ModelVersionStatus.READY),
        status_message: Optional[str] = None,
        tags: Optional[list[ModelVersionTag]] = None,
        run_link: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        # TODO: Make model_id a required field
        # (currently optional to minimize breakages during prototype development)
        model_id: Optional[str] = None,
        params: Optional[list[ModelParam]] = None,
        metrics: Optional[list[Metric]] = None,
        deployment_job_state: Optional[ModelVersionDeploymentJobState] = None,
    ):
        super().__init__()
        self._name: str = name
        self._version: str = version
        self._creation_time: int = creation_timestamp
        self._last_updated_timestamp: Optional[int] = last_updated_timestamp
        self._description: Optional[str] = description
        self._user_id: Optional[str] = user_id
        self._current_stage: Optional[str] = current_stage
        self._source: Optional[str] = source
        self._run_id: Optional[str] = run_id
        self._run_link: Optional[str] = run_link
        self._status: str = status
        self._status_message: Optional[str] = status_message
        self._tags: dict[str, str] = {tag.key: tag.value for tag in (tags or [])}
        self._aliases: list[str] = aliases or []
        self._model_id: Optional[str] = model_id
        self._params: Optional[list[ModelParam]] = params
        self._metrics: Optional[list[Metric]] = metrics
        self._deployment_job_state: Optional[ModelVersionDeploymentJobState] = deployment_job_state

    @property
    def name(self) -> str:
        """String. Unique name within Model Registry."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def version(self) -> str:
        """Version"""
        return self._version

    @property
    def creation_timestamp(self) -> int:
        """Integer. Model version creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_time

    @property
    def last_updated_timestamp(self) -> Optional[int]:
        """Integer. Timestamp of last update for this model version (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp: int):
        self._last_updated_timestamp = updated_timestamp

    @property
    def description(self) -> Optional[str]:
        """String. Description"""
        return self._description

    @description.setter
    def description(self, description: str):
        self._description = description

    @property
    def user_id(self) -> Optional[str]:
        """String. User ID that created this model version."""
        return self._user_id

    @property
    def current_stage(self) -> Optional[str]:
        """String. Current stage of this model version."""
        return self._current_stage

    @current_stage.setter
    def current_stage(self, stage: str):
        self._current_stage = stage

    @property
    def source(self) -> Optional[str]:
        """String. Source path for the model."""
        return self._source

    @property
    def run_id(self) -> Optional[str]:
        """String. MLflow run ID that generated this model."""
        return self._run_id

    @property
    def run_link(self) -> Optional[str]:
        """String. MLflow run link referring to the exact run that generated this model version."""
        return self._run_link

    @property
    def status(self) -> str:
        """String. Current Model Registry status for this model."""
        return self._status

    @property
    def status_message(self) -> Optional[str]:
        """String. Descriptive message for error status conditions."""
        return self._status_message

    @property
    def tags(self) -> dict[str, str]:
        """Dictionary of tag key (string) -> tag value for the current model version."""
        return self._tags

    @property
    def aliases(self) -> list[str]:
        """List of aliases (string) for the current model version."""
        return self._aliases

    @aliases.setter
    def aliases(self, aliases: list[str]):
        self._aliases = aliases

    @property
    def model_id(self) -> Optional[str]:
        """String. ID of the model associated with this version."""
        return self._model_id

    @property
    def params(self) -> Optional[list[ModelParam]]:
        """List of parameters associated with this model version."""
        return self._params

    @property
    def metrics(self) -> Optional[list[Metric]]:
        """List of metrics associated with this model version."""
        return self._metrics

    @property
    def deployment_job_state(self) -> Optional[ModelVersionDeploymentJobState]:
        """Deployment job state for the current model version."""
        return self._deployment_job_state

    @classmethod
    def _properties(cls) -> list[str]:
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper())

    def _add_tag(self, tag: ModelVersionTag):
        self._tags[tag.key] = tag.value

    # proto mappers
    @classmethod
    def from_proto(cls, proto) -> "ModelVersion":
        # input: mlflow.protos.model_registry_pb2.ModelVersion
        # returns: ModelVersion entity
        model_version = cls(
            proto.name,
            proto.version,
            proto.creation_timestamp,
            proto.last_updated_timestamp,
            proto.description if proto.HasField("description") else None,
            proto.user_id,
            proto.current_stage,
            proto.source,
            proto.run_id if proto.HasField("run_id") else None,
            ModelVersionStatus.to_string(proto.status),
            proto.status_message if proto.HasField("status_message") else None,
            run_link=proto.run_link,
            aliases=proto.aliases,
            deployment_job_state=ModelVersionDeploymentJobState.from_proto(
                proto.deployment_job_state
            ),
        )
        for tag in proto.tags:
            model_version._add_tag(ModelVersionTag.from_proto(tag))
        # TODO: Include params, metrics, and model ID in proto
        return model_version

    def to_proto(self):
        # input: ModelVersion entity
        # returns mlflow.protos.model_registry_pb2.ModelVersion
        model_version = ProtoModelVersion()
        model_version.name = self.name
        model_version.version = str(self.version)
        model_version.creation_timestamp = self.creation_timestamp
        if self.last_updated_timestamp is not None:
            model_version.last_updated_timestamp = self.last_updated_timestamp
        if self.description is not None:
            model_version.description = self.description
        if self.user_id is not None:
            model_version.user_id = self.user_id
        if self.current_stage is not None:
            model_version.current_stage = self.current_stage
        if self.source is not None:
            model_version.source = str(self.source)
        if self.run_id is not None:
            model_version.run_id = str(self.run_id)
        if self.run_link is not None:
            model_version.run_link = str(self.run_link)
        if self.status is not None:
            model_version.status = ModelVersionStatus.from_string(self.status)
        if self.status_message:
            model_version.status_message = self.status_message
        model_version.tags.extend(
            [ProtoModelVersionTag(key=key, value=value) for key, value in self._tags.items()]
        )
        model_version.aliases.extend(self.aliases)
        if self.deployment_job_state is not None:
            ModelVersionDeploymentJobState.to_proto(self.deployment_job_state)
        # TODO: Include params, metrics, and model ID in proto
        return model_version
