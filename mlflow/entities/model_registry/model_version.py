from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
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
        name,
        version,
        creation_timestamp,
        last_updated_timestamp=None,
        description=None,
        user_id=None,
        current_stage=None,
        source=None,
        run_id=None,
        status=ModelVersionStatus.to_string(ModelVersionStatus.READY),
        status_message=None,
        tags=None,
        run_link=None,
        aliases=None,
    ):
        super().__init__()
        self._name = name
        self._version = version
        self._creation_time = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp
        self._description = description
        self._user_id = user_id
        self._current_stage = current_stage
        self._source = source
        self._run_id = run_id
        self._run_link = run_link
        self._status = status
        self._status_message = status_message
        self._tags = {tag.key: tag.value for tag in (tags or [])}
        self._aliases = aliases or []

    @property
    def name(self):
        """String. Unique name within Model Registry."""
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def version(self):
        """version"""
        return self._version

    @property
    def creation_timestamp(self):
        """Integer. Model version creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_time

    @property
    def last_updated_timestamp(self):
        """Integer. Timestamp of last update for this model version (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp):
        self._last_updated_timestamp = updated_timestamp

    @property
    def description(self):
        """String. Description"""
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @property
    def user_id(self):
        """String. User ID that created this model version."""
        return self._user_id

    @property
    def current_stage(self):
        """String. Current stage of this model version."""
        return self._current_stage

    @current_stage.setter
    def current_stage(self, stage):
        self._current_stage = stage

    @property
    def source(self):
        """String. Source path for the model."""
        return self._source

    @property
    def run_id(self):
        """String. MLflow run ID that generated this model."""
        return self._run_id

    @property
    def run_link(self):
        """String. MLflow run link referring to the exact run that generated this model version."""
        return self._run_link

    @property
    def status(self):
        """String. Current Model Registry status for this model."""
        return self._status

    @property
    def status_message(self):
        """String. Descriptive message for error status conditions."""
        return self._status_message

    @property
    def tags(self):
        """Dictionary of tag key (string) -> tag value for the current model version."""
        return self._tags

    @property
    def aliases(self):
        """List of aliases (string) for the current model version."""
        return self._aliases

    @aliases.setter
    def aliases(self, aliases):
        self._aliases = aliases

    @classmethod
    def _properties(cls):
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper())

    def _add_tag(self, tag):
        self._tags[tag.key] = tag.value

    # proto mappers
    @classmethod
    def from_proto(cls, proto):
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
        )
        for tag in proto.tags:
            model_version._add_tag(ModelVersionTag.from_proto(tag))
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
        return model_version
