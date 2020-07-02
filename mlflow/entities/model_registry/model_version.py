from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.protos.model_registry_pb2 import ModelVersion as ProtoModelVersion, \
    ModelVersionTag as ProtoModelVersionTag


class ModelVersion(_ModelRegistryEntity):
    """
    .. note::
        Experimental: This entity may change or be removed in a future release without warning.

    MLflow entity for Model Version.
    """

    def __init__(self, name, version, creation_timestamp,
                 last_updated_timestamp=None, description=None, user_id=None, current_stage=None,
                 source=None, run_id=None, status=None, status_message=None, tags=None):
        super(ModelVersion, self).__init__()
        self._name = name
        self._version = version
        self._creation_time = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp
        self._description = description
        self._user_id = user_id
        self._current_stage = current_stage
        self._source = source
        self._run_id = run_id
        self._status = status
        self._status_message = status_message
        self._tags = {tag.key: tag.value for tag in (tags or [])}

    @property
    def name(self):
        """String. Unique name within Model Registry."""
        return self._name

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
        epoch)."""
        return self._last_updated_timestamp

    @property
    def description(self):
        """String. Description"""
        return self._description

    @property
    def user_id(self):
        """String. User ID that created this model version."""
        return self._user_id

    @property
    def current_stage(self):
        """String. Current stage of this model version."""
        return self._current_stage

    @property
    def source(self):
        """String. Source path for the model."""
        return self._source

    @property
    def run_id(self):
        """String. MLflow run ID that generated this model."""
        return self._run_id

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
        model_version = cls(proto.name,
                            proto.version,
                            proto.creation_timestamp,
                            proto.last_updated_timestamp,
                            proto.description,
                            proto.user_id,
                            proto.current_stage,
                            proto.source,
                            proto.run_id,
                            ModelVersionStatus.to_string(proto.status),
                            proto.status_message)
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
        if self.status is not None:
            model_version.status = ModelVersionStatus.from_string(self.status)
        if self.status_message:
            model_version.status_message = self.status_message
        model_version.tags.extend([ProtoModelVersionTag(key=key, value=value)
                                   for key, value in self._tags.items()])
        return model_version
