from mlflow.entities.model_registry import RegisteredModel
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.protos.model_registry_pb2 import ModelVersionDetailed as ProtoModelVersionDetailed


class ModelVersionDetailed(ModelVersion):
    """
    .. note::
        Experimental: This entity may change or be removed in a future release without warning.

    MLflow entity for Model Version Detailed.
    Provides additional metadata data for model version in addition to information in
    :py:class:`mlflow.entities.model_registry.ModelVersion`.
    """

    def __init__(self, registered_model, version, creation_timestamp, last_updated_timestamp=None,
                 description=None, user_id=None, current_stage=None, source=None, run_id=None,
                 status=None, status_message=None):
        # Constructor is called only from within the system by various backend stores.
        super(ModelVersionDetailed, self).__init__(registered_model=registered_model,
                                                   version=version)
        self._creation_time = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp
        self._description = description
        self._user_id = user_id
        self._current_stage = current_stage
        self._source = source
        self._run_id = run_id
        self._status = status
        self._status_message = status_message

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

    @classmethod
    def _properties(cls):
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper() + ModelVersion._properties())

    # proto mappers
    @classmethod
    def from_proto(cls, proto):
        # input: mlflow.protos.model_registry_pb2.ModelVersionDetailed
        # returns: ModelVersionDetailed entity
        return cls(RegisteredModel.from_proto(proto.model_version.registered_model),
                   proto.model_version.version,
                   proto.creation_timestamp,
                   proto.last_updated_timestamp,
                   proto.description,
                   proto.user_id,
                   proto.current_stage,
                   proto.source,
                   proto.run_id,
                   ModelVersionStatus.to_string(proto.status),
                   proto.status_message)

    def to_proto(self):
        # input: ModelVersionDetailed entity
        # returns mlflow.protos.model_registry_pb2.ModelVersionDetailed
        model_version_detailed = ProtoModelVersionDetailed()
        model_version_detailed.model_version.MergeFrom(super(ModelVersionDetailed, self).to_proto())
        model_version_detailed.creation_timestamp = self.creation_timestamp
        if self.last_updated_timestamp:
            model_version_detailed.last_updated_timestamp = self.last_updated_timestamp
        if self.description:
            model_version_detailed.description = self.description
        if self.user_id:
            model_version_detailed.user_id = self.user_id
        model_version_detailed.current_stage = self.current_stage
        model_version_detailed.source = self.source
        model_version_detailed.run_id = self.run_id
        model_version_detailed.status = ModelVersionStatus.from_string(self.status)
        if self.status_message:
            model_version_detailed.status_message = self.status_message
        return model_version_detailed
