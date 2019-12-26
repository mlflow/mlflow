from mlflow.entities.model_registry.model_version_detailed import ModelVersionDetailed
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.protos.model_registry_pb2 import RegisteredModelDetailed as ProtoRegisteredModelDetailed


class RegisteredModelDetailed(RegisteredModel):
    """
    .. note::
        Experimental: This entity may change or be removed in a future release without warning.

    MLflow entity for Registered Model Detailed.
    Provides additional metadata data for registered model in addition to information in
    :py:class:`mlflow.entities.model_registry.RegisteredModel`.
    """

    def __init__(self, name, creation_timestamp, last_updated_timestamp=None, description=None,
                 latest_versions=None):
        # Constructor is called only from within the system by various backend stores.
        super(RegisteredModelDetailed, self).__init__(name)
        self._creation_time = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp
        self._description = description
        self._latest_version = latest_versions

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
    def latest_versions(self):
        """List of the latest :py:class:`mlflow.entities.model_registry.ModelVersion` instances
        for each stage"""
        return self._latest_version

    @classmethod
    def _properties(cls):
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper() + RegisteredModel._properties())

    # proto mappers
    @classmethod
    def from_proto(cls, proto):
        # input: mlflow.protos.model_registry_pb2.RegisteredModelDetailed
        # returns RegisteredModelDetailed entity
        return cls(proto.registered_model.name,
                   proto.creation_timestamp,
                   proto.last_updated_timestamp,
                   proto.description,
                   [ModelVersionDetailed.from_proto(mvd) for mvd in proto.latest_versions])

    def to_proto(self):
        # returns mlflow.protos.model_registry_pb2.RegisteredModelDetailed
        rmd = ProtoRegisteredModelDetailed()
        rmd.registered_model.MergeFrom(super(RegisteredModelDetailed, self).to_proto())
        rmd.creation_timestamp = self.creation_timestamp
        if self.last_updated_timestamp:
            rmd.last_updated_timestamp = self.last_updated_timestamp
        if self.description:
            rmd.description = self.description
        rmd.latest_versions.extend([model_version_detailed.to_proto()
                                    for model_version_detailed in self.latest_versions])
        return rmd
