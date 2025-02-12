from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.registered_model_alias import RegisteredModelAlias
from mlflow.entities.model_registry.registered_model_tag import RegisteredModelTag
from mlflow.protos.model_registry_pb2 import RegisteredModel as ProtoRegisteredModel
from mlflow.protos.model_registry_pb2 import RegisteredModelAlias as ProtoRegisteredModelAlias
from mlflow.protos.model_registry_pb2 import RegisteredModelTag as ProtoRegisteredModelTag


class RegisteredModel(_ModelRegistryEntity):
    """
    MLflow entity for Registered Model.
    """

    def __init__(
        self,
        name,
        creation_timestamp=None,
        last_updated_timestamp=None,
        description=None,
        latest_versions=None,
        tags=None,
        aliases=None,
    ):
        # Constructor is called only from within the system by various backend stores.
        super().__init__()
        self._name = name
        self._creation_time = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp
        self._description = description
        self._latest_version = latest_versions
        self._tags = {tag.key: tag.value for tag in (tags or [])}
        self._aliases = {alias.alias: alias.version for alias in (aliases or [])}

    @property
    def name(self):
        """String. Registered model name."""
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

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
    def latest_versions(self):
        """List of the latest :py:class:`mlflow.entities.model_registry.ModelVersion` instances
        for each stage.
        """
        return self._latest_version

    @latest_versions.setter
    def latest_versions(self, latest_versions):
        self._latest_version = latest_versions

    @property
    def tags(self):
        """Dictionary of tag key (string) -> tag value for the current registered model."""
        return self._tags

    @property
    def aliases(self):
        """Dictionary of aliases (string) -> version for the current registered model."""
        return self._aliases

    @classmethod
    def _properties(cls):
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper())

    def _add_tag(self, tag):
        self._tags[tag.key] = tag.value

    def _add_alias(self, alias):
        self._aliases[alias.alias] = alias.version

    # proto mappers
    @classmethod
    def from_proto(cls, proto):
        # input: mlflow.protos.model_registry_pb2.RegisteredModel
        # returns RegisteredModel entity
        registered_model = cls(
            proto.name,
            proto.creation_timestamp,
            proto.last_updated_timestamp,
            proto.description,
            [ModelVersion.from_proto(mvd) for mvd in proto.latest_versions],
        )
        for tag in proto.tags:
            registered_model._add_tag(RegisteredModelTag.from_proto(tag))
        for alias in proto.aliases:
            registered_model._add_alias(RegisteredModelAlias.from_proto(alias))
        return registered_model

    def to_proto(self):
        # returns mlflow.protos.model_registry_pb2.RegisteredModel
        rmd = ProtoRegisteredModel()
        rmd.name = self.name
        if self.creation_timestamp is not None:
            rmd.creation_timestamp = self.creation_timestamp
        if self.last_updated_timestamp:
            rmd.last_updated_timestamp = self.last_updated_timestamp
        if self.description:
            rmd.description = self.description
        if self.latest_versions is not None:
            rmd.latest_versions.extend(
                [model_version.to_proto() for model_version in self.latest_versions]
            )
        rmd.tags.extend(
            [ProtoRegisteredModelTag(key=key, value=value) for key, value in self._tags.items()]
        )
        rmd.aliases.extend(
            [
                ProtoRegisteredModelAlias(alias=alias, version=str(version))
                for alias, version in self._aliases.items()
            ]
        )
        return rmd
