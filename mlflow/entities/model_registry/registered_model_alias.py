from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.protos.model_registry_pb2 import RegisteredModelAlias as ProtoRegisteredModelAlias


class RegisteredModelAlias(_ModelRegistryEntity):
    """Alias object associated with a registered model."""

    def __init__(self, name, alias, version):
        self._name = name
        self._alias = alias
        self._version = version

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def name(self):
        """String name of the registered model associated with the alias."""
        return self._name

    @property
    def alias(self):
        """String name of the alias."""
        return self._alias

    @property
    def version(self):
        """String model version number that the alias points to."""
        return self._version

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.name, proto.alias, proto.version)

    def to_proto(self):
        alias = ProtoRegisteredModelAlias()
        alias.name = self.name
        alias.alias = self.alias
        alias.version = self.version
        return alias

