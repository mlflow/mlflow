from typing import List
from mlflow.exceptions import MlflowException
from mlflow.utils.config import read_configs
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.model_registry_pb2 import ModelStage as ProtoModelStage
from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity


def get_configured_model_stage_names() -> List:
    model_stages = read_configs().get("model_stages")
    model_stage_names = [stage.get("name") for stage in model_stages]
    return model_stage_names


STAGE_NONE = "None"
STAGE_ARCHIVED = "Archived"

STAGE_DELETED_INTERNAL = "Deleted_Internal"

DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS = get_configured_model_stage_names()
ALL_STAGES = [STAGE_NONE] + DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS + [STAGE_ARCHIVED]

_CANONICAL_MAPPING = {stage.lower(): stage for stage in ALL_STAGES}


def get_canonical_stage(stage):
    key = stage.lower()
    if key not in _CANONICAL_MAPPING:
        raise MlflowException(
            "Invalid Model Version stage {}.".format(stage), INVALID_PARAMETER_VALUE
        )
    return _CANONICAL_MAPPING[key]


class ModelStage(_ModelRegistryEntity):
    """ModelStage object."""

    def __init__(self, name, color):
        self._name = name
        self._color = color

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def name(self):
        """String name of the tag."""
        return self._name

    @property
    def color(self):
        """String color of the tag."""
        return self._color

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.name, proto.color)

    def to_proto(self):
        tag = ProtoModelStage()
        tag.name = self.name
        tag.color = self.color
        return tag