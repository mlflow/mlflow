from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

STAGE_NONE = "None"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"

STAGE_DELETED_INTERNAL = "Deleted_Internal"

ALL_STAGES = [STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED]
DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS = [STAGE_STAGING, STAGE_PRODUCTION]
_CANONICAL_MAPPING = {stage.lower(): stage for stage in ALL_STAGES}


def get_canonical_stage(stage):
    key = stage.lower()
    if key not in _CANONICAL_MAPPING:
        raise MlflowException(
            "Invalid Model Version stage {}.".format(stage), INVALID_PARAMETER_VALUE
        )
    return _CANONICAL_MAPPING[key]
