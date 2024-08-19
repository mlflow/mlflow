from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    ModelVersionInfo,
    ModelVersionStatus,
    RegisteredModelInfo,
)

_STRING_TO_STATUS = {k: ModelVersionStatus.Value(k) for k in ModelVersionStatus.keys()}
_STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}

def registered_model_from_uc_oss_proto(uc_oss_proto: RegisteredModelInfo) -> RegisteredModel:
    return RegisteredModel(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.name}",
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
    )

def model_version_from_uc_oss_proto(uc_oss_proto: ModelVersionInfo) -> ModelVersion:
    return ModelVersion(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.model_name}",
        version=uc_oss_proto.version,
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
        source=uc_oss_proto.source,
        run_id=uc_oss_proto.run_id,
        status=uc_oss_model_version_status_to_string(uc_oss_proto.status),
    )

def uc_oss_model_version_status_to_string(status):
    return _STATUS_TO_STRING[status]
