from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersion as ProtoModelVersion,
    ModelVersionStatus as ProtoModelVersionStatus,
    RegisteredModel as ProtoRegisteredModel
)


_STRING_TO_STATUS = {k: ProtoModelVersionStatus.Value(k) for k in ProtoModelVersionStatus.keys()}
_STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}


def uc_model_version_status_to_string(status):
    return _STATUS_TO_STRING[status]


def model_version_from_uc_proto(uc_proto: ProtoModelVersion) -> ModelVersion:
    return ModelVersion(
        name=uc_proto.name,
        version=uc_proto.version,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        user_id=uc_proto.user_id,
        source=uc_proto.source,
        run_id=uc_proto.run_id,
        status=uc_model_version_status_to_string(uc_proto.status),
        status_message=uc_proto.status_message,
    )


def registered_model_from_uc_proto(uc_proto: ProtoRegisteredModel) -> RegisteredModel:
    return RegisteredModel(
        name=uc_proto.name,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
    )
