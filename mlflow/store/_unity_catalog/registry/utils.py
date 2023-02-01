from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersion,
    ModelVersionStatus,
    RegisteredModel,
)


def model_version_from_uc_proto(uc_proto):
    # input: mlflow.protos.databricks_uc_registry_messages_pb2.ModelVersion
    # returns: ModelVersion entity
    return ModelVersion(
        uc_proto.name,
        uc_proto.version,
        uc_proto.creation_timestamp,
        uc_proto.last_updated_timestamp,
        uc_proto.description,
        uc_proto.user_id,
        uc_proto.current_stage,
        uc_proto.source,
        uc_proto.run_id,
        ModelVersionStatus.to_string(uc_proto.status),
        uc_proto.status_message,
        run_link=uc_proto.run_link,
    )


def registered_model_from_uc_proto(uc_proto):
    # input: mlflow.protos.databricks_uc_registry_messages_pb2.RegisteredModel
    # returns: RegisteredModel entity
    return RegisteredModel(
        uc_proto.name,
        uc_proto.creation_timestamp,
        uc_proto.last_updated_timestamp,
        uc_proto.description,
        uc_proto.user_id,
    )


def registered_model_to_uc_proto(uc_proto):
    pass


def model_version_to_uc_proto(uc_proto):
    pass
