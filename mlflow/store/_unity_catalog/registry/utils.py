from mlflow.entities.model_registry import RegisteredModel
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModel as ProtoRegisteredModel,
)


def registered_model_from_uc_proto(uc_proto: ProtoRegisteredModel) -> RegisteredModel:
    return RegisteredModel(
        name=uc_proto.name,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
    )
