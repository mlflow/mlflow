from mlflow.entities.model_registry import RegisteredModel
from mlflow.protos.unity_catalog_oss_messages_pb2 import RegisteredModelInfo


def registered_model_from_uc_oss_proto(uc_oss_proto: RegisteredModelInfo) -> RegisteredModel:
    return RegisteredModel(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.name}",
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
    )
