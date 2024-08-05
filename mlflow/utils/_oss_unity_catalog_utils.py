from mlflow.entities.model_registry import RegisteredModel, RegisteredModelAlias, RegisteredModelTag
from mlflow.protos.databricks_managed_catalog_messages_pb2 import RegisteredModelInfo


def registered_model_from_uc_oss_proto(uc_oss_proto: RegisteredModelInfo) -> RegisteredModel:
    return RegisteredModel(
        name=f"{uc_oss_proto.catalog_name}.{uc_oss_proto.schema_name}.{uc_oss_proto.name}",
        creation_timestamp=uc_oss_proto.created_at,
        last_updated_timestamp=uc_oss_proto.updated_at,
        description=uc_oss_proto.comment,
        aliases=[
            RegisteredModelAlias(alias=alias.alias_name, version=str(alias.version_num))
            for alias in (uc_oss_proto.aliases or [])
        ],
        tags=[
            RegisteredModelTag(key=tag.key, value=tag.value) for tag in (uc_oss_proto.tags or [])
        ],
    )
