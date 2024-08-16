from mlflow.entities.model_registry import RegisteredModel, RegisteredModelAlias, RegisteredModelTag
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    RegisteredModelAliasInfo,
    RegisteredModelInfo,
    TagKeyValue,
)
from mlflow.utils._unity_catalog_oss_utils import registered_model_from_uc_oss_proto


def test_registered_model_from_uc_oss_proto():
    expected_registered_model = RegisteredModel(
        name="catalog.schema.name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[
            RegisteredModelAlias(alias="alias1", version="1"),
            RegisteredModelAlias(alias="alias2", version="2"),
        ],
        tags=[
            RegisteredModelTag(key="key1", value="value"),
            RegisteredModelTag(key="key2", value=""),
        ],
    )

    uc_oss_proto = RegisteredModelInfo(
        name="name",
        catalog_name="catalog",
        schema_name="schema",
        created_at=1,
        updated_at=2,
        comment="description",
        aliases=[
            RegisteredModelAliasInfo(alias_name="alias1", version_num=1),
            RegisteredModelAliasInfo(alias_name="alias2", version_num=2),
        ],
        tags=[
            TagKeyValue(key="key1", value="value"),
            TagKeyValue(key="key2", value=""),
        ],
    )

    actual_registered_model = registered_model_from_uc_oss_proto(uc_oss_proto)
    assert actual_registered_model == expected_registered_model
