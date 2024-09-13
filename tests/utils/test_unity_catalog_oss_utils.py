from mlflow.entities.model_registry import RegisteredModel
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    RegisteredModelInfo,
)
from mlflow.utils._unity_catalog_oss_utils import get_registered_model_from_uc_oss_proto


def test_registered_model_from_uc_oss_proto():
    expected_registered_model = RegisteredModel(
        name="catalog.schema.name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
    )

    uc_oss_proto = RegisteredModelInfo(
        name="name",
        catalog_name="catalog",
        schema_name="schema",
        created_at=1,
        updated_at=2,
        comment="description",
    )

    actual_registered_model = get_registered_model_from_uc_oss_proto(uc_oss_proto)
    assert actual_registered_model == expected_registered_model
