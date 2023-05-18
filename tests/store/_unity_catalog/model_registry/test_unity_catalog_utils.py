from mlflow.entities.model_registry import ModelVersion, RegisteredModel, RegisteredModelAlias
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersion as ProtoModelVersion,
    ModelVersionStatus as ProtoModelVersionStatus,
    RegisteredModel as ProtoRegisteredModel,
    RegisteredModelAlias as ProtoRegisteredModelAlias,
)
from mlflow.store._unity_catalog.registry.utils import (
    model_version_from_uc_proto,
    registered_model_from_uc_proto,
)


def test_model_version_from_uc_proto():
    expected_model_version = ModelVersion(
        name="name",
        version="1",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="user_id",
        source="source",
        run_id="run_id",
        status="READY",
        status_message="status_message",
        aliases=["alias1", "alias2"],
    )
    uc_proto = ProtoModelVersion(
        name="name",
        version="1",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="user_id",
        source="source",
        run_id="run_id",
        status=ProtoModelVersionStatus.Value("READY"),
        status_message="status_message",
        aliases=[
            ProtoRegisteredModelAlias(alias="alias1", version="1"),
            ProtoRegisteredModelAlias(alias="alias2", version="2"),
        ],
    )
    actual_model_version = model_version_from_uc_proto(uc_proto)
    assert actual_model_version == expected_model_version


def test_registered_model_from_uc_proto():
    expected_registered_model = RegisteredModel(
        name="name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[
            RegisteredModelAlias(alias="alias1", version="1"),
            RegisteredModelAlias(alias="alias2", version="2"),
        ],
    )
    uc_proto = ProtoRegisteredModel(
        name="name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[
            ProtoRegisteredModelAlias(alias="alias1", version="1"),
            ProtoRegisteredModelAlias(alias="alias2", version="2"),
        ],
    )
    actual_registered_model = registered_model_from_uc_proto(uc_proto)
    assert actual_registered_model == expected_registered_model
