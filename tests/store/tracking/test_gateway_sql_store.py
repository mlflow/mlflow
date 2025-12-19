from pathlib import Path

import pytest

from mlflow.entities import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelDefinition,
    GatewaySecretInfo,
)
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlGatewayEndpoint,
    SqlGatewayEndpointBinding,
    SqlGatewayEndpointModelMapping,
    SqlGatewayEndpointTag,
    SqlGatewayModelDefinition,
    SqlGatewaySecret,
)
from mlflow.store.tracking.gateway.config_resolver import (
    get_endpoint_config,
    get_resource_endpoint_configs,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock

TEST_PASSPHRASE = "test-passphrase-for-gateway-tests"


@pytest.fixture(autouse=True)
def set_kek_passphrase(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", TEST_PASSPHRASE)


def _cleanup_database(store: SqlAlchemyStore):
    """Clean up gateway-specific tables after each test."""
    with store.ManagedSessionMaker() as session:
        # Delete all rows in gateway tables in dependency order
        for model in (
            SqlGatewayEndpointTag,
            SqlGatewayEndpointBinding,
            SqlGatewayEndpointModelMapping,
            SqlGatewayEndpoint,
            SqlGatewayModelDefinition,
            SqlGatewaySecret,
        ):
            session.query(model).delete()


@pytest.fixture
def store(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    if db_uri_env := MLFLOW_TRACKING_URI.get():
        s = SqlAlchemyStore(db_uri_env, artifact_uri.as_uri())
        yield s
        _cleanup_database(s)
    else:
        s = SqlAlchemyStore(db_uri, artifact_uri.as_uri())
        yield s


# =============================================================================
# Secret Operations
# =============================================================================


def test_create_gateway_secret(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="my-api-key",
        secret_value={"api_key": "sk-test-123456"},
        provider="openai",
        created_by="test-user",
    )

    assert isinstance(secret, GatewaySecretInfo)
    assert secret.secret_id.startswith("s-")
    assert secret.secret_name == "my-api-key"
    assert secret.provider == "openai"
    assert secret.created_by == "test-user"
    assert isinstance(secret.masked_values, dict)
    assert "api_key" in secret.masked_values
    assert "sk-test-123456" not in secret.masked_values["api_key"]


def test_create_gateway_secret_with_auth_config(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="bedrock-creds",
        secret_value={"api_key": "aws-secret-key"},
        provider="bedrock",
        auth_config={"region": "us-east-1", "project_id": "my-project"},
    )

    assert secret.secret_name == "bedrock-creds"
    assert secret.provider == "bedrock"


def test_create_gateway_secret_with_dict_value(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="multi-secret",
        secret_value={
            "aws_access_key_id": "AKIA1234567890",
            "aws_secret_access_key": "secret-key-here",
        },
        provider="bedrock",
        auth_config={"auth_mode": "access_keys", "aws_region_name": "us-west-2"},
    )

    assert secret.secret_name == "multi-secret"
    assert secret.provider == "bedrock"
    assert isinstance(secret.masked_values, dict)
    assert "aws_access_key_id" in secret.masked_values
    assert "aws_secret_access_key" in secret.masked_values
    assert "AKIA1234567890" not in secret.masked_values["aws_access_key_id"]
    assert "secret-key-here" not in secret.masked_values["aws_secret_access_key"]


def test_create_gateway_secret_duplicate_name_raises(store: SqlAlchemyStore):
    store.create_gateway_secret(secret_name="duplicate-name", secret_value={"api_key": "value1"})

    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_gateway_secret(
            secret_name="duplicate-name", secret_value={"api_key": "value2"}
        )
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_get_gateway_secret_info_by_id(store: SqlAlchemyStore):
    created = store.create_gateway_secret(
        secret_name="test-secret",
        secret_value={"api_key": "secret-value"},
        provider="anthropic",
    )

    retrieved = store.get_secret_info(secret_id=created.secret_id)

    assert retrieved.secret_id == created.secret_id
    assert retrieved.secret_name == "test-secret"
    assert retrieved.provider == "anthropic"


def test_get_gateway_secret_info_by_name(store: SqlAlchemyStore):
    created = store.create_gateway_secret(
        secret_name="named-secret",
        secret_value={"api_key": "secret-value"},
    )

    retrieved = store.get_secret_info(secret_name="named-secret")

    assert retrieved.secret_id == created.secret_id
    assert retrieved.secret_name == "named-secret"


def test_get_gateway_secret_info_requires_one_of_id_or_name(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="Exactly one of") as exc:
        store.get_secret_info()
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(MlflowException, match="Exactly one of") as exc:
        store.get_secret_info(secret_id="id", secret_name="name")
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_gateway_secret_info_not_found(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_secret_info(secret_id="nonexistent")
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_update_gateway_secret(store: SqlAlchemyStore):
    created = store.create_gateway_secret(
        secret_name="rotate-me",
        secret_value={"api_key": "old-value"},
    )
    original_updated_at = created.last_updated_at

    updated = store.update_gateway_secret(
        secret_id=created.secret_id,
        secret_value={"api_key": "new-value"},
        updated_by="rotator-user",
    )

    assert updated.secret_id == created.secret_id
    assert updated.last_updated_by == "rotator-user"
    assert updated.last_updated_at > original_updated_at


def test_update_gateway_secret_with_auth_config(store: SqlAlchemyStore):
    created = store.create_gateway_secret(
        secret_name="auth-update",
        secret_value={"api_key": "value"},
        auth_config={"region": "us-east-1"},
    )

    store.update_gateway_secret(
        secret_id=created.secret_id,
        secret_value={"api_key": "new-value"},
        auth_config={"region": "eu-west-1", "new_key": "new_value"},
    )


def test_update_gateway_secret_clear_auth_config(store: SqlAlchemyStore):
    created = store.create_gateway_secret(
        secret_name="clear-auth",
        secret_value={"api_key": "value"},
        auth_config={"region": "us-east-1"},
    )

    store.update_gateway_secret(
        secret_id=created.secret_id,
        secret_value={"api_key": "new-value"},
        auth_config={},
    )


def test_delete_gateway_secret(store: SqlAlchemyStore):
    created = store.create_gateway_secret(
        secret_name="to-delete", secret_value={"api_key": "value"}
    )

    store.delete_gateway_secret(created.secret_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_secret_info(secret_id=created.secret_id)


def test_list_gateway_secret_infos(store: SqlAlchemyStore):
    s1 = store.create_gateway_secret(
        secret_name="openai-1", secret_value={"api_key": "v1"}, provider="openai"
    )
    s2 = store.create_gateway_secret(
        secret_name="openai-2", secret_value={"api_key": "v2"}, provider="openai"
    )
    s3 = store.create_gateway_secret(
        secret_name="anthropic-1", secret_value={"api_key": "v3"}, provider="anthropic"
    )
    created_ids = {s1.secret_id, s2.secret_id, s3.secret_id}

    all_secrets = store.list_secret_infos()
    all_ids = {s.secret_id for s in all_secrets}
    assert created_ids.issubset(all_ids)

    openai_secrets = store.list_secret_infos(provider="openai")
    openai_ids = {s.secret_id for s in openai_secrets}
    assert {s1.secret_id, s2.secret_id}.issubset(openai_ids)
    assert s3.secret_id not in openai_ids
    assert all(s.provider == "openai" for s in openai_secrets)


def test_secret_id_and_name_are_immutable_at_database_level(store: SqlAlchemyStore):
    """
    Verify that secret_id and secret_name cannot be modified at the database level.

    These fields are used as AAD (Additional Authenticated Data) in AES-GCM encryption.
    If they are modified, decryption will fail. A database trigger enforces this immutability
    to prevent any code path from accidentally allowing mutation.
    """
    from sqlalchemy import text
    from sqlalchemy.exc import DatabaseError, IntegrityError, OperationalError

    secret = store.create_gateway_secret(
        secret_name="immutable-test",
        secret_value={"api_key": "test-value"},
        provider="openai",
    )

    def attempt_mutation(session):
        session.execute(
            text("UPDATE secrets SET secret_name = :new_name WHERE secret_id = :id"),
            {"new_name": "modified-name", "id": secret.secret_id},
        )
        session.flush()

    with store.ManagedSessionMaker() as session:
        with pytest.raises((DatabaseError, IntegrityError, OperationalError)):
            attempt_mutation(session)

    retrieved = store.get_secret_info(secret_id=secret.secret_id)
    assert retrieved.secret_name == "immutable-test"


# =============================================================================
# Model Definition Operations
# =============================================================================


def test_create_gateway_model_definition(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="test-key", secret_value={"api_key": "value"})

    model_def = store.create_gateway_model_definition(
        name="gpt-4-turbo",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4-turbo-preview",
        created_by="test-user",
    )

    assert isinstance(model_def, GatewayModelDefinition)
    assert model_def.model_definition_id.startswith("d-")
    assert model_def.name == "gpt-4-turbo"
    assert model_def.secret_id == secret.secret_id
    assert model_def.secret_name == "test-key"
    assert model_def.provider == "openai"
    assert model_def.model_name == "gpt-4-turbo-preview"


def test_create_gateway_model_definition_duplicate_name_raises(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="dup-key", secret_value={"api_key": "value"})

    store.create_gateway_model_definition(
        name="duplicate-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_gateway_model_definition(
            name="duplicate-model",
            secret_id=secret.secret_id,
            provider="openai",
            model_name="gpt-4",
        )
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_create_gateway_model_definition_nonexistent_secret_raises(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.create_gateway_model_definition(
            name="orphan-model",
            secret_id="nonexistent",
            provider="openai",
            model_name="gpt-4",
        )
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_get_gateway_model_definition_by_id(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="get-key", secret_value={"api_key": "value"})
    created = store.create_gateway_model_definition(
        name="model-by-id",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3-sonnet",
    )

    retrieved = store.get_gateway_model_definition(model_definition_id=created.model_definition_id)

    assert retrieved.model_definition_id == created.model_definition_id
    assert retrieved.name == "model-by-id"


def test_get_gateway_model_definition_by_name(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="name-key", secret_value={"api_key": "value"})
    created = store.create_gateway_model_definition(
        name="model-by-name",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    retrieved = store.get_gateway_model_definition(name="model-by-name")

    assert retrieved.model_definition_id == created.model_definition_id


def test_get_gateway_model_definition_requires_one_of_id_or_name(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="Exactly one of") as exc:
        store.get_gateway_model_definition()
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_list_gateway_model_definitions(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="list-key", secret_value={"api_key": "value"})

    store.create_gateway_model_definition(
        name="list-model-1", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    store.create_gateway_model_definition(
        name="list-model-2",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3",
    )

    all_defs = store.list_gateway_model_definitions()
    assert len(all_defs) >= 2

    openai_defs = store.list_gateway_model_definitions(provider="openai")
    assert all(d.provider == "openai" for d in openai_defs)


def test_update_gateway_model_definition(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="update-key", secret_value={"api_key": "value"}
    )
    created = store.create_gateway_model_definition(
        name="update-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    updated = store.update_gateway_model_definition(
        model_definition_id=created.model_definition_id,
        model_name="gpt-4-turbo",
        updated_by="updater",
    )

    assert updated.model_name == "gpt-4-turbo"
    assert updated.last_updated_by == "updater"


def test_delete_gateway_model_definition(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="delete-key", secret_value={"api_key": "value"}
    )
    created = store.create_gateway_model_definition(
        name="delete-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    store.delete_gateway_model_definition(created.model_definition_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_gateway_model_definition(model_definition_id=created.model_definition_id)


def test_delete_gateway_model_definition_in_use_raises(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="in-use-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="in-use-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    store.create_gateway_endpoint(
        name="uses-model", model_definition_ids=[model_def.model_definition_id]
    )

    with pytest.raises(MlflowException, match="currently in use") as exc:
        store.delete_gateway_model_definition(model_def.model_definition_id)
    assert exc.value.error_code == ErrorCode.Name(INVALID_STATE)


# =============================================================================
# Endpoint Operations
# =============================================================================


def test_create_gateway_endpoint(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="ep-key", secret_value={"api_key": "value"})
    model_def = store.create_gateway_model_definition(
        name="ep-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )

    endpoint = store.create_gateway_endpoint(
        name="my-endpoint",
        model_definition_ids=[model_def.model_definition_id],
        created_by="test-user",
    )

    assert isinstance(endpoint, GatewayEndpoint)
    assert endpoint.endpoint_id.startswith("e-")
    assert endpoint.name == "my-endpoint"
    assert len(endpoint.model_mappings) == 1
    assert endpoint.model_mappings[0].model_definition_id == model_def.model_definition_id


def test_create_gateway_endpoint_empty_models_raises(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="at least one") as exc:
        store.create_gateway_endpoint(name="empty-endpoint", model_definition_ids=[])
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_create_gateway_endpoint_nonexistent_model_raises(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.create_gateway_endpoint(name="orphan-endpoint", model_definition_ids=["nonexistent"])
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_get_gateway_endpoint_by_id(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="get-ep-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="get-ep-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    created = store.create_gateway_endpoint(
        name="get-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    retrieved = store.get_gateway_endpoint(endpoint_id=created.endpoint_id)

    assert retrieved.endpoint_id == created.endpoint_id
    assert retrieved.name == "get-endpoint"


def test_get_gateway_endpoint_by_name(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="name-ep-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="name-ep-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    created = store.create_gateway_endpoint(
        name="named-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    retrieved = store.get_gateway_endpoint(name="named-endpoint")

    assert retrieved.endpoint_id == created.endpoint_id


def test_get_gateway_endpoint_requires_one_of_id_or_name(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="Exactly one of") as exc:
        store.get_gateway_endpoint()
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_update_gateway_endpoint(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="upd-ep-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="upd-ep-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    created = store.create_gateway_endpoint(
        name="update-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    updated = store.update_gateway_endpoint(
        endpoint_id=created.endpoint_id,
        name="renamed-endpoint",
        updated_by="updater",
    )

    assert updated.name == "renamed-endpoint"
    assert updated.last_updated_by == "updater"


def test_delete_gateway_endpoint(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="del-ep-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="del-ep-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    created = store.create_gateway_endpoint(
        name="delete-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    store.delete_gateway_endpoint(created.endpoint_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_gateway_endpoint(endpoint_id=created.endpoint_id)


def test_list_gateway_endpoints(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="list-ep-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="list-ep-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    store.create_gateway_endpoint(
        name="list-endpoint-1", model_definition_ids=[model_def.model_definition_id]
    )
    store.create_gateway_endpoint(
        name="list-endpoint-2", model_definition_ids=[model_def.model_definition_id]
    )

    endpoints = store.list_gateway_endpoints()
    assert len(endpoints) >= 2


# =============================================================================
# Model Mapping Operations
# =============================================================================


def test_attach_model_to_gateway_endpoint(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="attach-key", secret_value={"api_key": "value"}
    )
    model_def1 = store.create_gateway_model_definition(
        name="attach-model-1", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    model_def2 = store.create_gateway_model_definition(
        name="attach-model-2",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3",
    )
    endpoint = store.create_gateway_endpoint(
        name="attach-endpoint", model_definition_ids=[model_def1.model_definition_id]
    )

    mapping = store.attach_model_to_endpoint(
        endpoint_id=endpoint.endpoint_id,
        model_definition_id=model_def2.model_definition_id,
        weight=2.0,
        created_by="attacher",
    )

    assert isinstance(mapping, GatewayEndpointModelMapping)
    assert mapping.mapping_id.startswith("m-")
    assert mapping.endpoint_id == endpoint.endpoint_id
    assert mapping.model_definition_id == model_def2.model_definition_id
    assert mapping.weight == 2.0


def test_attach_duplicate_model_raises(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="dup-attach-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="dup-attach-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="dup-attach-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    with pytest.raises(MlflowException, match="already attached") as exc:
        store.attach_model_to_endpoint(
            endpoint_id=endpoint.endpoint_id,
            model_definition_id=model_def.model_definition_id,
        )
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_detach_model_from_gateway_endpoint(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="detach-key", secret_value={"api_key": "value"}
    )
    model_def1 = store.create_gateway_model_definition(
        name="detach-model-1",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    model_def2 = store.create_gateway_model_definition(
        name="detach-model-2",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3",
    )
    endpoint = store.create_gateway_endpoint(
        name="detach-endpoint",
        model_definition_ids=[model_def1.model_definition_id, model_def2.model_definition_id],
    )

    store.detach_model_from_endpoint(
        endpoint_id=endpoint.endpoint_id,
        model_definition_id=model_def1.model_definition_id,
    )

    updated_endpoint = store.get_gateway_endpoint(endpoint_id=endpoint.endpoint_id)
    assert len(updated_endpoint.model_mappings) == 1
    model_def_id = updated_endpoint.model_mappings[0].model_definition_id
    assert model_def_id == model_def2.model_definition_id


def test_detach_nonexistent_mapping_raises(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="no-map-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="no-map-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="no-map-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    with pytest.raises(MlflowException, match="not attached") as exc:
        store.detach_model_from_endpoint(
            endpoint_id=endpoint.endpoint_id,
            model_definition_id="nonexistent-model",
        )
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


# =============================================================================
# Binding Operations
# =============================================================================


def test_create_gateway_endpoint_binding(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="bind-key", secret_value={"api_key": "value"})
    model_def = store.create_gateway_model_definition(
        name="bind-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="bind-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    binding = store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="job-123",
        created_by="binder",
    )

    assert isinstance(binding, GatewayEndpointBinding)
    assert binding.endpoint_id == endpoint.endpoint_id
    assert binding.resource_type == "scorer_job"
    assert binding.resource_id == "job-123"
    assert binding.created_by == "binder"


def test_delete_gateway_endpoint_binding(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="del-bind-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="del-bind-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="del-bind-endpoint", model_definition_ids=[model_def.model_definition_id]
    )
    store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="job-456",
    )

    store.delete_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="job-456",
    )

    bindings = store.list_endpoint_bindings(endpoint_id=endpoint.endpoint_id)
    assert len(bindings) == 0


def test_list_gateway_endpoint_bindings(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="list-bind-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="list-bind-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="list-bind-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="job-1",
    )
    store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="job-2",
    )

    bindings = store.list_endpoint_bindings(endpoint_id=endpoint.endpoint_id)
    assert len(bindings) == 2

    filtered = store.list_endpoint_bindings(resource_type="scorer_job", resource_id="job-1")
    assert len(filtered) == 1
    assert filtered[0].resource_id == "job-1"


# =============================================================================
# Config Resolver Operations
# =============================================================================


def test_get_resource_gateway_endpoint_configs(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="resolver-key",
        secret_value={"api_key": "sk-secret-value-123"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="resolver-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4-turbo",
    )
    endpoint = store.create_gateway_endpoint(
        name="resolver-endpoint", model_definition_ids=[model_def.model_definition_id]
    )
    store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="resolver-job-123",
    )

    configs = get_resource_endpoint_configs(
        resource_type="scorer_job",
        resource_id="resolver-job-123",
        store=store,
    )

    assert len(configs) == 1
    config = configs[0]
    assert config.endpoint_id == endpoint.endpoint_id
    assert config.endpoint_name == "resolver-endpoint"
    assert len(config.models) == 1

    model_config = config.models[0]
    assert model_config.model_definition_id == model_def.model_definition_id
    assert model_config.provider == "openai"
    assert model_config.model_name == "gpt-4-turbo"
    assert model_config.secret_value == {"api_key": "sk-secret-value-123"}


def test_get_resource_endpoint_configs_with_auth_config(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="auth-resolver-key",
        secret_value={"api_key": "aws-secret"},
        provider="bedrock",
        auth_config={"region": "us-east-1", "profile": "default"},
    )
    model_def = store.create_gateway_model_definition(
        name="auth-resolver-model",
        secret_id=secret.secret_id,
        provider="bedrock",
        model_name="anthropic.claude-3",
    )
    endpoint = store.create_gateway_endpoint(
        name="auth-resolver-endpoint", model_definition_ids=[model_def.model_definition_id]
    )
    store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="auth-job",
    )

    configs = get_resource_endpoint_configs(
        resource_type="scorer_job",
        resource_id="auth-job",
        store=store,
    )

    model_config = configs[0].models[0]
    assert model_config.auth_config == {"region": "us-east-1", "profile": "default"}


def test_get_resource_endpoint_configs_with_dict_secret(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="aws-creds",
        secret_value={
            "aws_access_key_id": "AKIA1234567890",
            "aws_secret_access_key": "secret-key-value",
        },
        provider="bedrock",
        auth_config={"auth_mode": "access_keys", "aws_region_name": "us-west-2"},
    )
    model_def = store.create_gateway_model_definition(
        name="aws-model",
        secret_id=secret.secret_id,
        provider="bedrock",
        model_name="anthropic.claude-3",
    )
    endpoint = store.create_gateway_endpoint(
        name="aws-endpoint", model_definition_ids=[model_def.model_definition_id]
    )
    store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type="scorer_job",
        resource_id="aws-job",
    )

    configs = get_resource_endpoint_configs(
        resource_type="scorer_job",
        resource_id="aws-job",
        store=store,
    )

    model_config = configs[0].models[0]
    assert model_config.secret_value == {
        "aws_access_key_id": "AKIA1234567890",
        "aws_secret_access_key": "secret-key-value",
    }
    assert model_config.auth_config == {
        "auth_mode": "access_keys",
        "aws_region_name": "us-west-2",
    }


def test_get_resource_endpoint_configs_no_bindings(store: SqlAlchemyStore):
    configs = get_resource_endpoint_configs(
        resource_type="scorer_job",
        resource_id="nonexistent-resource",
        store=store,
    )

    assert configs == []


def test_get_resource_endpoint_configs_multiple_endpoints(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="multi-key", secret_value={"api_key": "value"})
    model_def1 = store.create_gateway_model_definition(
        name="multi-model-1",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    model_def2 = store.create_gateway_model_definition(
        name="multi-model-2",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3",
    )
    endpoint1 = store.create_gateway_endpoint(
        name="multi-endpoint-1", model_definition_ids=[model_def1.model_definition_id]
    )
    endpoint2 = store.create_gateway_endpoint(
        name="multi-endpoint-2", model_definition_ids=[model_def2.model_definition_id]
    )

    store.create_endpoint_binding(
        endpoint_id=endpoint1.endpoint_id,
        resource_type="scorer_job",
        resource_id="multi-resource",
    )
    store.create_endpoint_binding(
        endpoint_id=endpoint2.endpoint_id,
        resource_type="scorer_job",
        resource_id="multi-resource",
    )

    configs = get_resource_endpoint_configs(
        resource_type="scorer_job",
        resource_id="multi-resource",
        store=store,
    )

    assert len(configs) == 2
    endpoint_names = {c.endpoint_name for c in configs}
    assert endpoint_names == {"multi-endpoint-1", "multi-endpoint-2"}


def test_get_gateway_endpoint_config(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="ep-config-key",
        secret_value={"api_key": "sk-endpoint-secret-789"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="ep-config-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
    )
    endpoint = store.create_gateway_endpoint(
        name="ep-config-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    config = get_endpoint_config(
        endpoint_name=endpoint.name,
        store=store,
    )

    assert config.endpoint_id == endpoint.endpoint_id
    assert config.endpoint_name == "ep-config-endpoint"
    assert len(config.models) == 1

    model_config = config.models[0]
    assert model_config.model_definition_id == model_def.model_definition_id
    assert model_config.provider == "openai"
    assert model_config.model_name == "gpt-4o"
    assert model_config.secret_value == {"api_key": "sk-endpoint-secret-789"}


def test_get_gateway_endpoint_config_with_auth_config(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="ep-auth-key",
        secret_value={"api_key": "bedrock-secret"},
        provider="bedrock",
        auth_config={"region": "eu-west-1", "project_id": "test-project"},
    )
    model_def = store.create_gateway_model_definition(
        name="ep-auth-model",
        secret_id=secret.secret_id,
        provider="bedrock",
        model_name="anthropic.claude-3-sonnet",
    )
    endpoint = store.create_gateway_endpoint(
        name="ep-auth-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    config = get_endpoint_config(
        endpoint_name=endpoint.name,
        store=store,
    )

    model_config = config.models[0]
    assert model_config.auth_config == {"region": "eu-west-1", "project_id": "test-project"}


def test_get_gateway_endpoint_config_multiple_models(store: SqlAlchemyStore):
    secret1 = store.create_gateway_secret(
        secret_name="ep-multi-key-1", secret_value={"api_key": "secret-1"}
    )
    secret2 = store.create_gateway_secret(
        secret_name="ep-multi-key-2", secret_value={"api_key": "secret-2"}
    )

    model_def1 = store.create_gateway_model_definition(
        name="ep-multi-model-1",
        secret_id=secret1.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    model_def2 = store.create_gateway_model_definition(
        name="ep-multi-model-2",
        secret_id=secret2.secret_id,
        provider="anthropic",
        model_name="claude-3-opus",
    )

    endpoint = store.create_gateway_endpoint(
        name="ep-multi-endpoint",
        model_definition_ids=[model_def1.model_definition_id, model_def2.model_definition_id],
    )

    config = get_endpoint_config(
        endpoint_name=endpoint.name,
        store=store,
    )

    assert len(config.models) == 2
    providers = {m.provider for m in config.models}
    assert providers == {"openai", "anthropic"}
    model_names = {m.model_name for m in config.models}
    assert model_names == {"gpt-4", "claude-3-opus"}


def test_get_gateway_endpoint_config_nonexistent_endpoint_raises(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found") as exc:
        get_endpoint_config(
            endpoint_name="nonexistent-endpoint",
            store=store,
        )
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


# =============================================================================
# Endpoint Tag Operations
# =============================================================================


def test_set_gateway_endpoint_tag(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(secret_name="tag-key", secret_value={"api_key": "value"})
    model_def = store.create_gateway_model_definition(
        name="tag-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="tag-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    tag = GatewayEndpointTag(key="env", value="production")
    store.set_gateway_endpoint_tag(endpoint.endpoint_id, tag)

    retrieved = store.get_gateway_endpoint(endpoint_id=endpoint.endpoint_id)
    assert len(retrieved.tags) == 1
    assert retrieved.tags[0].key == "env"
    assert retrieved.tags[0].value == "production"


def test_set_gateway_endpoint_tag_update_existing(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="tag-upd-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="tag-upd-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="tag-upd-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    store.set_gateway_endpoint_tag(endpoint.endpoint_id, GatewayEndpointTag(key="env", value="dev"))
    store.set_gateway_endpoint_tag(
        endpoint.endpoint_id, GatewayEndpointTag(key="env", value="production")
    )

    retrieved = store.get_gateway_endpoint(endpoint_id=endpoint.endpoint_id)
    assert len(retrieved.tags) == 1
    assert retrieved.tags[0].key == "env"
    assert retrieved.tags[0].value == "production"


def test_set_multiple_endpoint_tags(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="multi-tag-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="multi-tag-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="multi-tag-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    store.set_gateway_endpoint_tag(
        endpoint.endpoint_id, GatewayEndpointTag(key="env", value="production")
    )
    store.set_gateway_endpoint_tag(endpoint.endpoint_id, GatewayEndpointTag(key="team", value="ml"))
    store.set_gateway_endpoint_tag(
        endpoint.endpoint_id, GatewayEndpointTag(key="version", value="v1")
    )

    retrieved = store.get_gateway_endpoint(endpoint_id=endpoint.endpoint_id)
    assert len(retrieved.tags) == 3
    tag_dict = {t.key: t.value for t in retrieved.tags}
    assert tag_dict == {"env": "production", "team": "ml", "version": "v1"}


def test_set_gateway_endpoint_tag_nonexistent_endpoint_raises(store: SqlAlchemyStore):
    tag = GatewayEndpointTag(key="env", value="production")
    with pytest.raises(MlflowException, match="not found") as exc:
        store.set_gateway_endpoint_tag("nonexistent-endpoint", tag)
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_delete_gateway_endpoint_tag(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="del-tag-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="del-tag-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="del-tag-endpoint", model_definition_ids=[model_def.model_definition_id]
    )
    store.set_gateway_endpoint_tag(
        endpoint.endpoint_id, GatewayEndpointTag(key="env", value="production")
    )
    store.set_gateway_endpoint_tag(endpoint.endpoint_id, GatewayEndpointTag(key="team", value="ml"))

    store.delete_gateway_endpoint_tag(endpoint.endpoint_id, "env")

    retrieved = store.get_gateway_endpoint(endpoint_id=endpoint.endpoint_id)
    assert len(retrieved.tags) == 1
    assert retrieved.tags[0].key == "team"


def test_delete_gateway_endpoint_tag_nonexistent_endpoint_raises(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.delete_gateway_endpoint_tag("nonexistent-endpoint", "env")
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_delete_gateway_endpoint_tag_nonexistent_key_no_op(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="del-noop-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="del-noop-model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4"
    )
    endpoint = store.create_gateway_endpoint(
        name="del-noop-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Should not raise even if tag doesn't exist
    store.delete_gateway_endpoint_tag(endpoint.endpoint_id, "nonexistent-key")

    retrieved = store.get_gateway_endpoint(endpoint_id=endpoint.endpoint_id)
    assert len(retrieved.tags) == 0


def test_endpoint_tags_deleted_with_endpoint(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="cascade-tag-key", secret_value={"api_key": "value"}
    )
    model_def = store.create_gateway_model_definition(
        name="cascade-tag-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="cascade-tag-endpoint", model_definition_ids=[model_def.model_definition_id]
    )
    store.set_gateway_endpoint_tag(
        endpoint.endpoint_id, GatewayEndpointTag(key="env", value="production")
    )

    store.delete_gateway_endpoint(endpoint.endpoint_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_gateway_endpoint(endpoint_id=endpoint.endpoint_id)
