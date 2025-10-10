import os

import pytest

from mlflow.secrets.crypto import SecretManager
from mlflow.secrets.scope import SecretScope
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "mlflow.db"
    artifact_path = tmp_path / "artifacts"
    backend_uri = f"sqlite:///{db_path}"
    artifact_root = f"file://{artifact_path}"
    os.environ["MLFLOW_SECRET_MASTER_KEY"] = "test_master_key_1234567890123456789012"
    os.environ["MLFLOW_SECRETS_VALIDATION_MODE"] = "skip"
    store = SqlAlchemyStore(backend_uri, artifact_root)
    yield store
    os.environ.pop("MLFLOW_SECRET_MASTER_KEY", None)
    os.environ.pop("MLFLOW_SECRETS_VALIDATION_MODE", None)


@pytest.fixture
def secret_manager():
    os.environ["MLFLOW_SECRET_MASTER_KEY"] = "test_master_key_1234567890123456789012"
    manager = SecretManager()
    yield manager
    os.environ.pop("MLFLOW_SECRET_MASTER_KEY", None)


def test_set_secret_with_envelope_encryption(store, secret_manager):
    name = "test_secret"
    value = "test_value"

    store.set_secret(name, value, SecretScope.GLOBAL)

    with store.ManagedSessionMaker() as session:
        from mlflow.store.tracking.dbmodels.models import SqlSecret

        name_hash = secret_manager.hash_name(name)
        secret = (
            session.query(SqlSecret)
            .filter_by(name_hash=name_hash, scope=SecretScope.GLOBAL)
            .first()
        )

        assert secret is not None
        assert secret.encrypted_dek is not None
        assert secret.master_key_version == 1


def test_get_secret_with_envelope_encryption(store):
    name = "test_secret"
    value = "test_value"

    store.set_secret(name, value, SecretScope.GLOBAL)
    retrieved_value = store.get_secret(name, SecretScope.GLOBAL)

    assert retrieved_value == value


def test_envelope_encryption_preserves_value(store):
    name = "test_secret"
    value = "a" * 10000

    store.set_secret(name, value, SecretScope.GLOBAL)
    retrieved_value = store.get_secret(name, SecretScope.GLOBAL)

    assert retrieved_value == value


def test_envelope_encryption_with_unicode(store):
    name = "test_secret"
    value = "🔐 Secret with émojis and spëcial çharacters 你好"

    store.set_secret(name, value, SecretScope.GLOBAL)
    retrieved_value = store.get_secret(name, SecretScope.GLOBAL)

    assert retrieved_value == value


def test_envelope_encryption_with_scope_id(store):
    name = "test_secret"
    value = "test_value"
    scope_id = 123

    store.set_secret(name, value, SecretScope.SCORER, scope_id)
    retrieved_value = store.get_secret(name, SecretScope.SCORER, scope_id)

    assert retrieved_value == value


def test_list_secret_names_with_envelope_encryption(store):
    store.set_secret("secret1", "value1", SecretScope.GLOBAL)
    store.set_secret("secret2", "value2", SecretScope.GLOBAL)

    names = store.list_secret_names(SecretScope.GLOBAL, None)

    assert sorted(names) == ["secret1", "secret2"]


def test_delete_secret_with_envelope_encryption(store):
    name = "test_secret"
    value = "test_value"

    store.set_secret(name, value, SecretScope.GLOBAL)
    store.delete_secret(name, SecretScope.GLOBAL)

    with pytest.raises(Exception, match="not found"):
        store.get_secret(name, SecretScope.GLOBAL)


def test_update_secret_with_envelope_encryption(store, secret_manager):
    name = "test_secret"
    value1 = "value1"
    value2 = "value2"

    store.set_secret(name, value1, SecretScope.GLOBAL)

    with store.ManagedSessionMaker() as session:
        from mlflow.store.tracking.dbmodels.models import SqlSecret

        name_hash = secret_manager.hash_name(name)
        secret = (
            session.query(SqlSecret)
            .filter_by(name_hash=name_hash, scope=SecretScope.GLOBAL)
            .first()
        )
        first_dek = secret.encrypted_dek

    store.set_secret(name, value2, SecretScope.GLOBAL)

    with store.ManagedSessionMaker() as session:
        from mlflow.store.tracking.dbmodels.models import SqlSecret

        name_hash = secret_manager.hash_name(name)
        secret = (
            session.query(SqlSecret)
            .filter_by(name_hash=name_hash, scope=SecretScope.GLOBAL)
            .first()
        )
        second_dek = secret.encrypted_dek

    assert first_dek != second_dek

    retrieved_value = store.get_secret(name, SecretScope.GLOBAL)
    assert retrieved_value == value2


def test_envelope_encryption_different_deks_per_secret(store, secret_manager):
    store.set_secret("secret1", "value1", SecretScope.GLOBAL)
    store.set_secret("secret2", "value2", SecretScope.GLOBAL)

    with store.ManagedSessionMaker() as session:
        from mlflow.store.tracking.dbmodels.models import SqlSecret

        name_hash1 = secret_manager.hash_name("secret1")
        name_hash2 = secret_manager.hash_name("secret2")

        secret1 = (
            session.query(SqlSecret)
            .filter_by(name_hash=name_hash1, scope=SecretScope.GLOBAL)
            .first()
        )
        secret2 = (
            session.query(SqlSecret)
            .filter_by(name_hash=name_hash2, scope=SecretScope.GLOBAL)
            .first()
        )

        assert secret1.encrypted_dek != secret2.encrypted_dek
