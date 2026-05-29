import shutil
from pathlib import Path
from unittest import mock

import pytest

from mlflow.server.auth.config import AuthConfig, read_auth_config
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def write_db_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'auth_write.db'}"


@pytest.fixture
def read_db_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'auth_read.db'}"


@pytest.fixture
def store_with_replica(tmp_path: Path, write_db_uri, read_db_uri):
    store = SqlAlchemyStore()
    store.init_db(write_db_uri)
    store.engine.dispose()
    shutil.copy(tmp_path / "auth_write.db", tmp_path / "auth_read.db")
    store2 = SqlAlchemyStore()
    store2.init_db(write_db_uri, read_db_uri=read_db_uri)
    return store2


@pytest.fixture
def store_no_replica(tmp_path: Path, write_db_uri):
    store = SqlAlchemyStore()
    store.init_db(write_db_uri)
    return store


# --- TestDualEngineSetup ---


def test_no_replica_uses_single_engine(store_no_replica):
    assert store_no_replica.read_engine is None


def test_replica_creates_separate_engine(store_with_replica):
    assert store_with_replica.read_engine is not None
    assert store_with_replica.engine is not store_with_replica.read_engine


def test_same_uri_skips_replica(tmp_path: Path, write_db_uri):
    store = SqlAlchemyStore()
    store.init_db(write_db_uri, read_db_uri=write_db_uri)
    assert store.read_engine is None


def test_none_read_uri_skips_replica(store_no_replica):
    assert store_no_replica.read_engine is None


# --- TestReadWriteRouting ---


def test_write_goes_to_primary(store_with_replica):
    user = store_with_replica.create_user("test_user", "password12345678")
    assert user.username == "test_user"


def test_read_from_replica(store_with_replica, tmp_path, write_db_uri, read_db_uri):
    read_store = SqlAlchemyStore()
    read_store.init_db(read_db_uri)
    read_store.create_user("replica_user", "password12345678")

    user = store_with_replica.get_user("replica_user")
    assert user.username == "replica_user"


def test_has_user_uses_replica(store_with_replica, read_db_uri):
    read_store = SqlAlchemyStore()
    read_store.init_db(read_db_uri)
    read_store.create_user("check_user", "password12345678")

    assert store_with_replica.has_user("check_user") is True


def test_authenticate_uses_replica(store_with_replica, read_db_uri):
    read_store = SqlAlchemyStore()
    read_store.init_db(read_db_uri)
    read_store.create_user("auth_user", "secretpass12345678")

    assert store_with_replica.authenticate_user("auth_user", "secretpass12345678") is True
    assert store_with_replica.authenticate_user("auth_user", "wrongpass12345678") is False


# --- TestBackwardCompatibility ---


def test_backward_compat_create_and_read_user(store_no_replica):
    store_no_replica.create_user("compat_user", "password12345678")
    user = store_no_replica.get_user("compat_user")
    assert user.username == "compat_user"


def test_backward_compat_list_users(store_no_replica):
    store_no_replica.create_user("list_user", "password12345678")
    users = store_no_replica.list_users()
    assert any(u.username == "list_user" for u in users)


def test_backward_compat_authenticate(store_no_replica):
    store_no_replica.create_user("auth_compat", "mypassword12345678")
    assert store_no_replica.authenticate_user("auth_compat", "mypassword12345678") is True


# --- TestAuthConfig ---


def test_config_with_read_uri(tmp_path):
    config = AuthConfig(
        default_permission="READ",
        database_uri="sqlite:///auth.db",
        read_database_uri="sqlite:///auth_read.db",
        admin_username="admin",
        admin_password="password",
        authorization_function="mlflow.server.auth:authenticate_request_basic_auth",
        grant_default_workspace_access=False,
        workspace_cache_max_size=10000,
        workspace_cache_ttl_seconds=3600,
        auth_cache_max_size=10000,
        auth_cache_ttl_seconds=3600,
    )
    assert config.read_database_uri == "sqlite:///auth_read.db"


def test_config_without_read_uri(tmp_path):
    config = AuthConfig(
        default_permission="READ",
        database_uri="sqlite:///auth.db",
        read_database_uri=None,
        admin_username="admin",
        admin_password="password",
        authorization_function="mlflow.server.auth:authenticate_request_basic_auth",
        grant_default_workspace_access=False,
        workspace_cache_max_size=10000,
        workspace_cache_ttl_seconds=3600,
        auth_cache_max_size=10000,
        auth_cache_ttl_seconds=3600,
    )
    assert config.read_database_uri is None


def test_read_auth_config_without_read_uri(tmp_path):
    ini_path = tmp_path / "test_auth.ini"
    ini_path.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        "database_uri = sqlite:///test.db\n"
        "admin_username = admin\n"
        "admin_password = password\n"
    )

    with mock.patch("mlflow.server.auth.config._get_auth_config_path", return_value=str(ini_path)):
        config = read_auth_config()
        assert config.read_database_uri is None
        assert config.database_uri == "sqlite:///test.db"


def test_read_auth_config_with_read_uri(tmp_path):
    ini_path = tmp_path / "test_auth.ini"
    ini_path.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        "database_uri = sqlite:///test.db\n"
        "read_database_uri = sqlite:///test_read.db\n"
        "admin_username = admin\n"
        "admin_password = password\n"
    )

    with mock.patch("mlflow.server.auth.config._get_auth_config_path", return_value=str(ini_path)):
        config = read_auth_config()
        assert config.read_database_uri == "sqlite:///test_read.db"
