import shutil
from pathlib import Path

import pytest
import sqlalchemy
import sqlalchemy.orm

from mlflow.store.db.utils import _get_routing_session_maker
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def write_db_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'write.db'}"


@pytest.fixture
def read_db_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'read.db'}"


@pytest.fixture
def store_with_replica(tmp_path: Path, write_db_uri, read_db_uri):
    write_store = SqlAlchemyStore(write_db_uri)
    write_store.engine.dispose()
    shutil.copy(tmp_path / "write.db", tmp_path / "read.db")
    return SqlAlchemyStore(write_db_uri, read_db_uri=read_db_uri)


@pytest.fixture
def store_no_replica(tmp_path: Path, write_db_uri):
    return SqlAlchemyStore(write_db_uri)


# --- TestDualEngineSetup ---


def test_no_replica_uses_single_engine(store_no_replica):
    assert store_no_replica.read_engine is None


def test_replica_creates_separate_engine(store_with_replica):
    assert store_with_replica.read_engine is not None
    assert store_with_replica.engine is not store_with_replica.read_engine


def test_same_uri_skips_replica(tmp_path: Path, write_db_uri):
    store = SqlAlchemyStore(write_db_uri, read_db_uri=write_db_uri)
    assert store.read_engine is None


def test_none_read_uri_skips_replica(store_no_replica):
    assert store_no_replica.read_engine is None


# --- TestRoutingSessionMaker ---


def test_routing_read_only_uses_read_engine():
    write_engine = sqlalchemy.create_engine("sqlite:///:memory:")
    read_engine = sqlalchemy.create_engine("sqlite:///:memory:")
    write_sm = sqlalchemy.orm.sessionmaker(bind=write_engine)
    read_sm = sqlalchemy.orm.sessionmaker(bind=read_engine)
    routing_maker = _get_routing_session_maker(write_sm, read_sm, "sqlite")

    with routing_maker(read_only=True) as session:
        assert session.get_bind() is read_engine
    with routing_maker(read_only=False) as session:
        assert session.get_bind() is write_engine
    with routing_maker() as session:
        # Default is read_only=True
        assert session.get_bind() is read_engine


# --- TestReadWriteRouting ---


def test_write_goes_to_primary(store_with_replica):
    store_with_replica.create_registered_model("test_model")


def test_read_from_replica(store_with_replica, tmp_path):
    read_store = SqlAlchemyStore(str(store_with_replica.read_engine.url))
    read_store.create_registered_model("only_on_replica")

    rm = store_with_replica.get_registered_model("only_on_replica")
    assert rm.name == "only_on_replica"


def test_search_registered_models_uses_replica(store_with_replica, tmp_path):
    read_store = SqlAlchemyStore(str(store_with_replica.read_engine.url))
    read_store.create_registered_model("replica_model")

    results = store_with_replica.search_registered_models()
    names = [rm.name for rm in results]
    assert "replica_model" in names


def test_get_latest_versions_uses_replica(store_with_replica, tmp_path):
    read_store = SqlAlchemyStore(str(store_with_replica.read_engine.url))
    read_store.create_registered_model("version_test")

    versions = store_with_replica.get_latest_versions("version_test")
    assert versions == []


# --- TestBackwardCompatibility ---


def test_backward_compat_create_and_read_model(store_no_replica):
    store_no_replica.create_registered_model("compat_model")
    fetched = store_no_replica.get_registered_model("compat_model")
    assert fetched.name == "compat_model"


def test_backward_compat_search_models(store_no_replica):
    store_no_replica.create_registered_model("search_test")
    results = store_no_replica.search_registered_models()
    names = [rm.name for rm in results]
    assert "search_test" in names


def test_backward_compat_create_and_get_model_version(store_no_replica):
    store_no_replica.create_registered_model("mv_test")
    mv = store_no_replica.create_model_version("mv_test", "file:///tmp/model", "run123")
    fetched = store_no_replica.get_model_version("mv_test", mv.version)
    assert fetched.name == "mv_test"


# --- TestServerWiring ---


def test_registry_handler_passes_read_uri(tmp_path, monkeypatch):
    from mlflow.server.constants import READ_REPLICA_BACKEND_STORE_URI_ENV_VAR
    from mlflow.server.handlers import ModelRegistryStoreRegistryWrapper

    write_uri = f"sqlite:///{tmp_path / 'write.db'}"
    read_uri = f"sqlite:///{tmp_path / 'read.db'}"

    init_store = SqlAlchemyStore(write_uri)
    init_store.engine.dispose()
    shutil.copy(tmp_path / "write.db", tmp_path / "read.db")

    monkeypatch.setenv(READ_REPLICA_BACKEND_STORE_URI_ENV_VAR, read_uri)
    store = ModelRegistryStoreRegistryWrapper._get_sqlalchemy_store(write_uri)
    assert store.read_engine is not None


def test_registry_handler_no_read_uri(tmp_path, monkeypatch):
    from mlflow.server.constants import READ_REPLICA_BACKEND_STORE_URI_ENV_VAR
    from mlflow.server.handlers import ModelRegistryStoreRegistryWrapper

    write_uri = f"sqlite:///{tmp_path / 'write.db'}"

    monkeypatch.delenv(READ_REPLICA_BACKEND_STORE_URI_ENV_VAR, raising=False)
    store = ModelRegistryStoreRegistryWrapper._get_sqlalchemy_store(write_uri)
    assert store.read_engine is None
