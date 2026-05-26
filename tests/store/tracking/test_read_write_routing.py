import os
import shutil
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy
import sqlalchemy.orm

from mlflow.entities import Metric, ViewType
from mlflow.exceptions import MlflowException
from mlflow.store.db.utils import _get_routing_session_maker
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

ARTIFACT_URI = "artifact_folder"


@pytest.fixture
def write_db_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'write.db'}"


@pytest.fixture
def read_db_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'read.db'}"


@pytest.fixture
def store_with_replica(tmp_path: Path, write_db_uri, read_db_uri):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    write_store = SqlAlchemyStore(write_db_uri, str(artifact_uri))
    write_store.engine.dispose()
    shutil.copy(tmp_path / "write.db", tmp_path / "read.db")
    return SqlAlchemyStore(write_db_uri, str(artifact_uri), read_db_uri=read_db_uri)


@pytest.fixture
def store_no_replica(tmp_path: Path, write_db_uri):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(write_db_uri, str(artifact_uri))


# --- TestManagedSessionMakerReadOnlyParam ---


def test_managed_session_maker_accepts_read_only_false(store_no_replica):
    with store_no_replica.ManagedSessionMaker(read_only=False) as session:
        assert session is not None


def test_managed_session_maker_accepts_read_only_true(store_no_replica):
    with store_no_replica.ManagedSessionMaker(read_only=True) as session:
        assert session is not None


def test_managed_session_maker_default_no_read_only(store_no_replica):
    with store_no_replica.ManagedSessionMaker() as session:
        assert session is not None


# --- TestDualEngineSetup ---


def test_no_replica_uses_single_engine(store_no_replica):
    assert store_no_replica.read_engine is None


def test_replica_creates_separate_engine(store_with_replica, write_db_uri, read_db_uri):
    assert store_with_replica.read_engine is not None
    assert store_with_replica.engine is not store_with_replica.read_engine


def test_same_uri_skips_replica(tmp_path: Path, write_db_uri):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    store = SqlAlchemyStore(write_db_uri, str(artifact_uri), read_db_uri=write_db_uri)
    assert store.read_engine is None


def test_none_read_uri_skips_replica(store_no_replica):
    assert store_no_replica.read_engine is None


# --- TestRoutingSessionMaker ---


def test_routing_read_only_true_uses_read_session():
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
    store_with_replica.create_experiment("test_write")
    with store_with_replica.ManagedSessionMaker(read_only=False) as session:
        from mlflow.store.tracking.dbmodels.models import SqlExperiment

        exp = session.query(SqlExperiment).filter_by(name="test_write").first()
        assert exp is not None


def test_read_from_replica(store_with_replica, tmp_path):
    read_store = SqlAlchemyStore(
        str(store_with_replica.read_engine.url),
        str(tmp_path / "artifacts"),
    )
    read_store.create_experiment("only_on_replica")

    exp = store_with_replica.get_experiment_by_name("only_on_replica")
    assert exp is not None
    assert exp.name == "only_on_replica"


def test_search_experiments_uses_replica(store_with_replica, tmp_path):
    read_store = SqlAlchemyStore(
        str(store_with_replica.read_engine.url),
        str(tmp_path / "artifacts"),
    )
    read_store.create_experiment("replica_exp")

    results = store_with_replica.search_experiments(view_type=ViewType.ALL)
    names = [e.name for e in results]
    assert "replica_exp" in names


def test_get_run_uses_replica(store_with_replica, tmp_path):
    read_store = SqlAlchemyStore(
        str(store_with_replica.read_engine.url),
        str(tmp_path / "artifacts"),
    )
    exp_id = read_store.create_experiment("run_test")
    run = read_store.create_run(exp_id, "user", 0, [], None)
    run_id = run.info.run_id

    fetched = store_with_replica.get_run(run_id)
    assert fetched.info.run_id == run_id


# --- TestBackwardCompatibility ---


def test_backward_compat_create_and_read_experiment(store_no_replica):
    exp_id = store_no_replica.create_experiment("compat_test")
    exp = store_no_replica.get_experiment(exp_id)
    assert exp.name == "compat_test"


def test_backward_compat_create_and_search_runs(store_no_replica):
    exp_id = store_no_replica.create_experiment("run_compat")
    run = store_no_replica.create_run(exp_id, "user", 0, [], None)
    fetched = store_no_replica.get_run(run.info.run_id)
    assert fetched.info.run_id == run.info.run_id


def test_backward_compat_log_and_get_metrics(store_no_replica):
    exp_id = store_no_replica.create_experiment("metric_compat")
    run = store_no_replica.create_run(exp_id, "user", 0, [], None)
    store_no_replica.log_metric(run.info.run_id, Metric("acc", 0.95, 1, 0))
    history = store_no_replica.get_metric_history(run.info.run_id, "acc")
    assert len(history) == 1
    assert history[0].value == 0.95


# --- TestServerWiring ---


def test_run_server_passes_read_uri_env_var():
    from mlflow.server import _run_server
    from mlflow.server.constants import READ_REPLICA_BACKEND_STORE_URI_ENV_VAR

    with mock.patch("mlflow.server._exec_cmd") as mock_exec:
        mock_exec.side_effect = SystemExit(0)
        try:
            _run_server(
                file_store_path="sqlite:///test.db",
                read_replica_backend_store_uri="sqlite:///read.db",
                registry_store_uri=None,
                default_artifact_root="/tmp/artifacts",
                serve_artifacts=False,
                artifacts_only=False,
                artifacts_destination=None,
                host="127.0.0.1",
                port=5000,
            )
        except SystemExit:
            pass

        call_kwargs = mock_exec.call_args
        env_map = call_kwargs[1].get("extra_env", {}) if call_kwargs[1] else {}
        if not env_map and call_kwargs[0]:
            for arg in call_kwargs[0]:
                if isinstance(arg, dict):
                    env_map = arg
                    break
        assert READ_REPLICA_BACKEND_STORE_URI_ENV_VAR in env_map
        assert env_map[READ_REPLICA_BACKEND_STORE_URI_ENV_VAR] == "sqlite:///read.db"


def test_run_server_omits_read_uri_when_none():
    from mlflow.server import _run_server
    from mlflow.server.constants import READ_REPLICA_BACKEND_STORE_URI_ENV_VAR

    with mock.patch("mlflow.server._exec_cmd") as mock_exec:
        mock_exec.side_effect = SystemExit(0)
        try:
            _run_server(
                file_store_path="sqlite:///test.db",
                registry_store_uri=None,
                default_artifact_root="/tmp/artifacts",
                serve_artifacts=False,
                artifacts_only=False,
                artifacts_destination=None,
                host="127.0.0.1",
                port=5000,
            )
        except SystemExit:
            pass

        call_kwargs = mock_exec.call_args
        env_map = call_kwargs[1].get("extra_env", {}) if call_kwargs[1] else {}
        if not env_map and call_kwargs[0]:
            for arg in call_kwargs[0]:
                if isinstance(arg, dict):
                    env_map = arg
                    break
        assert READ_REPLICA_BACKEND_STORE_URI_ENV_VAR not in env_map


def test_initialize_backend_stores_sets_env_var():
    from mlflow.server.constants import READ_REPLICA_BACKEND_STORE_URI_ENV_VAR
    from mlflow.server.handlers import initialize_backend_stores

    # `initialize_backend_stores` mutates `os.environ` directly. Don't use `monkeypatch.delenv`
    # for cleanup: it records the env var's *current* value (set by `initialize_backend_stores`)
    # for rollback, which causes the value to be restored at teardown and leak into later tests.
    os.environ.pop(
        READ_REPLICA_BACKEND_STORE_URI_ENV_VAR, None
    )  # clint: disable=os-environ-delete-in-test
    try:
        with (
            mock.patch("mlflow.server.handlers._get_tracking_store"),
            mock.patch("mlflow.server.handlers._get_model_registry_store"),
        ):
            initialize_backend_stores(
                backend_store_uri="sqlite:///test.db",
                read_replica_backend_store_uri="sqlite:///read.db",
            )
            assert os.environ.get(READ_REPLICA_BACKEND_STORE_URI_ENV_VAR) == "sqlite:///read.db"
    finally:
        os.environ.pop(
            READ_REPLICA_BACKEND_STORE_URI_ENV_VAR, None
        )  # clint: disable=os-environ-delete-in-test


def test_get_sqlalchemy_store_reads_env_var(tmp_path, monkeypatch):
    from mlflow.server.constants import READ_REPLICA_BACKEND_STORE_URI_ENV_VAR
    from mlflow.server.handlers import TrackingStoreRegistryWrapper

    write_uri = f"sqlite:///{tmp_path / 'write.db'}"
    read_uri = f"sqlite:///{tmp_path / 'read.db'}"

    init_store = SqlAlchemyStore(write_uri, str(tmp_path / "artifacts"))
    init_store.engine.dispose()
    shutil.copy(tmp_path / "write.db", tmp_path / "read.db")

    monkeypatch.setenv(READ_REPLICA_BACKEND_STORE_URI_ENV_VAR, read_uri)
    store = TrackingStoreRegistryWrapper._get_sqlalchemy_store(
        write_uri, str(tmp_path / "artifacts")
    )
    assert store.read_engine is not None


# --- TestReadOnlyAnnotations ---
# With read_only=True as the default, write methods must explicitly pass read_only=False.
# The before_flush listener (active when _MLFLOW_TESTING is set) enforces this at runtime.


def test_write_on_read_only_session_raises(store_with_replica, monkeypatch):
    monkeypatch.setenv("_MLFLOW_TESTING", "true")
    from mlflow.store.tracking.dbmodels.models import SqlExperiment

    def _do_write():
        with store_with_replica.ManagedSessionMaker() as session:
            session.add(SqlExperiment(name="should_fail", lifecycle_stage="active"))
            session.flush()

    with pytest.raises(MlflowException, match="Write operation detected"):
        _do_write()


def test_write_on_write_session_succeeds(store_with_replica, monkeypatch):
    monkeypatch.setenv("_MLFLOW_TESTING", "true")
    from mlflow.store.tracking.dbmodels.models import SqlExperiment

    with store_with_replica.ManagedSessionMaker(read_only=False) as session:
        session.add(SqlExperiment(name="should_succeed", lifecycle_stage="active"))
        session.flush()
