import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest import mock

import pytest
import sqlalchemy as sa
from testcontainers.community.mssql import SqlServerContainer
from testcontainers.community.mysql import MySqlContainer
from testcontainers.community.postgres import PostgresContainer

from mlflow.entities.skill_source import GitSource
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE, ErrorCode
from mlflow.store.tracking.dbmodels.models import SqlSkillVersion
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture(scope="module", params=["mysql", "mssql", "postgres"])
def database_uri(request):
    if request.param == "mysql":
        database = MySqlContainer("mysql:8.0", dialect="pymysql").with_command(
            "--character-set-server=utf8mb4 --collation-server=utf8mb4_0900_ai_ci "
            "--log-bin-trust-function-creators=1"
        )
    elif request.param == "mssql":
        database = SqlServerContainer(
            "mcr.microsoft.com/mssql/server:2022-latest", platform="linux/amd64"
        ).with_env("MSSQL_COLLATION", "SQL_Latin1_General_CP1_CI_AS")
    else:
        database = PostgresContainer("postgres:16")
    with database:
        yield database.get_connection_url()


@pytest.fixture(params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def store(database_uri, request, monkeypatch, tmp_path):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, str(request.param).lower())
    store_cls = WorkspaceAwareSqlAlchemyStore if request.param else SqlAlchemyStore
    with WorkspaceContext("default"):
        yield store_cls(database_uri, tmp_path.as_uri())


@pytest.fixture
def definition():
    return {
        "name": f"reviewer-{uuid.uuid4().hex}",
        "source_type": "git",
        "source": "https://example.com/skills.git",
        "ref": "main",
        "subpath": "skills/reviewer",
        "digest": "a" * 64,
    }


@pytest.fixture(
    params=[
        ("source", "https://example.com/Skills.git"),
        ("ref", "Main"),
        ("subpath", "skills/Reviewer"),
    ],
    ids=["repository-path", "git-ref", "skill-subpath"],
)
def case_change(request):
    return request.param


def test_bulk_registration_reuses_highest_case_sensitive_match(store, definition, case_change):
    original = store.bulk_register_skills([definition], created_by="original")[0]
    exact = store.create_skill_version(**definition, created_by="second")
    field, value = case_change
    changed = {**definition, field: value}

    imported = store.bulk_register_skills([changed], created_by="importer")[0]

    assert (original.version, exact.version, imported.version) == (1, 2, 3)
    assert imported.source == GitSource(
        url=changed["source"], ref=changed["ref"], subpath=changed["subpath"]
    )
    assert imported.created_by == "importer"
    assert store.bulk_register_skills([changed])[0] == imported
    assert store.bulk_register_skills([definition], created_by="importer")[0] == exact
    assert store.get_skill_version(definition["name"], 1) == original
    assert store.get_skill_version(definition["name"], 3) == imported
    with store.ManagedSessionMaker() as session:
        versions = store._get_query(session, SqlSkillVersion).filter_by(name=definition["name"])
        assert versions.count() == 3
        if session.bind.dialect.name in {"mysql", "mssql"}:
            # Ensure the fixture actually exercises a case-insensitive SQL comparison.
            assert versions.filter(getattr(SqlSkillVersion, field) == value).count() == 3


def _wait_for_postgres_blocker(engine, blocked_pid, blocker_pid):
    deadline = time.monotonic() + 10
    with engine.connect() as observer:
        while time.monotonic() < deadline:
            blockers = observer.execute(
                sa.text("SELECT pg_blocking_pids(:pid)"), {"pid": blocked_pid}
            ).scalar_one()
            if blocker_pid in blockers:
                return
            time.sleep(0.01)
    pytest.fail("Standalone INSERT did not reach the bulk transaction's parent lock")


@pytest.mark.parametrize("victim", ["bulk", "single"])
@pytest.mark.parametrize("database_uri", ["postgres"], indirect=True, scope="module")
def test_postgres_bulk_and_single_registration_deadlock(store, definition, victim):
    store.create_skill(definition["name"])
    allocated = Event()
    resume_single = Event()
    bulk_locked = Event()
    resume_bulk = Event()
    pids = {}
    attempts = {"bulk": [], "single": []}
    deadlocks = []
    persist = store._persist_skill_version

    def coordinate_persistence(*args, **kwargs):
        role = kwargs["created_by"]
        attempts[role].append(kwargs["version"])
        session = kwargs["session"]
        connection = session.connection()
        connection.info["skill_registration_role"] = role
        pids[role] = session.execute(sa.text("SELECT pg_backend_pid()")).scalar_one()
        # The selected victim checks the completed cycle first. Timeouts only bound failures;
        # pg_blocking_pids below establishes that the foreign-key lock wait really happened.
        timeout = "100ms" if role == victim else "10s"
        session.execute(sa.text(f"SET LOCAL deadlock_timeout = '{timeout}'"))
        session.execute(sa.text("SET LOCAL statement_timeout = '15s'"))
        if len(attempts[role]) == 1:
            if role == "single":
                allocated.set()
                assert resume_single.wait(timeout=10)
            else:
                bulk_locked.set()
                assert resume_bulk.wait(timeout=10)
        return persist(*args, **kwargs)

    def record_deadlock(context):
        if getattr(context.original_exception, "pgcode", None) == "40P01":
            deadlocks.append(context.connection.info["skill_registration_role"])

    def register(role):
        with WorkspaceContext("default"):
            if role == "single":
                return store.create_skill_version(**definition, created_by=role)
            return store.bulk_register_skills([definition], created_by=role)[0]

    sa.event.listen(store.engine, "handle_error", record_deadlock)
    try:
        with (
            mock.patch.object(store, "_persist_skill_version", side_effect=coordinate_persistence),
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="postgres-skill-race") as executor,
        ):
            single = executor.submit(register, "single")
            try:
                assert allocated.wait(timeout=10)
                bulk = executor.submit(register, "bulk")
                assert bulk_locked.wait(timeout=10)
                resume_single.set()
                _wait_for_postgres_blocker(store.engine, pids["single"], pids["bulk"])
            finally:
                resume_single.set()
                resume_bulk.set()
            bulk_result = bulk.result(timeout=20)
            single_error = single.exception(timeout=20)
    finally:
        sa.event.remove(store.engine, "handle_error", record_deadlock)

    assert deadlocks == [victim]
    assert bulk_result.version == 1
    assert store.get_skill_version(definition["name"], 1) == bulk_result
    expected_versions = 2 if victim == "single" and single_error is None else 1
    with store.ManagedSessionMaker() as session:
        assert (
            store._get_query(session, SqlSkillVersion).filter_by(name=definition["name"]).count()
            == expected_versions
        )
    if single_error is not None:
        assert isinstance(single_error, MlflowException)
        assert single_error.error_code == ErrorCode.Name(TEMPORARILY_UNAVAILABLE)
        assert single_error.__cause__.orig.pgcode == "40P01"
        raise single_error
    single_result = single.result()
    assert single_result.version == expected_versions
    assert single_result.created_by == "single"
    assert store.bulk_register_skills([definition])[0] == single_result
