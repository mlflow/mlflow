import importlib
import os
import platform
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path
from subprocess import Popen
from uuid import uuid4

import pytest
from sqlalchemy import text

from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext

_podman_socket_proc = None


def _import_container_class(module: str, class_name: str):
    pytest.importorskip("testcontainers", reason="testcontainers is required")
    for module_name in (f"testcontainers.community.{module}", f"testcontainers.{module}"):
        try:
            container_module = importlib.import_module(module_name)
        except ImportError:
            continue
        return getattr(container_module, class_name)
    pytest.skip(f"{class_name} is unavailable in this testcontainers installation")


def _command_succeeds(command: list[str]) -> bool:
    try:
        return (
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="module", autouse=True)
def _configure_testcontainers() -> None:
    """
    Configure testcontainers to use Docker if available, or a Podman socket otherwise.

    For Linux Podman setups without a machine-managed socket, this fixture starts
    ``podman system service --time=0`` for the duration of the module.
    """
    global _podman_socket_proc

    docker = shutil.which("docker")
    if docker is not None and _command_succeeds([docker, "info"]):
        yield
        return

    podman = shutil.which("podman")
    if podman is None:
        pytest.skip("Docker or Podman is required")

    result = subprocess.run(
        [podman, "machine", "inspect", "--format", "{{.ConnectionInfo.PodmanSocket.Path}}"],
        capture_output=True,
        text=True,
        check=False,
    )

    podman_socket = result.stdout.strip() if result.returncode == 0 else None
    started_podman_service = False

    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not podman_socket and xdg_runtime_dir is not None:
        podman_socket = f"{xdg_runtime_dir}/podman/podman.sock"
        _podman_socket_proc = Popen([podman, "system", "service", "--time=0"])
        started_podman_service = True

        for _ in range(5):
            time.sleep(0.5)
            if Path(podman_socket).exists():
                break

    if not podman_socket or not Path(podman_socket).exists():
        pytest.skip("Podman socket is unavailable")

    env_docker_host = "DOCKER_HOST"
    env_testcontainers_ryuk_disabled = "TESTCONTAINERS_RYUK_DISABLED"

    docker_host = os.environ.get(env_docker_host)
    ryuk_disabled = os.environ.get(env_testcontainers_ryuk_disabled)

    if docker_host is None:
        os.environ[env_docker_host] = f"unix://{podman_socket}"
    if ryuk_disabled is None:
        os.environ[env_testcontainers_ryuk_disabled] = "true"

    try:
        yield
    finally:
        if docker_host is None:
            os.environ.pop(env_docker_host, None)
        if ryuk_disabled is None:
            os.environ.pop(env_testcontainers_ryuk_disabled, None)

        if started_podman_service and _podman_socket_proc is not None:
            _podman_socket_proc.terminate()
            _podman_socket_proc.wait(20)
            _podman_socket_proc = None


@pytest.fixture(
    scope="module",
    params=[
        "mysql",
        pytest.param(
            "mssql",
            marks=pytest.mark.skipif(
                platform.machine().lower() in {"aarch64", "arm64"},
                reason="SQL Server container image is not supported on ARM",
            ),
        ),
        "postgres",
    ],
)
def database_uri(request):
    if request.param == "mysql":
        mysql_container = _import_container_class("mysql", "MySqlContainer")
        database = mysql_container("mysql:8.0", dialect="pymysql").with_command(
            "--character-set-server=utf8mb4 --collation-server=utf8mb4_0900_ai_ci "
            "--log-bin-trust-function-creators=1"
        )
    elif request.param == "mssql":
        sql_server_container = _import_container_class("mssql", "SqlServerContainer")
        database = sql_server_container(
            "mcr.microsoft.com/mssql/server:2019-latest", platform="linux/amd64"
        ).with_env("MSSQL_COLLATION", "SQL_Latin1_General_CP1_CI_AS")
    else:
        postgres_container = _import_container_class("postgres", "PostgresContainer")
        database = postgres_container("postgres:16")

    with database:
        yield database.get_connection_url()


@pytest.fixture(params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def store(database_uri, request, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(request.param).lower())
    store_cls = WorkspaceAwareSqlAlchemyStore if request.param else SqlAlchemyStore
    with WorkspaceContext("default"):
        instance = store_cls(database_uri, tmp_path.as_uri())
        try:
            # Verify the database's default comparison behavior without assuming
            # the fixed tag key column remains case-insensitive.
            with instance.engine.connect() as connection:
                equal = connection.execute(
                    text("SELECT CASE WHEN 'Team' = 'team' THEN 1 ELSE 0 END")
                ).scalar_one()
            expected = 0 if instance.engine.dialect.name == "postgresql" else 1
            assert equal == expected, "Unexpected database collation"
            yield instance
        finally:
            instance.engine.dispose()


@pytest.mark.parametrize("target", ["parent", "version"])
def test_skill_tag_keys_preserve_case(store, target):
    name = f"tag-case-{uuid4().hex}"
    version = store.create_skill_version(name).version

    if target == "parent":
        set_tag = partial(store.set_skill_tag, name)
        delete_tag = partial(store.delete_skill_tag, name)
        get_entity = partial(store.get_skill, name)
        search = store.search_skills
    else:
        set_tag = partial(store.set_skill_version_tag, name, version)
        delete_tag = partial(store.delete_skill_version_tag, name, version)
        get_entity = partial(store.get_skill_version, name, version)
        search = partial(store.search_skill_versions, name)

    set_tag("Team", "original")
    set_tag("team", "new")

    # Capture every symptom before asserting, so one run shows overwrite, search,
    # and deletion behavior even when the first check would fail.
    observed = {
        "tags_after_set": get_entity().tags,
        "lowercase_key_found": any(
            entity.name == name for entity in search(filter_string="tags.team = 'new'")
        ),
        "uppercase_key_found": any(
            entity.name == name for entity in search(filter_string="tags.Team = 'original'")
        ),
    }
    delete_tag("team")
    observed["tags_after_delete"] = get_entity().tags

    assert observed == {
        "tags_after_set": {"Team": "original", "team": "new"},
        "lowercase_key_found": True,
        "uppercase_key_found": True,
        "tags_after_delete": {"Team": "original"},
    }
