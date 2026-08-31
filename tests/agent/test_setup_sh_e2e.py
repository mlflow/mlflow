import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from mlflow import MlflowClient

from tests.agent.setup_sh_test_utils import run_interactive
from tests.helper_functions import get_safe_port
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server

SETUP_SCRIPT = Path(__file__).parents[2] / "mlflow" / "agent" / "setup" / "setup.sh"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create an isolated Git repository in which the setup wizard can run.

    Args:
        tmp_path: Pytest-managed temporary directory for this test.

    Returns:
        Path to the initialized Git repository.
    """
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=project, check=True)
    return project


@pytest.fixture
def setup_env(tmp_path: Path) -> dict[str, str]:
    """Build an isolated environment with a fake Codex binary that records its prompt.

    Args:
        tmp_path: Pytest-managed temporary directory for the fake binary, home, and cache.

    Returns:
        Environment variables for invoking the setup wizard without user-level state.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    prompt_path = tmp_path / "agent-prompt.txt"
    (fake_bin / "codex").write_text('#!/bin/sh\nprintf \'%s\' "$1" > "$MLFLOW_TEST_PROMPT_PATH"\n')
    (fake_bin / "codex").chmod(0o755)

    env = os.environ.copy()
    for name in (
        "DATABRICKS_CONFIG_PROFILE",
        "DATABRICKS_HOST",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_TOKEN",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
    ):
        env.pop(name, None)
    env.update({
        "HOME": str(tmp_path / "home"),
        "MLFLOW_TEST_PROMPT_PATH": str(prompt_path),
        "NO_COLOR": "1",
        "PATH": os.pathsep.join([str(fake_bin), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
        "TERM": "dumb",
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    })
    return env


@pytest.fixture(scope="module")
def mlflow_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Run a real, unauthenticated MLflow server for remote-setup tests.

    Args:
        tmp_path_factory: Pytest factory used to isolate the server database and artifacts.

    Yields:
        Base URL of the running MLflow server.
    """
    tmp_path = tmp_path_factory.mktemp("setup-sh-mlflow-server")
    port = get_safe_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host=127.0.0.1",
            f"--port={port}",
            f"--backend-store-uri=sqlite:///{tmp_path / 'mlflow.db'}",
            f"--default-artifact-root={(tmp_path / 'artifacts').as_uri()}",
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        for _ in range(60):
            if process.poll() is not None:
                pytest.fail(f"MLflow server exited with status {process.returncode}")
            try:
                if requests.get(f"{url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.25)
        else:
            pytest.fail("MLflow server did not become ready")
        yield url
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


@pytest.fixture
def basic_auth_mlflow_server(tmp_path: Path) -> Iterator[str]:
    """Run MLflow's basic-auth app with isolated tracking and authentication databases.

    Args:
        tmp_path: Pytest-managed directory for the server databases, configuration, and artifacts.

    Yields:
        Base URL of the running authenticated MLflow server.
    """
    auth_config_path = write_isolated_auth_config(tmp_path)
    with _init_server(
        backend_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
        root_artifact_uri=(tmp_path / "artifacts").as_uri(),
        extra_env={
            "MLFLOW_AUTH_CONFIG_PATH": str(auth_config_path),
            "MLFLOW_FLASK_SERVER_SECRET_KEY": "test-secret-key",
        },
        app="mlflow.server.auth:create_app",
        server_type="flask",
    ) as url:
        yield url


def _run_setup(cwd: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    """Run a non-interactive setup invocation and capture its output for assertions.

    Args:
        cwd: Working directory in which to run the setup script.
        env: Complete environment passed to the setup process.
        args: Command-line arguments forwarded to the setup script.

    Returns:
        Completed process containing the exit status and captured text streams.
    """
    return subprocess.run(
        [str(SETUP_SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.timeout(60)
def test_interactive_remote_setup_creates_experiment_and_launches_agent(
    project: Path, setup_env: dict[str, str], mlflow_server: str
):
    exit_code, output = run_interactive(
        [str(SETUP_SCRIPT)],
        project,
        setup_env,
        [
            ("Where should MLflow store traces?", b"\x1b[B\r"),
            ("MLflow tracking server URL", f"{mlflow_server}\r".encode()),
            ("Choose an experiment", b"\r"),
            ("Experiment name", b"setup-sh-e2e\r"),
            ("Choose a coding agent", b"\r"),
        ],
    )

    assert exit_code == 0, output
    experiment = MlflowClient(tracking_uri=mlflow_server).get_experiment_by_name("setup-sh-e2e")
    assert experiment is not None
    assert experiment.tags["mlflow.experimentKind"] == "genai_development"

    prompt = Path(setup_env["MLFLOW_TEST_PROMPT_PATH"]).read_text()
    assert f"- Tracking URI: {mlflow_server}" in prompt
    assert f"- Experiment ID: {experiment.experiment_id}" in prompt


@pytest.mark.timeout(30)
def test_existing_experiment_is_reused(
    project: Path, setup_env: dict[str, str], mlflow_server: str
):
    client = MlflowClient(tracking_uri=mlflow_server)
    experiment_id = client.create_experiment("existing-setup-sh-e2e")

    result = _run_setup(
        project,
        setup_env,
        "--tracking-uri",
        mlflow_server,
        "--experiment-name",
        "existing-setup-sh-e2e",
        "--agent",
        "codex",
    )

    assert result.returncode == 0, result.stderr
    assert "Experiment created" not in result.stderr
    prompt = Path(setup_env["MLFLOW_TEST_PROMPT_PATH"]).read_text()
    assert f"- Experiment ID: {experiment_id}" in prompt


@pytest.mark.timeout(30)
def test_basic_authentication_against_mlflow_server(
    project: Path,
    setup_env: dict[str, str],
    basic_auth_mlflow_server: str,
):
    response = requests.post(
        f"{basic_auth_mlflow_server}/api/2.0/mlflow/experiments/create",
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
        json={"name": "authenticated"},
        timeout=5,
    )
    response.raise_for_status()
    experiment_id = response.json()["experiment_id"]

    exit_code, output = run_interactive(
        [
            str(SETUP_SCRIPT),
            "--tracking-uri",
            basic_auth_mlflow_server,
            "--experiment-id",
            experiment_id,
            "--agent",
            "codex",
        ],
        project,
        setup_env,
        [
            ("Authentication required", b"\x1b[B\r"),
            ("MLflow username", f"{ADMIN_USERNAME}\r".encode()),
            ("MLflow password", f"{ADMIN_PASSWORD}\r".encode()),
        ],
    )

    assert exit_code == 0, output
    assert ADMIN_PASSWORD not in output
    prompt = Path(setup_env["MLFLOW_TEST_PROMPT_PATH"]).read_text()
    assert f"- Experiment ID: {experiment_id}" in prompt
    assert "- Experiment name: authenticated" in prompt


@pytest.mark.timeout(30)
def test_missing_git_repository_can_continue(
    tmp_path: Path, setup_env: dict[str, str], mlflow_server: str
):
    non_git_project = tmp_path / "not-a-repository"
    non_git_project.mkdir()
    exit_code, output = run_interactive(
        [
            str(SETUP_SCRIPT),
            "--tracking-uri",
            mlflow_server,
            "--experiment-name",
            "missing-git-setup-sh-e2e",
            "--agent",
            "codex",
        ],
        non_git_project,
        setup_env,
        [("Continue with setup?", b"\x1b[B\r")],
    )

    assert exit_code == 0, output
    assert "Git repository not found" in output
    prompt = Path(setup_env["MLFLOW_TEST_PROMPT_PATH"]).read_text()
    assert "missing-git-setup-sh-e2e" in prompt


@pytest.mark.timeout(30)
def test_cli_arguments_skip_interactive_prompts(
    project: Path, setup_env: dict[str, str], mlflow_server: str
):
    client = MlflowClient(tracking_uri=mlflow_server)
    experiment_id = client.create_experiment("cli-setup-sh-e2e")

    result = _run_setup(
        project,
        setup_env,
        "--tracking-uri",
        f"{mlflow_server}/?ignored=true",
        "--experiment-id",
        experiment_id,
        "--agent",
        "codex",
    )

    assert result.returncode == 0, result.stderr
    assert "Where should MLflow store traces?" not in result.stderr
    assert "Choose an experiment" not in result.stderr
    assert "Choose a coding agent" not in result.stderr
    prompt = Path(setup_env["MLFLOW_TEST_PROMPT_PATH"]).read_text()
    assert f"- Tracking URI: {mlflow_server}" in prompt
    assert f"- Experiment ID: {experiment_id}" in prompt


@pytest.mark.timeout(30)
def test_manual_setup_prints_instructions(
    project: Path, setup_env: dict[str, str], mlflow_server: str
):
    exit_code, output = run_interactive(
        [
            str(SETUP_SCRIPT),
            "--tracking-uri",
            mlflow_server,
            "--experiment-name",
            "manual-setup-sh-e2e",
        ],
        project,
        setup_env,
        [("Choose a coding agent", b"\x1b[B\r")],
    )

    assert exit_code == 0, output
    assert "Continue manually" in output
    assert f"Set MLFLOW_TRACKING_URI={mlflow_server}" in output
    assert "MLflow Tracing quickstart:" in output
    assert "Setup complete" in output
    assert not Path(setup_env["MLFLOW_TEST_PROMPT_PATH"]).exists()


@pytest.mark.timeout(30)
def test_dirty_repository_setup_can_be_cancelled(project: Path, setup_env: dict[str, str]):
    tracked_file = project / "tracked.txt"
    tracked_file.write_text("before\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=project, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MLflow Tests",
            "-c",
            "user.email=tests@mlflow.org",
            "commit",
            "--quiet",
            "-m",
            "Initial commit",
        ],
        cwd=project,
        check=True,
    )
    tracked_file.write_text("after\n")

    exit_code, output = run_interactive(
        [str(SETUP_SCRIPT)], project, setup_env, [("Continue with setup?", b"\r")]
    )

    assert exit_code == 1
    assert "Git changes detected" in output
    assert "Setup cancelled. Commit or stash the changes, then rerun setup." in output
