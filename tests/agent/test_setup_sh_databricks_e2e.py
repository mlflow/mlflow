import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

SETUP_SCRIPT = Path(__file__).parents[2] / "mlflow" / "agent" / "setup" / "setup.sh"

_FAKE_DATABRICKS_CLI = r"""#!/bin/sh
set -eu

printf '%s\n' "$*" >> "$DATABRICKS_TEST_CALLS"

case "$1 ${2:-}" in
"version ")
    printf '%s\n' 'Databricks CLI v1.14.1'
    ;;
"auth profiles")
    printf '%s\n' 'Name Host Valid' 'DEFAULT https://workspace.example.com YES'
    ;;
"auth token")
    printf '%s\n' '{}'
    ;;
"workspace list")
    printf '%s\n' '[]'
    ;;
"experiments get-experiment")
    if [ "$DATABRICKS_TEST_SCENARIO" = "uc" ]; then
        printf '%s\n' \
            '{"experiment":{' \
            '"experiment_id":"existing-id",' \
            '"name":"/Users/test/existing",' \
            '"tags":[{' \
            '"key":"mlflow.experiment.databricksTraceDestinationPath",' \
            '"value":"catalog.schema.custom-prefix"' \
            '}]}}'
    else
        printf '%s\n' \
            '{"experiment":{' \
            '"experiment_id":"existing-id",' \
            '"name":"/Users/test/existing",' \
            '"tags":[]}}'
    fi
    ;;
"experiments get-by-name")
    exit 1
    ;;
"experiments create-experiment")
    printf '%s\n' '{"experiment_id":"new-id"}'
    ;;
"experiments set-experiment-tag")
    printf '%s\n' '{}'
    ;;
"api post")
    case "$3" in
    /api/5.0/mlflow/tracing/locations)
        printf '%s\n' \
            '{' \
            '"catalog_name":"test_catalog",' \
            '"schema_name":"test_schema",' \
            '"table_prefix":"new-id"' \
            '}'
        ;;
    /api/5.0/mlflow/experiments/*/trace-location:link)
        printf '%s\n' '{}'
        ;;
    *)
        printf 'Unexpected Databricks API path: %s\n' "$3" >&2
        exit 2
        ;;
    esac
    ;;
*)
    printf 'Unexpected Databricks CLI invocation: %s\n' "$*" >&2
    exit 2
    ;;
esac
"""


@dataclass
class DatabricksHarness:
    project: Path
    env: dict[str, str]
    calls_path: Path
    prompt_path: Path

    def run(self, scenario: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Run the setup script against the fake Databricks CLI.

        Args:
            scenario: Fake backend behavior to use for the invocation.
            args: Command-line arguments forwarded to the setup script.

        Returns:
            Completed process containing the exit status and captured text streams.
        """
        return subprocess.run(
            [str(SETUP_SCRIPT), *args],
            cwd=self.project,
            env=self.env | {"DATABRICKS_TEST_SCENARIO": scenario},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def calls(self) -> list[str]:
        """Return the Databricks CLI invocations recorded by the fake executable."""
        return self.calls_path.read_text().splitlines()

    def prompt(self) -> str:
        """Return the prompt delivered to the fake coding agent."""
        return self.prompt_path.read_text()


@pytest.fixture
def databricks_harness(tmp_path: Path) -> DatabricksHarness:
    """Create a clean project with fake Databricks and coding-agent executables.

    Args:
        tmp_path: Pytest-managed directory for the project, executables, and recorded calls.

    Returns:
        Harness for running the real setup script against the fake backend.
    """
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=project, check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    databricks = fake_bin / "databricks"
    databricks.write_text(_FAKE_DATABRICKS_CLI)
    databricks.chmod(0o755)

    prompt_path = tmp_path / "agent-prompt.txt"
    codex = fake_bin / "codex"
    codex.write_text('#!/bin/sh\nprintf \'%s\' "$1" > "$MLFLOW_TEST_PROMPT_PATH"\n')
    codex.chmod(0o755)

    calls_path = tmp_path / "databricks-calls.txt"
    calls_path.touch()
    env = os.environ.copy()
    for name in (
        "DATABRICKS_CONFIG_PROFILE",
        "DATABRICKS_HOST",
        "MLFLOW_TRACKING_URI",
    ):
        env.pop(name, None)
    env.update({
        "DATABRICKS_TEST_CALLS": str(calls_path),
        "HOME": str(tmp_path / "home"),
        "MLFLOW_TEST_PROMPT_PATH": str(prompt_path),
        "NO_COLOR": "1",
        "PATH": os.pathsep.join([str(fake_bin), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
        "TERM": "dumb",
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    })
    return DatabricksHarness(project, env, calls_path, prompt_path)


@pytest.mark.timeout(30)
def test_existing_uc_experiment_selects_warehouse_and_preserves_destination(
    databricks_harness: DatabricksHarness,
):
    result = databricks_harness.run(
        "uc",
        "--profile",
        "DEFAULT",
        "--experiment-id",
        "existing-id",
        "--warehouse-id",
        "warehouse-1",
        "--agent",
        "codex",
    )

    assert result.returncode == 0, result.stderr
    calls = databricks_harness.calls()
    assert not any("/api/5.0/mlflow/tracing/locations" in call for call in calls)
    assert not any("trace-location:link" in call for call in calls)
    prompt = databricks_harness.prompt()
    assert "- Unity Catalog trace destination: catalog.schema.custom-prefix" in prompt
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID=warehouse-1" in prompt
    assert 'table_prefix="custom-prefix"' in prompt


@pytest.mark.timeout(30)
def test_new_experiment_creates_and_links_uc_destination(
    databricks_harness: DatabricksHarness,
):
    result = databricks_harness.run(
        "new",
        "--profile",
        "DEFAULT",
        "--experiment-name",
        "/Users/test/new",
        "--warehouse-id",
        "warehouse-1",
        "--uc-schema",
        "test_catalog.test_schema",
        "--agent",
        "codex",
    )

    assert result.returncode == 0, result.stderr
    calls = databricks_harness.calls()
    assert any(call.startswith("experiments create-experiment /Users/test/new ") for call in calls)
    create_location_call = next(
        call for call in calls if "/api/5.0/mlflow/tracing/locations" in call
    )
    assert '"catalog_name":"test_catalog"' in create_location_call
    assert '"schema_name":"test_schema"' in create_location_call
    assert '"table_prefix":"new-id"' in create_location_call
    assert '"sql_warehouse_id":"warehouse-1"' in create_location_call
    link_call = next(call for call in calls if "trace-location:link" in call)
    assert '"experiment_id":"new-id"' in link_call
    prompt = databricks_harness.prompt()
    assert "- Experiment ID: new-id" in prompt
    assert "- Unity Catalog trace destination: test_catalog.test_schema.new-id" in prompt
    assert 'table_prefix="new-id"' in prompt


@pytest.mark.timeout(30)
def test_existing_workspace_experiment_skips_uc_configuration(
    databricks_harness: DatabricksHarness,
):
    result = databricks_harness.run(
        "legacy",
        "--profile",
        "DEFAULT",
        "--experiment-id",
        "existing-id",
        "--agent",
        "codex",
    )

    assert result.returncode == 0, result.stderr
    assert "Existing experiment uses workspace storage" in result.stderr
    calls = databricks_harness.calls()
    assert not any(call.startswith("warehouses list ") for call in calls)
    assert not any("/api/5.0/mlflow/tracing/locations" in call for call in calls)
    assert "UnityCatalog" not in databricks_harness.prompt()
