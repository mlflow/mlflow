import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.agent.setup_sh_test_utils import run_interactive

SETUP_SCRIPT = Path(__file__).parents[2] / "mlflow" / "agent" / "setup" / "setup.sh"


def _write_mock_databricks_cli(path: Path) -> None:
    """Create a response-driven stand-in for the external Databricks CLI process.

    The setup wizard runs in a child shell, so Python mocks cannot intercept its
    `databricks` commands. This executable reads command-prefix responses supplied by
    each test, records every invocation, and fails on commands the test did not declare.

    Args:
        path: Location where the executable should be written.
    """
    path.write_text(
        f"#!{sys.executable}\n"
        + r"""import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
calls_path = Path(os.environ["DATABRICKS_TEST_CALLS"])
with calls_path.open("a") as calls:
    calls.write(json.dumps(args) + "\n")

if "--host" in args and args[:2] != ["auth", "login"]:
    print(f"unknown flag: --host for {' '.join(args[:2])}", file=sys.stderr)
    raise SystemExit(2)

routes = json.loads(Path(os.environ["DATABRICKS_TEST_ROUTES"]).read_text())
state_path = Path(os.environ["DATABRICKS_TEST_STATE"])
state = json.loads(state_path.read_text()) if state_path.exists() else {}

for index, route in enumerate(routes):
    prefix = route["args"]
    if args[: len(prefix)] != prefix:
        continue

    responses = route.get("responses", [route])
    response_index = state.get(str(index), 0)
    state[str(index)] = response_index + 1
    state_path.write_text(json.dumps(state))
    response = responses[min(response_index, len(responses) - 1)]

    if stdout := response.get("stdout"):
        print(stdout)
    if stderr := response.get("stderr"):
        print(stderr, file=sys.stderr)
    raise SystemExit(response.get("returncode", 0))

print(f"Unexpected Databricks CLI invocation: {' '.join(args)}", file=sys.stderr)
raise SystemExit(2)
"""
    )
    path.chmod(0o755)


@dataclass
class DatabricksTestConfig:
    """Filesystem paths and environment for a Databricks setup test."""

    project: Path
    env: dict[str, str]
    calls_path: Path
    prompt_path: Path
    routes_path: Path


@pytest.fixture
def databricks_config(tmp_path: Path) -> DatabricksTestConfig:
    """Create an isolated project and external command configuration.

    Args:
        tmp_path: Pytest-managed directory for the project, commands, and recorded data.

    Returns:
        Paths and environment for running the setup script against declared CLI responses.
    """
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=project, check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_mock_databricks_cli(fake_bin / "databricks")

    prompt_path = tmp_path / "agent-prompt.txt"
    codex = fake_bin / "codex"
    codex.write_text('#!/bin/sh\nprintf \'%s\' "$1" > "$MLFLOW_TEST_PROMPT_PATH"\n')
    codex.chmod(0o755)

    calls_path = tmp_path / "databricks-calls.jsonl"
    calls_path.touch()
    routes_path = tmp_path / "databricks-routes.json"
    env = os.environ.copy()
    for name in (
        "DATABRICKS_CONFIG_PROFILE",
        "DATABRICKS_HOST",
        "MLFLOW_TRACKING_URI",
    ):
        env.pop(name, None)
    env.update({
        "DATABRICKS_TEST_CALLS": str(calls_path),
        "DATABRICKS_TEST_ROUTES": str(routes_path),
        "DATABRICKS_TEST_STATE": str(tmp_path / "databricks-state.json"),
        "HOME": str(tmp_path / "home"),
        "MLFLOW_TEST_PROMPT_PATH": str(prompt_path),
        "NO_COLOR": "1",
        "PATH": os.pathsep.join([str(fake_bin), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
        "TERM": "dumb",
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    })
    return DatabricksTestConfig(project, env, calls_path, prompt_path, routes_path)


def _base_routes(
    *,
    token_responses: list[dict[str, object]] | None = None,
    current_user: str | None = None,
):
    """Return responses for CLI setup, profile discovery, and authentication checks.

    Args:
        token_responses: Sequential responses for authentication checks. The final
            response is reused if the command is called additional times.
        current_user: User identity encoded in the mock OAuth token, if any.

    Returns:
        Route declarations consumed by the mock Databricks CLI.
    """
    if token_responses is None:
        token_stdout = (
            '{"access_token":"header.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIn0.signature"}'
            if current_user == "test@example.com"
            else "{}"
        )
        token_responses = [{"stdout": token_stdout}]

    return [
        {
            "args": ["auth", "profiles"],
            "stdout": "Name Host Valid\nDEFAULT https://workspace.example.com YES",
        },
        {
            "args": ["auth", "token"],
            "responses": token_responses,
        },
    ]


def _new_experiment_routes(
    *,
    warehouse_json: str,
    catalog_json: str,
    schema_json: str,
    catalog: str,
    schema: str,
):
    """Return responses used to create an experiment and its UC trace destination.

    Args:
        warehouse_json: Response from listing SQL warehouses.
        catalog_json: Response from listing Unity Catalog catalogs.
        schema_json: Response from listing schemas in the selected catalog.
        catalog: Catalog returned after creating the trace destination.
        schema: Schema returned after creating the trace destination.

    Returns:
        Route declarations consumed by the mock Databricks CLI.
    """
    return [
        {"args": ["experiments", "get-by-name"], "returncode": 1},
        {"args": ["experiments", "create-experiment"], "stdout": '{"experiment_id":"new-id"}'},
        {"args": ["experiments", "set-experiment-tag"], "stdout": "{}"},
        {"args": ["warehouses", "list"], "stdout": warehouse_json},
        {"args": ["catalogs", "list"], "stdout": catalog_json},
        {"args": ["schemas", "list"], "stdout": schema_json},
        {
            "args": ["api", "post", "/api/5.0/mlflow/tracing/locations"],
            "stdout": json.dumps({
                "catalog_name": catalog,
                "schema_name": schema,
                "table_prefix": "new-id",
            }),
        },
        {
            "args": [
                "api",
                "post",
                "/api/5.0/mlflow/experiments/new-id/trace-location:link",
            ],
            "stdout": "{}",
        },
    ]


def _set_routes(config: DatabricksTestConfig, routes: list[dict[str, object]]) -> None:
    """Write the CLI responses for one setup invocation.

    Args:
        config: Test configuration containing the route file path.
        routes: Ordered command-prefix routes and their responses.
    """
    config.routes_path.write_text(json.dumps(routes))


def _read_calls(config: DatabricksTestConfig) -> list[list[str]]:
    """Read Databricks CLI argument arrays recorded during setup.

    Args:
        config: Test configuration containing the call log path.

    Returns:
        One argument array for each Databricks CLI invocation.
    """
    return [json.loads(line) for line in config.calls_path.read_text().splitlines()]


def _run_setup(config: DatabricksTestConfig, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the setup script non-interactively against configured CLI responses.

    Args:
        config: Test configuration containing the project and process environment.
        args: Command-line arguments forwarded to the setup script.

    Returns:
        Completed setup process with captured output.
    """
    return subprocess.run(
        [str(SETUP_SCRIPT), *args],
        cwd=config.project,
        env=config.env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _experiment_json(*, trace_destination: str | None) -> str:
    """Build a get-experiment response with optional UC trace storage.

    Args:
        trace_destination: Existing `catalog.schema.table_prefix`, or None for workspace storage.

    Returns:
        Serialized Databricks get-experiment response.
    """
    tags = []
    if trace_destination:
        tags.append({
            "key": "mlflow.experiment.databricksTraceDestinationPath",
            "value": trace_destination,
        })
    return json.dumps({
        "experiment": {
            "experiment_id": "existing-id",
            "name": "/Users/test/existing",
            "tags": tags,
        }
    })


def _json_body(call: list[str]) -> dict[str, object]:
    """Parse the JSON value following a Databricks CLI `--json` flag.

    Args:
        call: Recorded Databricks CLI argument array.

    Returns:
        Parsed JSON request body.
    """
    return json.loads(call[call.index("--json") + 1])


@pytest.mark.parametrize(
    ("entered_path", "expected_path"),
    [
        pytest.param(b"\r", "/Users/test@example.com/project", id="default-path"),
        pytest.param(b"  /Users/test/new  \r", "/Users/test/new", id="custom-path"),
    ],
)
@pytest.mark.timeout(30)
def test_interactive_databricks_setup_creates_uc_experiment(
    databricks_config: DatabricksTestConfig,
    entered_path: bytes,
    expected_path: str,
):
    _set_routes(
        databricks_config,
        _base_routes(current_user="test@example.com")
        + _new_experiment_routes(
            warehouse_json=json.dumps({
                "warehouses": [{"id": "warehouse-1", "name": "Test Warehouse", "state": "RUNNING"}]
            }),
            catalog_json='{"catalogs":[{"name":"test_catalog"}]}',
            schema_json='{"schemas":[{"name":"test_schema"}]}',
            catalog="test_catalog",
            schema="test_schema",
        ),
    )

    exit_code, output = run_interactive(
        [str(SETUP_SCRIPT)],
        databricks_config.project,
        databricks_config.env,
        [
            ("Where should MLflow store traces?", b"\r"),
            ("DEFAULT    workspace.example.com", b"\r"),
            ("Create a new experiment", b"\r"),
            ("New experiment path", entered_path),
            ("Test Warehouse", b"\r"),
            ("test_catalog", b"\x1b[B\r"),
            ("test_schema", b"\x1b[B\r"),
            ("Choose a coding agent", b"\r"),
        ],
    )

    assert exit_code == 0, output
    assert "Type an experiment path, or press Enter to use the default." in output
    assert "Default: /Users/test@example.com/project" in output
    calls = _read_calls(databricks_config)
    create_experiment_call = next(
        call for call in calls if call[:2] == ["experiments", "create-experiment"]
    )
    assert create_experiment_call[2] == expected_path
    assert any(call[:2] == ["warehouses", "list"] for call in calls)
    assert any(call[:2] == ["catalogs", "list"] for call in calls)
    assert any(call[:3] == ["schemas", "list", "test_catalog"] for call in calls)
    create_call = next(
        call for call in calls if call[:3] == ["api", "post", "/api/5.0/mlflow/tracing/locations"]
    )
    assert _json_body(create_call) == {
        "uc_table_prefix": {
            "catalog_name": "test_catalog",
            "schema_name": "test_schema",
            "table_prefix": "new-id",
        },
        "sql_warehouse_id": "warehouse-1",
    }
    prompt = databricks_config.prompt_path.read_text()
    assert "- Experiment ID: new-id" in prompt
    assert "- Unity Catalog trace destination: test_catalog.test_schema.new-id" in prompt


@pytest.mark.timeout(30)
def test_interactive_databricks_setup_trims_existing_experiment_path(
    databricks_config: DatabricksTestConfig,
):
    _set_routes(
        databricks_config,
        _base_routes(current_user="test@example.com")
        + [
            {
                "args": ["experiments", "get-by-name"],
                "stdout": _experiment_json(trace_destination=None),
            }
        ],
    )

    exit_code, output = run_interactive(
        [str(SETUP_SCRIPT), "--profile", "DEFAULT", "--agent", "codex"],
        databricks_config.project,
        databricks_config.env,
        [
            ("Create a new experiment", b"\x1b[B\r"),
            ("Existing experiment path or ID", b"  /Users/test/existing  \r"),
        ],
    )

    assert exit_code == 0, output
    get_by_name_call = next(
        call
        for call in _read_calls(databricks_config)
        if call[:2] == ["experiments", "get-by-name"]
    )
    assert get_by_name_call[2] == "/Users/test/existing"


@pytest.mark.timeout(30)
def test_existing_uc_experiment_selects_warehouse_without_relinking(
    databricks_config: DatabricksTestConfig,
):
    _set_routes(
        databricks_config,
        _base_routes()
        + [
            {
                "args": ["experiments", "get-experiment"],
                "stdout": _experiment_json(trace_destination="catalog.schema.custom-prefix"),
            },
            {
                "args": ["warehouses", "list"],
                "stdout": json.dumps({
                    "warehouses": [
                        {"id": "warehouse-1", "name": "Test Warehouse", "state": "RUNNING"}
                    ]
                }),
            },
        ],
    )

    exit_code, output = run_interactive(
        [
            str(SETUP_SCRIPT),
            "--profile",
            "DEFAULT",
            "--experiment-id",
            "existing-id",
            "--agent",
            "codex",
        ],
        databricks_config.project,
        databricks_config.env,
        [("Test Warehouse", b"\r")],
    )

    assert exit_code == 0, output
    calls = _read_calls(databricks_config)
    assert not any(call[:2] == ["api", "post"] for call in calls)
    prompt = databricks_config.prompt_path.read_text()
    assert "- Unity Catalog trace destination: catalog.schema.custom-prefix" in prompt
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID=warehouse-1" in prompt
    assert 'table_prefix="custom-prefix"' in prompt


@pytest.mark.timeout(30)
def test_existing_workspace_experiment_skips_uc_configuration(
    databricks_config: DatabricksTestConfig,
):
    _set_routes(
        databricks_config,
        _base_routes()
        + [
            {
                "args": ["experiments", "get-experiment"],
                "stdout": _experiment_json(trace_destination=None),
            }
        ],
    )

    result = _run_setup(
        databricks_config,
        "--profile",
        "DEFAULT",
        "--experiment-id",
        "existing-id",
        "--agent",
        "codex",
    )

    assert result.returncode == 0, result.stderr
    assert "Existing experiment uses workspace storage" in result.stderr
    calls = _read_calls(databricks_config)
    assert not any(call[:2] == ["warehouses", "list"] for call in calls)
    assert not any(call[:2] == ["api", "post"] for call in calls)
    assert "UnityCatalog" not in databricks_config.prompt_path.read_text()


@pytest.mark.timeout(30)
def test_authentication_fallback_uses_host_and_profile_flags(
    databricks_config: DatabricksTestConfig,
):
    _set_routes(
        databricks_config,
        _base_routes(token_responses=[{"returncode": 1}, {"stdout": "{}"}])
        + [
            {"args": ["auth", "login"], "stdout": "{}"},
            {
                "args": ["experiments", "get-experiment"],
                "stdout": _experiment_json(trace_destination=None),
            },
        ],
    )

    exit_code, output = run_interactive(
        [
            str(SETUP_SCRIPT),
            "--profile",
            "DEFAULT",
            "--experiment-id",
            "existing-id",
            "--agent",
            "codex",
        ],
        databricks_config.project,
        databricks_config.env,
        [],
    )

    assert exit_code == 0, output
    assert [
        "auth",
        "login",
        "--host",
        "https://workspace.example.com",
        "--profile",
        "DEFAULT",
    ] in _read_calls(databricks_config)


@pytest.mark.timeout(30)
def test_manual_warehouse_and_uc_schema_entry(databricks_config: DatabricksTestConfig):
    _set_routes(
        databricks_config,
        _base_routes()
        + _new_experiment_routes(
            warehouse_json='{"warehouses":[]}',
            catalog_json='{"catalogs":[]}',
            schema_json='{"schemas":[]}',
            catalog="manual_catalog",
            schema="manual_schema",
        ),
    )

    exit_code, output = run_interactive(
        [
            str(SETUP_SCRIPT),
            "--profile",
            "DEFAULT",
            "--experiment-name",
            "/Users/test/manual",
            "--agent",
            "codex",
        ],
        databricks_config.project,
        databricks_config.env,
        [
            ("Enter a warehouse ID", b"manual-warehouse\r"),
            ("Enter catalog.schema", b"manual_catalog.manual_schema\r"),
        ],
    )

    assert exit_code == 0, output
    calls = _read_calls(databricks_config)
    create_call = next(
        call for call in calls if call[:3] == ["api", "post", "/api/5.0/mlflow/tracing/locations"]
    )
    assert _json_body(create_call) == {
        "uc_table_prefix": {
            "catalog_name": "manual_catalog",
            "schema_name": "manual_schema",
            "table_prefix": "new-id",
        },
        "sql_warehouse_id": "manual-warehouse",
    }
