import ast
import http.server
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import certifi
import charset_normalizer
import idna
import packaging
import pytest
import requests
import urllib3

import mlflow.telemetry.constant
import mlflow.telemetry.utils

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="setup.sh tests are flaky on Windows"
)

SETUP_SCRIPT = Path(__file__).parents[2] / "mlflow" / "agent" / "setup" / "setup.sh"


def run_shell(body: str, *args: str) -> subprocess.CompletedProcess[str]:
    command = f"""
MLFLOW_SETUP_SKIP_MAIN=1
export MLFLOW_SETUP_SKIP_MAIN
. "$1"
shift
{body}
"""
    return subprocess.run(
        ["sh", "-c", command, "sh", str(SETUP_SCRIPT), *args],
        env=os.environ.copy() | {"NO_COLOR": "1", "TERM": "dumb"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_normalize_workspace_url():
    result = run_shell('normalize_workspace_url "$1"', "workspace.example.com/some/path/")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "https://workspace.example.com"


def test_normalize_tracking_uri():
    result = run_shell(
        'normalize_tracking_uri "$1"', "http://localhost:5000/some/path/?query=value"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "http://localhost:5000/some/path"


def test_trim_whitespace():
    result = run_shell('trim_whitespace "$1"', "  /Users/test/experiment  ")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "/Users/test/experiment"


def test_json_escape():
    result = run_shell(
        r"""
value='a\path"with-quote'
json_escape "$value"
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == 'a\\\\path\\"with-quote'


def test_parse_args():
    result = run_shell(
        """
parse_args "$@"
printf '%s\n' "$TRACKING_URI" "$EXPERIMENT_NAME" "$AGENT_NAME"
""",
        "--tracking-uri",
        "mlflow.example.com/",
        "--experiment-name",
        "tracing-test",
        "--agent",
        "codex",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "mlflow.example.com/",
        "tracing-test",
        "codex",
    ]


@pytest.mark.parametrize(
    ("installed", "requested", "expected"),
    [
        (("claude", "codex", "opencode"), "", "claude"),
        (("codex", "opencode"), "", "codex"),
        (("opencode",), "", "opencode"),
        ((), "", "manual"),
        (("claude", "codex", "opencode"), "codex", "codex"),
        (("claude", "codex", "opencode"), "opencode", "opencode"),
        (("claude", "codex", "opencode"), "manual", "manual"),
        ((), "manual", "manual"),
    ],
)
def test_choose_agent_without_prompting(tmp_path: Path, installed, requested, expected):
    for name in installed:
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\nexit 99\n")
        executable.chmod(0o755)

    result = run_shell(
        """
# Use the POSIX form so a Windows drive colon (C:) does not split PATH.
PATH=$(cd "$1" && pwd)
shift
parse_args "$@"
validate_agent_name
select_option() { exit 98; }
show_manual_setup() { printf 'manual\\n'; }
choose_agent
printf '%s\\n' "$agent_choice"
""",
        str(tmp_path),
        *(["--agent", requested] if requested else []),
    )

    assert result.returncode == 0
    assert result.stdout.strip() == expected
    assert "Choose a coding agent" not in result.stderr


@pytest.mark.parametrize("agent", ["claude", "codex", "opencode"])
def test_choose_agent_rejects_missing_explicit_agent(tmp_path: Path, agent: str):
    result = run_shell(
        """
# Use the POSIX form so a Windows drive colon (C:) does not split PATH.
PATH=$(cd "$1" && pwd)
shift
parse_args "$@"
validate_agent_name
choose_agent
""",
        str(tmp_path),
        "--agent",
        agent,
    )

    assert result.returncode != 0
    assert f"Coding agent '{agent}' is not installed." in result.stderr


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, False),
        ("", False),
        ("# [DEFAULT]\n# host = https://workspace.example.com\n", False),
        ("[DEFAULT]\ntoken = token\n", False),
        ("host = https://workspace.example.com\n", False),
        ("[DEFAULT]\nhost = \n", False),
        ("[DEFAULT]\nhost = # workspace URL\n", False),
        ("[DEFAULT]\nhost = ; workspace URL\n", False),
        ("[DEFAULT]\nhost = https://workspace.example.com\n", True),
        ("[team-profile]\n  host = https://workspace.example.com\n", True),
        ("[first]\ntoken = token\n[second]\nhost = https://workspace.example.com\n", True),
    ],
)
def test_has_databricks_profile(tmp_path: Path, config, expected: bool):
    config_path = tmp_path / "databrickscfg"
    if config is not None:
        config_path.write_text(config)

    result = run_shell(
        """
DATABRICKS_CONFIG_FILE=$1
has_databricks_profile
""",
        str(config_path),
    )

    assert result.returncode == (0 if expected else 1)


@pytest.mark.parametrize(
    ("has_profile", "config_profile", "host"),
    [
        (True, "", ""),
        (False, "selected", ""),
        (False, "", "https://workspace.example.com"),
    ],
)
def test_choose_backend_uses_databricks_configuration(
    tmp_path: Path, has_profile: bool, config_profile: str, host: str
):
    config_path = tmp_path / "databrickscfg"
    if has_profile:
        config_path.write_text("[DEFAULT]\nhost = https://workspace.example.com\n")

    result = run_shell(
        """
DATABRICKS_CONFIG_FILE=$1
DATABRICKS_CONFIG_PROFILE=$2
DATABRICKS_HOST=$3
MLFLOW_TRACKING_URI=
select_option() { exit 98; }
choose_backend
printf '%s\\n' "$backend"
""",
        str(config_path),
        config_profile,
        host,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "databricks"


@pytest.mark.parametrize("source", ["flag", "environment"])
def test_choose_backend_tracking_uri_overrides_saved_profile(tmp_path: Path, source: str):
    config_path = tmp_path / "databrickscfg"
    config_path.write_text("[DEFAULT]\nhost = https://workspace.example.com\n")
    tracking_uri = "https://mlflow.example.com"

    result = run_shell(
        """
DATABRICKS_CONFIG_FILE=$1
DATABRICKS_CONFIG_PROFILE=
DATABRICKS_HOST=
MLFLOW_TRACKING_URI=
if [ "$2" = flag ]; then
    parse_args --tracking-uri "$3"
else
    MLFLOW_TRACKING_URI=$3
fi
select_option() { exit 98; }
choose_backend
printf '%s\\n' "$backend" "$TRACKING_URI"
""",
        str(config_path),
        source,
        tracking_uri,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["remote", tracking_uri]


@pytest.mark.parametrize(
    ("choice", "expected"), [("0", "databricks"), ("1", "remote"), ("2", "local")]
)
def test_choose_backend_prompts_without_configuration(tmp_path: Path, choice: str, expected: str):
    result = run_shell(
        """
DATABRICKS_CONFIG_FILE=$1
DATABRICKS_CONFIG_PROFILE=
DATABRICKS_HOST=
MLFLOW_TRACKING_URI=
backend_choice=$2
select_option() {
    printf '%s\\n' "$@"
    selected_index=$backend_choice
}
choose_backend
printf '%s\\n' "$backend"
""",
        str(tmp_path / "missing-config"),
        choice,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "Where should MLflow store traces?",
        "Databricks",
        "Existing OSS MLflow server",
        "New local MLflow server",
        expected,
    ]


def test_build_remote_agent_prompt():
    result = run_shell(
        """
backend=remote
TRACKING_URI=http://localhost:5050
EXPERIMENT_ID=42
EXPERIMENT_NAME=tracing-test
build_agent_prompt
"""
    )

    assert result.returncode == 0, result.stderr
    assert "- Tracking URI: http://localhost:5050" in result.stdout
    assert "- Experiment ID: 42" in result.stdout
    assert "- Experiment name: tracing-test" in result.stdout


def test_build_agent_prompt_requires_repo_summary_and_realistic_trace_input():
    result = run_shell(
        """
backend=remote
TRACKING_URI=http://localhost:5050
EXPERIMENT_ID=42
EXPERIMENT_NAME=tracing-test
build_agent_prompt
"""
    )

    assert result.returncode == 0, result.stderr
    checklist = [
        "## 1. Understand the application",
        "## 2. Document the application",
        "## 3. Configure MLflow Tracing",
        "## 4. Instrument the application",
        "## 5. Verify tracing with a realistic input",
        "## 6. Validate trace quality",
        "## 7. Report results",
    ]
    assert [result.stdout.index(item) for item in checklist] == sorted(
        result.stdout.index(item) for item in checklist
    )
    assert "- Summarize what you learned in the MLflow experiment description" in result.stdout
    assert "MlflowClient.set_experiment_tag" in result.stdout
    assert "mlflow.note.content" in result.stdout
    assert "interfaces to one application, not as separate instrumentation targets" in result.stdout
    assert "Do not ask the user to choose among shared wrappers" in result.stdout
    assert "Ask which application to instrument only when" in result.stdout
    assert "multiple independent agents or applications" in result.stdout
    assert "- Derive a realistic input from the README" in result.stdout
    assert "Do not use a placeholder prompt whose only purpose is to produce" in result.stdout
    assert "Prefer MLflow autologging" in result.stdout
    assert "mlflow.update_current_trace(session_id=...)" in result.stdout
    assert "request_preview and response_preview" in result.stdout
    assert "deployment environment (dev, staging, or prod)" in result.stdout
    assert "confirm input and output token counts, along with cost, are present" in result.stdout
    assert "never fabricate values" in result.stdout
    assert "Trace URL opens" not in result.stdout
    assert "exercise one traced operation" not in result.stdout


def test_build_remote_agent_prompt_includes_workspace():
    result = run_shell(
        """
backend=remote
TRACKING_URI=https://mlflow.example.com
EXPERIMENT_ID=42
EXPERIMENT_NAME=tracing-test
MLFLOW_WORKSPACE=workspace-a
build_agent_prompt
"""
    )

    assert result.returncode == 0, result.stderr
    assert "MLFLOW_WORKSPACE=workspace-a" in result.stdout


def test_build_databricks_uc_agent_prompt():
    result = run_shell(
        """
backend=databricks
TRACKING_URI=databricks://DEFAULT
EXPERIMENT_ID=42
EXPERIMENT_NAME=/Users/test@example.com/tracing-test
UC_SCHEMA=catalog.schema
trace_destination=catalog.schema.custom-prefix
WAREHOUSE_ID=warehouse-id
build_agent_prompt
"""
    )

    assert result.returncode == 0, result.stderr
    assert "- Unity Catalog trace destination: catalog.schema.custom-prefix" in result.stdout
    assert 'experiment_id="42"' in result.stdout
    assert 'catalog_name="catalog"' in result.stdout
    assert 'schema_name="schema"' in result.stdout
    assert 'table_prefix="custom-prefix"' in result.stdout
    assert "Do not replace it with MlflowExperimentLocation" in result.stdout


def test_build_host_only_databricks_prompt_includes_host():
    result = run_shell(
        """
backend=databricks
PROFILE=
WORKSPACE_URL=https://workspace.example.com
TRACKING_URI=databricks
EXPERIMENT_ID=42
EXPERIMENT_NAME=/Users/test/tracing-test
trace_destination=
WAREHOUSE_ID=
build_agent_prompt
"""
    )

    assert result.returncode == 0, result.stderr
    assert "DATABRICKS_HOST=https://workspace.example.com" in result.stdout
    assert "Discover available SQL warehouses" not in result.stdout


def test_build_databricks_prompt_defers_uc_storage_until_warehouse_is_selected():
    result = run_shell(
        """
backend=databricks
PROFILE=selected
TRACKING_URI=databricks://selected
EXPERIMENT_ID=42
EXPERIMENT_NAME=/Users/test/tracing-test
UC_SCHEMA=catalog.schema
trace_destination=
uc_trace_storage_pending=true
build_agent_prompt
"""
    )

    assert result.returncode == 0, result.stderr
    assert "- Unity Catalog destination to configure: catalog.schema.42" in result.stdout
    assert "- Unity Catalog trace destination:" not in result.stdout
    assert "Discover available SQL warehouses" in result.stdout
    assert "same authentication profile/host as the tracking URI" in result.stdout
    assert "preferring a running warehouse" in result.stdout
    assert "Set MLFLOW_TRACING_SQL_WAREHOUSE_ID to its ID" in result.stdout
    assert "trace tables have not been created or linked yet" in result.stdout
    assert "Complete this before enabling tracing, including for non-Python" in result.stdout
    assert 'experiment_id="42"' in result.stdout
    assert 'catalog_name="catalog"' in result.stdout
    assert 'schema_name="schema"' in result.stdout
    assert 'table_prefix="42"' in result.stdout


def test_manual_databricks_setup_explains_pending_uc_storage():
    result = run_shell(
        """
backend=databricks
PROFILE=selected
WORKSPACE_URL=https://workspace.example.com
DATABRICKS_BIN=databricks
TRACKING_URI=databricks://selected
EXPERIMENT_ID=42
UC_SCHEMA=catalog.schema
trace_destination=
uc_trace_storage_pending=true
manual_setup_docs=https://mlflow.org/docs/latest/genai/tracing/quickstart/
show_manual_setup
"""
    )

    assert result.returncode == 0, result.stderr
    output = " ".join(result.stderr.replace("│", "").split())
    assert "Choose an available SQL warehouse" in output
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" in output
    assert "Configure Unity Catalog trace storage in catalog.schema with table prefix 42" in output
    assert 'mlflow.set_experiment(experiment_id="42"' in output
    assert 'catalog_name="catalog", schema_name="schema", table_prefix="42"' in output


def test_json_tag_value():
    result = run_shell(
        """
printf '%s\n' '{
  "tags": [{
    "key": "mlflow.experiment.databricksTraceDestinationPath",
    "value": "catalog.schema.prefix"
  }]
}' | json_tag_value mlflow.experiment.databricksTraceDestinationPath
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "catalog.schema.prefix\n"


@pytest.mark.parametrize("value", ["catalog.schema.extra", ".schema", "catalog."])
def test_validate_uc_schema_rejects_invalid_values(value: str):
    result = run_shell('validate_uc_schema "$1"', value)

    assert result.returncode != 0


def test_oss_curl_uses_protected_config_and_standard_options(tmp_path: Path):
    args_path = tmp_path / "curl-args"
    result = run_shell(
        """
curl_args_file=$1
curl() {
    printf '<%s>\n' "$@" > "$curl_args_file"
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--config" ]; then
            cat "$2"
            break
        fi
        shift
    done
}
TRACKING_URI=https://mlflow.example
MLFLOW_WORKSPACE=workspace-a
MLFLOW_TRACKING_USERNAME=user
MLFLOW_TRACKING_PASSWORD=password
MLFLOW_TRACKING_TOKEN=token
MLFLOW_TRACKING_SERVER_CERT_PATH=/tmp/server.pem
MLFLOW_TRACKING_CLIENT_CERT_PATH=/tmp/client.pem
export MLFLOW_WORKSPACE MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD MLFLOW_TRACKING_TOKEN
oss_curl https://mlflow.example
""",
        str(args_path),
    )

    assert result.returncode == 0, result.stderr
    args = args_path.read_text()
    assert "<X-MLFLOW-WORKSPACE: workspace-a>" in args
    assert "<--connect-timeout>\n<10>" in args
    assert "<--max-time>\n<60>" in args
    assert "<--cacert>\n</tmp/server.pem>" in args
    assert "<--cert>\n</tmp/client.pem>" in args
    assert "user:password" not in args
    assert 'user = "user:password"' in result.stdout
    assert "Bearer token" not in result.stdout


def test_authenticated_remote_http_is_rejected():
    result = run_shell(
        """
TRACKING_URI=http://mlflow.example.com
MLFLOW_TRACKING_TOKEN=secret
curl() { return 98; }
oss_curl https://mlflow.example
"""
    )

    assert result.returncode != 0
    assert "Authentication requires HTTPS" in result.stderr


def test_oss_curl_honors_insecure_tls():
    result = run_shell(
        """
curl() { printf '<%s>\n' "$@"; }
MLFLOW_TRACKING_INSECURE_TLS=true
oss_curl https://mlflow.example
"""
    )

    assert result.returncode == 0, result.stderr
    assert "<--insecure>" in result.stdout


def test_tracking_uri_rejects_embedded_credentials():
    result = run_shell(
        """
TRACKING_URI=https://user:password@mlflow.example.com
validate_tracking_uri
"""
    )

    assert result.returncode != 0
    assert "Do not include credentials" in result.stderr


def test_json_experiment_strings_fallback_handles_compact_response():
    result = run_shell(
        """
sed_bin=$(command -v sed)
PATH=
sed() { "$sed_bin" "$@"; }
printf '%s%s\n' \
    '{"experiments":[{"experiment_id":"1","name":"one"},' \
    '{"experiment_id":"2","name":"two"}]}' |
    json_experiment_strings name
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["one", "two"]


def test_explicit_workspace_url_ignores_ambient_profile():
    result = run_shell(
        """
WORKSPACE_URL=https://workspace-a.example.com
WORKSPACE_URL_EXPLICIT=true
PROFILE=
DATABRICKS_CONFIG_PROFILE=OTHER
run_with_spinner() { spinner_output='DEFAULT|https://workspace-b.example.com'; }
resolve_databricks_profile
printf '%s\n' "$PROFILE" "$WORKSPACE_URL"
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["", "https://workspace-a.example.com"]


def test_explicit_profile_ignores_ambient_workspace_url():
    result = run_shell(
        """
PROFILE=DEFAULT
PROFILE_EXPLICIT=true
WORKSPACE_URL=
DATABRICKS_HOST=https://workspace-b.example.com
run_with_spinner() { spinner_output='DEFAULT|https://workspace-a.example.com'; }
resolve_databricks_profile
printf '%s\n' "$PROFILE" "$WORKSPACE_URL"
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["DEFAULT", "https://workspace-a.example.com"]


def test_explicit_profile_and_workspace_url_must_match():
    result = run_shell(
        """
PROFILE=DEFAULT
PROFILE_EXPLICIT=true
WORKSPACE_URL=https://workspace-b.example.com
WORKSPACE_URL_EXPLICIT=true
run_with_spinner() { spinner_output='DEFAULT|https://workspace-a.example.com'; }
resolve_databricks_profile
"""
    )

    assert result.returncode != 0
    assert "points to https://workspace-a.example.com" in result.stderr


def test_dbx_json_passes_host_through_environment(tmp_path: Path):
    databricks = tmp_path / "databricks"
    databricks.write_text(
        r"""#!/bin/sh
printf '%s\n' "${DATABRICKS_HOST:-}|${DATABRICKS_CONFIG_PROFILE:-}|$*"
"""
    )
    databricks.chmod(0o755)

    result = run_shell(
        """
DATABRICKS_BIN=$1
PROFILE=
WORKSPACE_URL=https://workspace.example.com
DATABRICKS_CONFIG_PROFILE=AMBIENT
export DATABRICKS_CONFIG_PROFILE
dbx_json experiments get-experiment 42
""",
        str(databricks),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "https://workspace.example.com||experiments get-experiment 42 --output json"
    )


def test_dbx_json_clears_ambient_host_for_profile(tmp_path: Path):
    databricks = tmp_path / "databricks"
    databricks.write_text(
        r"""#!/bin/sh
printf '%s\n' "${DATABRICKS_HOST:-}|${DATABRICKS_CONFIG_PROFILE:-}|$*"
"""
    )
    databricks.chmod(0o755)

    result = run_shell(
        """
DATABRICKS_BIN=$1
PROFILE=selected
WORKSPACE_URL=https://workspace.example.com
DATABRICKS_HOST=https://ambient.example.com
DATABRICKS_CONFIG_PROFILE=AMBIENT
export DATABRICKS_HOST DATABRICKS_CONFIG_PROFILE
dbx_json experiments get-experiment 42
""",
        str(databricks),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "|AMBIENT|experiments get-experiment 42 --output json --profile selected"
    )


def test_validate_agent_name_rejects_unknown_agent():
    result = run_shell(
        """
AGENT_NAME=unknown
validate_agent_name
"""
    )

    assert result.returncode != 0
    assert "Unsupported coding agent: unknown" in result.stderr


def test_databricks_cli_installs_latest_release_in_user_bin(tmp_path: Path):
    result = run_shell(
        r"""
XDG_BIN_HOME=$1/bin
HOME=$1/home
export XDG_BIN_HOME HOME
find_compatible_databricks_cli() { :; }
progress() { :; }
success() { :; }
uname() {
    if [ "$1" = "-s" ]; then printf 'Darwin\n'; else printf 'arm64\n'; fi
}
curl() {
    case "$*" in
        *api.github.com*) printf '%s\n' '{"tag_name":"v9.8.7"}' ;;
        *databricks_cli_9.8.7_darwin_arm64.zip*)
            while [ "$#" -gt 0 ]; do
                if [ "$1" = "-o" ]; then : > "$2"; break; fi
                shift
            done
            ;;
        *) return 2 ;;
    esac
}
unzip() {
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "-d" ]; then
            printf '#!/bin/sh\n' > "$2/databricks"
            break
        fi
        shift
    done
}
ensure_databricks_cli
printf '%s\n' "$DATABRICKS_BIN"
""",
        str(tmp_path),
    )

    expected = tmp_path / "bin" / "databricks"
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == expected
    assert expected.exists()


def test_search_oss_experiments_paginates():
    result = run_shell(
        """
setup_tmp_dir=$(mktemp -d)
TRACKING_URI=https://mlflow.example
oss_curl() {
    case "$*" in
        *page_token*) printf '%s\n' '{"experiments":[{"name":"second"}]}' ;;
        *) printf '%s\n' '{"experiments":[{"name":"first"}],"next_page_token":"next"}' ;;
    esac
}
search_oss_experiments
cat "$setup_tmp_dir/experiments.json"
"""
    )

    assert result.returncode == 0, result.stderr
    assert '"name":"first"' in result.stdout
    assert '"name":"second"' in result.stdout


def test_local_server_always_prints_mlflow_command():
    result = run_shell(
        """
PATH=
curl() { :; }
run_with_spinner() { :; }
success() { :; }
configure_remote() { printf '%s\n' "$TRACKING_URI"; }
configure_local
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "http://127.0.0.1:5000"
    assert "mlflow server --port 5000" in result.stderr
    assert "uvx" not in result.stderr


@pytest.mark.parametrize("experiment_id", ["", "existing-id"])
def test_local_setup_resolves_experiment_without_prompting(tmp_path: Path, experiment_id: str):
    requests_path = tmp_path / "requests"
    result = run_shell(
        """
backend=local
repo_name=project
EXPERIMENT_ID=$1
requests_path=$2
lsof() { return 1; }
curl() { :; }
select_option() { return 98; }
prompt_text() { return 99; }
oss_curl() {
    case "$*" in
        */experiments/search)
            printf '200'
            ;;
        */experiments/get*)
            while [ "$1" != "-o" ]; do shift; done
            printf '%s' '{"experiment":{"name":"existing"}}' > "$2"
            printf '200'
            ;;
        */experiments/create)
            while [ "$1" != "-d" ]; do shift; done
            printf '%s' "$2" > "$requests_path"
            printf '%s' '{"experiment_id":"new-id"}'
            ;;
        *) return 97 ;;
    esac
}
configure_local
build_agent_prompt
""",
        experiment_id,
        str(requests_path),
    )

    assert result.returncode == 0, result.stderr
    assert "Do not start another server" in result.stdout
    assert "- Tracking URI: http://127.0.0.1:5000" in result.stdout
    assert f"- Experiment ID: {experiment_id or 'new-id'}" in result.stdout
    if experiment_id:
        assert "- Experiment name: existing" in result.stdout
        assert not requests_path.exists()
    else:
        request = json.loads(requests_path.read_text())
        assert request["name"].startswith("project-")
        assert f"- Experiment name: {request['name']}" in result.stdout
        assert request["tags"] == [{"key": "mlflow.experimentKind", "value": "genai_development"}]


@pytest.mark.parametrize("warehouse_id", ["", "warehouse-id"])
def test_existing_uc_experiment_preserves_warehouse_flag_without_reconfiguring_storage(
    warehouse_id: str,
):
    result = run_shell(
        """
ensure_databricks_cli() { :; }
resolve_databricks_profile() { PROFILE=DEFAULT; WORKSPACE_URL=https://example.databricks.com; }
authenticate_databricks() { :; }
resolve_databricks_experiment() {
    experiment_created=false
    trace_destination=catalog.schema.prefix
}
WAREHOUSE_ID=$1
link_uc_trace_storage() { return 98; }
configure_databricks
printf '%s\n' "$UC_SCHEMA" "$WAREHOUSE_ID"
""",
        warehouse_id,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["catalog.schema", warehouse_id]


def test_existing_workspace_experiment_skips_uc_configuration():
    result = run_shell(
        """
ensure_databricks_cli() { :; }
resolve_databricks_profile() { PROFILE=DEFAULT; WORKSPACE_URL=https://example.databricks.com; }
authenticate_databricks() { :; }
resolve_databricks_experiment() {
    experiment_created=false
    trace_destination=
}
link_uc_trace_storage() { return 99; }
configure_databricks
"""
    )

    assert result.returncode == 0, result.stderr
    assert "Existing experiment uses workspace storage" in result.stderr


def test_databricks_authentication_uses_host_and_profile_flags(tmp_path: Path):
    calls_path = tmp_path / "calls"
    auth_state_path = tmp_path / "authenticated"
    databricks = tmp_path / "databricks"
    databricks.write_text(
        """#!/bin/sh
if [ "$1 $2" = "auth login" ]; then
    printf '%s\n' "$*" > "$DATABRICKS_TEST_CALLS"
elif [ "$1 $2" = "auth token" ]; then
    if [ -f "$DATABRICKS_TEST_AUTH_STATE" ]; then
        printf '%s\n' '{}'
    else
        : > "$DATABRICKS_TEST_AUTH_STATE"
        exit 1
    fi
else
    exit 2
fi
"""
    )
    databricks.chmod(0o755)

    result = run_shell(
        """
DATABRICKS_BIN=$1
DATABRICKS_TEST_CALLS=$2
DATABRICKS_TEST_AUTH_STATE=$3
export DATABRICKS_TEST_CALLS DATABRICKS_TEST_AUTH_STATE
WORKSPACE_URL=https://workspace.example.com
PROFILE=DEFAULT
TTY_DEVICE=/dev/null
authenticate_databricks
""",
        str(databricks),
        str(calls_path),
        str(auth_state_path),
    )

    assert result.returncode == 0, result.stderr
    assert calls_path.read_text().strip() == (
        "auth login --host https://workspace.example.com --profile DEFAULT"
    )


TELEMETRY_CONFIG = {
    "mlflow_version": "3.18.0",
    "disable_telemetry": False,
    "rollout_percentage": 100,
    "disable_sdks": [],
    "disable_os": [],
    "disable_events": [],
    "disable_ui_events": ["agent_setup_script"],
}
TELEMETRY_DEPENDENCIES = [requests, packaging, urllib3, idna, certifi, charset_normalizer]


def telemetry_ci_vars() -> set[str]:
    tree = ast.parse(Path(mlflow.telemetry.utils.__file__).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_is_ci_env_or_testing"
    )
    literal = next(node for node in ast.walk(function) if isinstance(node, ast.Set))
    return {element.value for element in literal.elts}


class TelemetryServer(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), TelemetryHandler)
        self.config = TELEMETRY_CONFIG | {"ingestion_url": f"{self.url}/log"}
        self.config_status = 200
        self.config_delay = 0.0
        self.requests: list[str] = []
        self.posts: list[dict[str, Any]] = []
        self.post_statuses: list[int] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class TelemetryHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append(self.path)
        time.sleep(self.server.config_delay)
        body = json.dumps(self.server.config).encode()
        self.send_response(self.server.config_status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.server.requests.append(self.path)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.posts.append(json.loads(body))
        self.send_response(self.server.post_statuses.pop(0) if self.server.post_statuses else 200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class TelemetryHarness:
    def __init__(self, tmp_path: Path, server: TelemetryServer):
        self.server = server
        self.tmp_path = tmp_path
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.site = tmp_path / "site"
        self.site.mkdir()
        deps = tmp_path / "deps"
        deps.mkdir()
        for module in TELEMETRY_DEPENDENCIES:
            (deps / module.__name__).symlink_to(Path(module.__file__).parent)
        self.install("mlflow", "3.18.0")
        for name in ("python3", "python"):
            # -S hides the test environment's own mlflow so tests control what is installed.
            self.write_bin(name, f'exec "{sys.executable}" -S "$@"')
        self.xdg = tmp_path / "xdg"
        self.databricks_calls = tmp_path / "databricks.calls"
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in telemetry_ci_vars()
            and not key.startswith(("MLFLOW_", "DATABRICKS_", "_MLFLOW_"))
            and key not in ("DO_NOT_TRACK", "IS_IN_DB_MODEL_SERVING_ENV")
        }
        self.env = env | {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "PYTHONPATH": f"{self.site}:{deps}",
            "XDG_CONFIG_HOME": str(self.xdg),
            "_MLFLOW_SETUP_TELEMETRY_CONFIG_URL": server.url,
        }

    def install(self, name: str, version: str):
        dist = self.site / f"{name.replace('-', '_')}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")

    def write_bin(self, name: str, body: str):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)

    def run(
        self, body: str, shell: str = "sh", new_session: bool = False, **env: str
    ) -> subprocess.CompletedProcess[str]:
        command = f"""
MLFLOW_SETUP_SKIP_MAIN=1
export MLFLOW_SETUP_SKIP_MAIN
. "$1"
shift
{body}
"""
        # Every descendant inherits fd 3, so EOF on it means the detached sender has exited.
        read_fd, write_fd = os.pipe()
        process = subprocess.Popen(
            [shell, "-c", command, shell, str(SETUP_SCRIPT)],
            env=self.env | env,
            cwd=self.tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=(write_fd,),
            preexec_fn=lambda: (os.dup2(write_fd, 3), signal.signal(signal.SIGINT, signal.SIG_DFL)),
            start_new_session=new_session,
        )
        os.close(write_fd)
        stdout, stderr = process.communicate(timeout=30)
        with os.fdopen(read_fd) as descendants:
            descendants.read()
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)

    def started(self, body: str, **kwargs) -> subprocess.CompletedProcess[str]:
        return self.run(
            f"telemetry_start=$(($(date +%s) - 2))\nagent_choice=claude\nbackend=remote\n"
            f'telemetry_flags=" agent profile agent"\n{body}',
            **kwargs,
        )

    def data(self) -> dict[str, Any]:
        [post] = self.server.posts
        [record] = post["records"]
        return record["data"]


@pytest.fixture
def telemetry_server():
    server = TelemetryServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="telemetry-server")
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def telemetry(tmp_path, telemetry_server):
    return TelemetryHarness(tmp_path, telemetry_server)


def test_telemetry_ci_vars_match_python_client(telemetry):
    result = telemetry.run('printf "%s" "$telemetry_ci_vars"')

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.split()) == telemetry_ci_vars()


def test_telemetry_body_matches_python_client(telemetry):
    telemetry.xdg.joinpath("mlflow").mkdir(parents=True)
    installation_id = "123e4567-e89b-42d3-a456-426614174000"
    telemetry.xdg.joinpath("mlflow", "telemetry.json").write_text(
        json.dumps({"installation_id": installation_id})
    )

    result = telemetry.started(
        "TRACKING_URI=http://localhost:5000\ntelemetry_post success", MLFLOW_WORKSPACE="team"
    )

    assert result.returncode == 0, result.stderr
    assert telemetry.server.requests == ["/3.18.0.json", "/log"]
    [record] = telemetry.server.posts[0]["records"]
    assert re.fullmatch(r"[0-9a-f]{32}", record["partition-key"])
    data = record["data"]
    assert re.fullmatch(r"[0-9a-f]{32}", data["session_id"])
    assert data["duration_ms"] >= 2000
    assert isinstance(data["timestamp_ns"], int)
    assert json.loads(data["params"]) == {
        "agent": "claude",
        "backend": "oss",
        "flags": ["agent", "profile"],
    }
    version = sys.version_info
    assert {
        key: data[key]
        for key in data
        if key not in {"session_id", "timestamp_ns", "duration_ms", "params"}
    } == {
        "source_sdk": "mlflow",
        "mlflow_version": "3.18.0",
        "schema_version": 2,
        "python_version": f"{version.major}.{version.minor}.{version.micro}",
        "operating_system": platform.platform(),
        "environment": None,
        "coding_agent": None,
        "tracking_uri_scheme": "http",
        "is_localhost": True,
        "installation_id": installation_id,
        "ws_enabled": True,
        "event_name": "agent_setup_script",
        "status": "success",
    }


@pytest.mark.parametrize("contents", [None, '{"installation_id": "not-a-uuid"}'])
def test_telemetry_installation_id_missing_or_invalid(telemetry, contents):
    if contents:
        telemetry.xdg.joinpath("mlflow").mkdir(parents=True)
        telemetry.xdg.joinpath("mlflow", "telemetry.json").write_text(contents)

    telemetry.started("telemetry_post success")

    assert telemetry.data()["installation_id"] is None
    assert contents or not telemetry.xdg.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLFLOW_DISABLE_TELEMETRY", "true"),
        ("MLFLOW_DISABLE_TELEMETRY", "1"),
        ("MLFLOW_DISABLE_TELEMETRY", "yes"),
        ("MLFLOW_DISABLE_TELEMETRY", ""),
        ("DO_NOT_TRACK", "TRUE"),
        ("DO_NOT_TRACK", "1"),
        ("_MLFLOW_TESTING_TELEMETRY", "maybe"),
        *[(name, "") for name in sorted(telemetry_ci_vars())],
    ],
)
def test_telemetry_opt_outs(telemetry, name, value):
    telemetry.write_bin("python3", "exit 9")

    result = telemetry.started("telemetry_post success", **{name: value})

    assert result.returncode == 0, result.stderr
    assert telemetry.server.requests == []


@pytest.mark.parametrize(
    "env",
    [
        {"MLFLOW_DISABLE_TELEMETRY": "false"},
        {"MLFLOW_DISABLE_TELEMETRY": "0"},
        {"DO_NOT_TRACK": "yes"},
        {"_MLFLOW_TESTING_TELEMETRY": "true", "CI": "true", "MLFLOW_DISABLE_TELEMETRY": "1"},
    ],
)
def test_telemetry_env_values_that_still_send(telemetry, env):
    telemetry.started("telemetry_post success", **env)

    assert len(telemetry.server.posts) == 1


def test_telemetry_skips_oss_endpoint_inside_databricks(telemetry):
    telemetry.started("telemetry_post success", DATABRICKS_RUNTIME_VERSION="15.4")

    assert telemetry.server.requests == ["/3.18.0.json"]


def test_telemetry_requires_installed_mlflow(telemetry):
    shutil.rmtree(telemetry.site)
    telemetry.site.mkdir()

    telemetry.started("telemetry_post success")

    assert telemetry.server.requests == []


@pytest.mark.parametrize("sdk", ["mlflow-skinny", "mlflow-tracing"])
def test_telemetry_detects_source_sdk(telemetry, sdk):
    shutil.rmtree(telemetry.site)
    telemetry.site.mkdir()
    telemetry.install(sdk, "3.18.0")

    telemetry.started("telemetry_post success")

    assert telemetry.data()["source_sdk"] == sdk


def test_telemetry_falls_back_to_python(telemetry):
    telemetry.write_bin("python3", f'PYTHONPATH= exec "{sys.executable}" -S "$@"')

    telemetry.started("telemetry_post success")

    assert len(telemetry.server.posts) == 1


@pytest.mark.parametrize("version", ["3.19.0.dev0", "3.18.0.post1"])
def test_telemetry_skips_unpublished_versions(telemetry, version):
    shutil.rmtree(telemetry.site)
    telemetry.site.mkdir()
    telemetry.install("mlflow", version)

    telemetry.started("telemetry_post success")

    assert telemetry.server.requests == []


@pytest.mark.parametrize(
    "config",
    [
        {"disable_telemetry": True},
        {"ingestion_url": None},
        {"mlflow_version": "3.17.0"},
        {"disable_events": ["other", "agent_setup_script"]},
        {"disable_sdks": ["mlflow"]},
        {"disable_os": [sys.platform]},
        {"rollout_percentage": -1},
    ],
)
def test_telemetry_remote_config_gating(telemetry, config):
    telemetry.server.config |= config

    telemetry.started("telemetry_post success")

    assert telemetry.server.posts == []


def test_telemetry_config_error_sends_nothing(telemetry):
    telemetry.server.config_status = 403

    telemetry.started("telemetry_post success")

    assert telemetry.server.requests == ["/3.18.0.json"]


@pytest.mark.parametrize(
    ("setup", "host"),
    [
        ("PROFILE=DEFAULT", ""),
        ("WORKSPACE_URL=https://workspace.example.com", "https://workspace.example.com"),
    ],
)
def test_telemetry_databricks_backend_uses_workspace_endpoint(telemetry, setup, host):
    telemetry.write_bin(
        "databricks",
        f'printf "%s\\n" "host=${{DATABRICKS_HOST-unset}}" "$@" > "{telemetry.databricks_calls}"',
    )

    telemetry.started(
        f"backend=databricks\nDATABRICKS_BIN={telemetry.bin / 'databricks'}\n{setup}\n"
        "TRACKING_URI=databricks\ntelemetry_post success",
        DATABRICKS_RUNTIME_VERSION="15.4",
    )

    assert telemetry.server.posts == []
    calls = telemetry.databricks_calls.read_text().splitlines()
    assert calls[:5] == [
        f"host={host}",
        "api",
        "post",
        "/api/2.0/mlflow/client-telemetry/ingest",
        "--json",
    ]
    assert calls[6:] == (["--profile", "DEFAULT"] if host == "" else [])
    [event] = json.loads(calls[5])["events"]
    assert "params" not in event
    assert json.loads(event["params_json"])["backend"] == "databricks"
    assert event["tracking_uri_scheme"] == "databricks"
    assert event["event_name"] == "agent_setup_script"


def test_telemetry_sends_once_and_survives_exec(telemetry):
    result = telemetry.started("telemetry_send success\ntelemetry_send failure\nexec true")

    assert result.returncode == 0, result.stderr
    assert telemetry.data()["status"] == "success"


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ("die boom", "failure"),
        ("value=$(die boom)", "failure"),
        ("kill -INT $$", "canceled"),
        ("kill -TERM $$", "canceled"),
        ("kill -HUP $$", "canceled"),
        ("telemetry_send success\ndie boom", "success"),
    ],
)
@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_telemetry_exit_paths_send_once(telemetry, body, status, shell):
    result = telemetry.started(body, shell=shell)

    assert result.returncode != 0
    assert telemetry.data()["status"] == status


def test_telemetry_child_survives_sighup(telemetry):
    telemetry.server.config_delay = 0.8

    result = telemetry.started(
        "telemetry_send success\nsleep 0.3\nkill -HUP 0\nsleep 5", new_session=True
    )

    assert result.returncode != 0
    assert telemetry.data()["status"] == "success"


def test_telemetry_does_not_retry_interpreter_after_other_errors(telemetry):
    telemetry.write_bin("python3", "exit 1")

    telemetry.started("telemetry_post success")

    assert telemetry.server.requests == []


def test_telemetry_skipped_before_main_starts(telemetry):
    telemetry.run("die boom")
    telemetry.run("exit 0")

    assert telemetry.server.requests == []


def test_telemetry_main_manual_path(telemetry):
    result = telemetry.run(
        "inspect_repository() { repo_root=$PWD; }\n"
        "configure_remote() { :; }\n"
        "show_manual_setup() { :; }\n"
        'main --agent manual --tracking-uri "https://mlflow.example.com" --experiment-id 1'
    )

    assert result.returncode == 0, result.stderr
    data = telemetry.data()
    assert data["status"] == "success"
    assert data["tracking_uri_scheme"] == "https"
    assert data["is_localhost"] is False
    assert json.loads(data["params"]) == {
        "agent": "manual",
        "backend": "oss",
        "flags": ["agent", "experiment-id", "tracking-uri"],
    }


def test_telemetry_main_launch_agent_path(telemetry):
    telemetry.write_bin("claude", "exit 0")

    result = telemetry.run(
        "inspect_repository() { repo_root=$PWD; }\n"
        "configure_local() { :; }\n"
        "choose_backend() { backend=local; }\n"
        "build_agent_prompt() { printf prompt; }\n"
        "main --agent claude"
    )

    assert result.returncode == 0, result.stderr
    data = telemetry.data()
    assert data["status"] == "success"
    assert json.loads(data["params"]) == {"agent": "claude", "backend": "local", "flags": ["agent"]}


def test_telemetry_main_early_failure_reports_requested_agent(telemetry):
    result = telemetry.run("inspect_repository() { die boom; }\nmain --agent codex")

    assert result.returncode == 1
    data = telemetry.data()
    assert data["status"] == "failure"
    assert json.loads(data["params"]) == {"agent": "codex", "backend": None, "flags": ["agent"]}


def test_telemetry_ctrl_c_in_selection_is_canceled(telemetry):
    telemetry.write_bin("stty", "exit 0")

    result = telemetry.run(
        "telemetry_start=$(date +%s)\nTTY_DEVICE=/dev/null\n"
        "read_key() { key_code=3; }\nselect_option Question One Two"
    )

    assert result.returncode == 1
    assert telemetry.data()["status"] == "canceled"


def test_telemetry_exit_choice_is_canceled(telemetry):
    result = telemetry.run(
        "telemetry_start=$(date +%s)\nselect_option() { selected_index=0; }\ninspect_repository",
        GIT_CEILING_DIRECTORIES=str(telemetry.tmp_path.parent),
    )

    assert result.returncode == 1
    assert telemetry.data()["status"] == "canceled"


def test_telemetry_config_urls_match_python_client():
    urls = set(
        re.findall(r'"(https://config[a-z-]*\.mlflow-telemetry\.io)"', SETUP_SCRIPT.read_text())
    )

    assert urls == {
        mlflow.telemetry.constant.CONFIG_URL,
        mlflow.telemetry.constant.CONFIG_STAGING_URL,
    }


def test_telemetry_retries_oss_server_errors(telemetry):
    telemetry.server.post_statuses = [500, 429]

    telemetry.started("telemetry_post success")

    assert len(telemetry.server.posts) == 3


def test_telemetry_dirty_repo_exit_is_canceled(telemetry):
    repo = telemetry.tmp_path / "repo"
    repo.mkdir()
    git = [
        "git",
        "-C",
        str(repo),
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.check_call([*git, "init", "-q"])
    (repo / "file.txt").write_text("a")
    subprocess.check_call([*git, "add", "file.txt"])
    subprocess.check_call([*git, "commit", "-qm", "init"])
    (repo / "file.txt").write_text("b")

    result = telemetry.run(
        f'cd "{repo}"\ntelemetry_start=$(date +%s)\n'
        "select_option() { selected_index=0; }\ninspect_repository"
    )

    assert result.returncode == 1
    assert "Commit or stash the changes" in result.stderr
    assert telemetry.data()["status"] == "canceled"
