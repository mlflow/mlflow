import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
