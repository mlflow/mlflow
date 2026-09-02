import os
import subprocess
from pathlib import Path

import pytest

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
    result = run_shell('json_escape "$1"', 'a\\path"with-quote')

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


def test_json_experiment_strings_fallback_handles_compact_response(tmp_path: Path):
    for command in ("head", "sed"):
        (tmp_path / command).symlink_to(Path("/usr/bin") / command)
    result = run_shell(
        """
PATH=$1
printf '%s%s\n' \
    '{"experiments":[{"experiment_id":"1","name":"one"},' \
    '{"experiment_id":"2","name":"two"}]}' |
    json_experiment_strings name
""",
        str(tmp_path),
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
        "#!/bin/sh\n"
        "printf '%s\\n' \"${DATABRICKS_HOST:-}|${DATABRICKS_CONFIG_PROFILE:-}|$*\"\n"
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
    assert result.stdout.strip() == str(expected)
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


def test_local_server_always_prints_mlflow_command(tmp_path: Path):
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\n")
    curl.chmod(0o755)

    result = run_shell(
        """
PATH=$1
run_with_spinner() { :; }
success() { :; }
EXPERIMENT_NAME=test
configure_local
""",
        str(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "mlflow server --port 5000" in result.stderr
    assert "uvx" not in result.stderr


def test_existing_uc_experiment_selects_warehouse_without_reconfiguring_storage():
    result = run_shell(
        """
ensure_databricks_cli() { :; }
resolve_databricks_profile() { PROFILE=DEFAULT; WORKSPACE_URL=https://example.databricks.com; }
authenticate_databricks() { :; }
resolve_databricks_experiment() {
    experiment_created=false
    trace_destination=catalog.schema.prefix
}
select_warehouse() { WAREHOUSE_ID=warehouse-id; }
link_uc_trace_storage() { return 98; }
configure_databricks
printf '%s\n' "$UC_SCHEMA" "$WAREHOUSE_ID"
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["catalog.schema", "warehouse-id"]


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
select_warehouse() { return 98; }
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
