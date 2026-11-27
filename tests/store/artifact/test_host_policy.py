import pytest

from mlflow.environment_variables import (
    MLFLOW_ALLOW_LOCAL_ARTIFACT_LOCATION_OUTSIDE_ROOT,
    MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES,
)
from mlflow.exceptions import MlflowException
from mlflow.server.constants import (
    ARTIFACT_ROOT_ENV_VAR,
    ARTIFACTS_DESTINATION_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
)
from mlflow.store.artifact.host_policy import (
    _SERVER_ARTIFACT_ROOT_ENV_VAR,
    _SERVER_ARTIFACTS_DESTINATION_ENV_VAR,
    _SERVER_SERVE_ARTIFACTS_ENV_VAR,
    enforce_server_artifact_uri_host_policy,
    host_addressed_uri_targets,
    rejected_host_addressed_scheme,
    rejected_local_artifact_path,
)
from mlflow.utils.os import is_windows


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://bucket/path", None),
        ("file:///tmp/mlruns", None),
        ("/tmp/mlruns", None),
        ("mlflow-artifacts:/experiments/1", None),
        # An empty authority resolves against the tracking server itself, like the form above.
        ("mlflow-artifacts:///experiments/1", None),
        # A port with no host is an authority, so it is a (never trusted) target, not "self".
        ("mlflow-artifacts://:5000/experiments/1", {("http", "", 5000)}),
        ("ftp://host/pub", {("ftp", "host", 21)}),
        ("ftp://user:pw@HOST:2121/pub", {("ftp", "host", 2121)}),
        ("sftp://host/data", {("sftp", "host", 22)}),
        ("hdfs://namenode:8020/mlflow", {("hdfs", "namenode", 8020)}),
        ("hdfs://namenode/mlflow", {("hdfs", "namenode", None)}),
        ("viewfs://namenode:8020/mlflow", {("hdfs", "namenode", 8020)}),
        ("http://host/api", {("http", "host", 80)}),
        ("http://host:80/api", {("http", "host", 80)}),
        ("https://host/api", {("http", "host", 443)}),
        ("mlflow-artifacts://host:5000/exp", {("http", "host", 5000)}),
        ("mlflow-artifacts://host/exp", {("http", "host", 80), ("http", "host", 443)}),
        ("ftp:///pub", {("ftp", "", 21)}),
        ("hdfs:///mlflow", {("hdfs", "", None)}),
        (
            "r2://bucket@acct.r2.cloudflarestorage.com/x",
            {("r2", "acct.r2.cloudflarestorage.com", 443)},
        ),
        (
            "b2://bucket@s3.us-west-004.backblazeb2.com/x",
            {("b2", "s3.us-west-004.backblazeb2.com", 443)},
        ),
        ("abfss://fs@acct.dfs.core.windows.net/x", {("abfss", "acct.dfs.core.windows.net", 443)}),
        ("wasbs://container@acct.blob.core.windows.net/x", None),
        ("s3://bucket/x", None),
    ],
)
def test_host_addressed_uri_targets(uri, expected):
    assert host_addressed_uri_targets(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "http://evil.example\\@trusted.example:5000/api",
        "http://trusted.example:5000 /api",
        "http://trusted.example:notaport/api",
    ],
)
def test_malformed_authority_is_never_trusted(monkeypatch, uri):
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "http://trusted.example:5000/root")
    assert {t[0] for t in host_addressed_uri_targets(uri)} == {"malformed"}
    assert rejected_host_addressed_scheme(uri) == "http"


@pytest.mark.parametrize(
    ("trusted", "uri"),
    [
        ("hdfs://namenode:8020/mlflow", "hdfs://NAMENODE:8020/other"),
        ("hdfs://namenode:8020/mlflow", "viewfs://namenode:8020/other"),
        ("hdfs:///mlflow", "hdfs:///other"),
        ("http://artifacts/root", "http://artifacts:80/root/x"),
        ("https://artifacts/root", "https://artifacts:443/root/x"),
        (
            "mlflow-artifacts://artifacts:5000",
            "http://artifacts:5000/api/2.0/mlflow-artifacts/artifacts/x",
        ),
        ("sftp://svc@sftp-host/data", "sftp://other@sftp-host:22/data/x"),
        # A portless `mlflow-artifacts://host` resolves to the tracking transport's port, so it
        # matches the configured root on either well-known HTTP port and vice versa.
        ("http://artifacts/root", "mlflow-artifacts://artifacts/root/x"),
        ("https://artifacts/root", "mlflow-artifacts://artifacts/root/x"),
        ("mlflow-artifacts://artifacts", "http://artifacts/api/2.0/mlflow-artifacts/artifacts/x"),
        ("mlflow-artifacts://artifacts", "https://artifacts/api/2.0/mlflow-artifacts/artifacts/x"),
        ("mlflow-artifacts://artifacts", "mlflow-artifacts://ARTIFACTS/x"),
        (
            "r2://mlflow@acct.r2.cloudflarestorage.com/root",
            "r2://other@acct.r2.cloudflarestorage.com/x",
        ),
        ("abfss://fs@acct.dfs.core.windows.net/root", "abfss://other@acct.dfs.core.windows.net/x"),
    ],
)
def test_locations_on_trusted_hosts_are_accepted(monkeypatch, trusted, uri):
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, trusted)
    assert rejected_host_addressed_scheme(uri) is None


@pytest.mark.parametrize(
    ("trusted", "uri", "scheme"),
    [
        ("hdfs://namenode:8020/mlflow", "hdfs://other:8020/x", "hdfs"),
        ("hdfs://namenode:8020/mlflow", "hdfs://namenode:9000/x", "hdfs"),
        ("hdfs://namenode:8020/mlflow", "hdfs://namenode/x", "hdfs"),
        ("hdfs://namenode:8020/mlflow", "ftp://namenode:8020/x", "ftp"),
        ("hdfs://namenode:8020/mlflow", "ftp:///pub", "ftp"),
        ("https://artifacts/root", "http://artifacts/root/x", "http"),
        ("http://artifacts/root", "http://artifacts:8080/root/x", "http"),
        ("mlflow-artifacts:/", "mlflow-artifacts://other:5000/x", "mlflow-artifacts"),
        ("mlflow-artifacts://artifacts:5000", "mlflow-artifacts://artifacts/x", "mlflow-artifacts"),
        ("mlflow-artifacts://artifacts", "http://artifacts:5000/x", "http"),
        ("http://artifacts/root", "mlflow-artifacts://other/root/x", "mlflow-artifacts"),
        ("r2://mlflow@acct.r2.cloudflarestorage.com/root", "r2://mlflow@evil.example/x", "r2"),
        ("b2://mlflow@s3.us-west-004.backblazeb2.com/root", "b2://mlflow@evil.example/x", "b2"),
        ("abfss://fs@acct.dfs.core.windows.net/root", "abfss://fs@acct.evil.example/x", "abfss"),
        # The trusted name appears only as userinfo; the host actually contacted is `evil.example`.
        ("http://trusted.example/root", "http://trusted.example@evil.example/root", "http"),
        (
            "r2://mlflow@acct.r2.cloudflarestorage.com/root",
            "r2://acct.r2.cloudflarestorage.com@evil.example/x",
            "r2",
        ),
    ],
)
def test_locations_on_other_hosts_are_rejected(monkeypatch, trusted, uri, scheme):
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, trusted)
    assert rejected_host_addressed_scheme(uri) == scheme


def test_artifacts_destination_host_is_trusted(monkeypatch):
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "mlflow-artifacts:/")
    monkeypatch.setenv(_SERVER_ARTIFACTS_DESTINATION_ENV_VAR, "hdfs://namenode:8020/mlflow")
    assert rejected_host_addressed_scheme("hdfs://namenode:8020/mlflow/1/run") is None
    assert rejected_host_addressed_scheme("hdfs://other:8020/mlflow") == "hdfs"


def test_allowed_schemes_env_var_is_case_insensitive(monkeypatch):
    monkeypatch.setenv(MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES.name, "HDFS, sftp")
    assert rejected_host_addressed_scheme("hdfs://other/x") is None
    assert rejected_host_addressed_scheme("sftp://other/x") is None
    assert rejected_host_addressed_scheme("ftp://other/x") == "ftp"


def test_marker_env_vars_match_server_constants():
    # The policy cannot import `mlflow.server.constants`; a rename there must not silently turn
    # the registry-level enforcement or `TracingClient`'s in-server artifact resolution off.
    assert _SERVER_ARTIFACT_ROOT_ENV_VAR == ARTIFACT_ROOT_ENV_VAR
    assert _SERVER_ARTIFACTS_DESTINATION_ENV_VAR == ARTIFACTS_DESTINATION_ENV_VAR
    assert _SERVER_SERVE_ARTIFACTS_ENV_VAR == SERVE_ARTIFACTS_ENV_VAR


def test_enforcement_only_applies_in_tracking_server_processes(monkeypatch):
    monkeypatch.delenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, raising=False)
    enforce_server_artifact_uri_host_policy("ftp://other/x")

    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "./mlruns")
    with pytest.raises(MlflowException, match="does not connect to artifact location"):
        enforce_server_artifact_uri_host_policy("ftp://other/x")
    enforce_server_artifact_uri_host_policy("s3://bucket/x")


def _setup_local_root(monkeypatch, tmp_path):
    """Configure the server with a local artifacts destination and a proxied default root."""
    root = tmp_path / "mlartifacts"
    root.mkdir()
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "mlflow-artifacts:/")
    monkeypatch.setenv(_SERVER_ARTIFACTS_DESTINATION_ENV_VAR, str(root))
    return root


def test_local_path_inside_trusted_root_is_accepted(monkeypatch, tmp_path):
    root = _setup_local_root(monkeypatch, tmp_path)
    assert rejected_local_artifact_path(str(root)) is None
    assert rejected_local_artifact_path(f"file://{root}/1/run/artifacts") is None
    assert rejected_local_artifact_path(f"{root}/1/run/artifacts") is None


def test_local_path_on_default_artifact_root_is_accepted(monkeypatch, tmp_path):
    root = tmp_path / "mlruns"
    root.mkdir()
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, str(root))
    assert rejected_local_artifact_path(f"file://{root}/0/run/artifacts") is None


@pytest.mark.parametrize("make_uri", [lambda p: f"file://{p}", lambda p: str(p)])
def test_local_path_outside_trusted_roots_is_rejected(monkeypatch, tmp_path, make_uri):
    _setup_local_root(monkeypatch, tmp_path)
    outside = tmp_path / "elsewhere"
    assert rejected_local_artifact_path(make_uri(outside)) == str(outside)


def test_local_path_traversal_out_of_root_is_rejected(monkeypatch, tmp_path):
    root = _setup_local_root(monkeypatch, tmp_path)
    escape = f"file://{root}/../secret"
    # The resolved path escapes the root even though the string starts with it.
    assert rejected_local_artifact_path(escape) is not None


@pytest.mark.skipif(is_windows(), reason="POSIX authority handling")
def test_local_file_uri_authority_is_resolved_like_the_repository(monkeypatch, tmp_path):
    _setup_local_root(monkeypatch, tmp_path)
    # LocalArtifactRepository drops the authority on POSIX, so `file://host/srv/x` uses `/srv/x`.
    assert rejected_local_artifact_path("file://other-host/srv/shared/data") == "/srv/shared/data"


def test_local_path_accepted_when_opted_out(monkeypatch, tmp_path):
    _setup_local_root(monkeypatch, tmp_path)
    monkeypatch.setenv(MLFLOW_ALLOW_LOCAL_ARTIFACT_LOCATION_OUTSIDE_ROOT.name, "true")
    assert rejected_local_artifact_path("file:///srv/shared") is None


def test_local_path_not_enforced_off_server(monkeypatch):
    monkeypatch.delenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, raising=False)
    assert rejected_local_artifact_path("file:///srv/shared") is None


def test_local_path_rejected_when_server_has_no_local_root(monkeypatch):
    # A server that stores only on a remote backend has no trusted local directory at all, so
    # every local location is refused.
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "s3://bucket/root")
    monkeypatch.setenv(_SERVER_ARTIFACTS_DESTINATION_ENV_VAR, "s3://bucket/root")
    assert rejected_local_artifact_path("file:///srv/shared") == "/srv/shared"


def test_non_local_locations_are_not_subject_to_local_policy(monkeypatch, tmp_path):
    _setup_local_root(monkeypatch, tmp_path)
    for uri in ("s3://bucket/x", "mlflow-artifacts:/1/run/artifacts", "hdfs://namenode/x"):
        assert rejected_local_artifact_path(uri) is None


def test_enforce_does_not_check_stored_local_locations(monkeypatch, tmp_path):
    _setup_local_root(monkeypatch, tmp_path)
    # Only client-submitted locations are confined; a stored local location stays servable.
    enforce_server_artifact_uri_host_policy(f"file://{tmp_path}/mlruns/0/run/artifacts")
    enforce_server_artifact_uri_host_policy("/srv/shared/1/run/artifacts")
