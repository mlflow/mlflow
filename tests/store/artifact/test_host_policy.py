import pytest

from mlflow.environment_variables import MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.host_policy import (
    _SERVER_ARTIFACT_ROOT_ENV_VAR,
    _SERVER_ARTIFACTS_DESTINATION_ENV_VAR,
    enforce_server_artifact_uri_host_policy,
    host_addressed_uri_targets,
    rejected_host_addressed_scheme,
)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://bucket/path", None),
        ("file:///tmp/mlruns", None),
        ("/tmp/mlruns", None),
        ("mlflow-artifacts:/experiments/1", None),
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


def test_enforcement_only_applies_in_tracking_server_processes(monkeypatch):
    monkeypatch.delenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, raising=False)
    enforce_server_artifact_uri_host_policy("ftp://other/x")

    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "./mlruns")
    with pytest.raises(MlflowException, match="does not connect to artifact location"):
        enforce_server_artifact_uri_host_policy("ftp://other/x")
    enforce_server_artifact_uri_host_policy("s3://bucket/x")
