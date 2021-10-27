import os
from unittest import mock

from mlflow.projects.backend.local import _get_docker_artifact_storage_cmd_and_envs


def test_docker_s3_artifact_cmd_and_envs_from_env():
    mock_env = {
        "AWS_SECRET_ACCESS_KEY": "mock_secret",
        "AWS_ACCESS_KEY_ID": "mock_access_key",
        "MLFLOW_S3_ENDPOINT_URL": "mock_endpoint",
        "MLFLOW_S3_IGNORE_TLS": "false",
    }
    with mock.patch.dict("os.environ", mock_env), mock.patch(
        "posixpath.exists", return_value=False
    ):
        cmds, envs = _get_docker_artifact_storage_cmd_and_envs("s3://mock_bucket")
        assert cmds == []
        assert envs == mock_env


def test_docker_s3_artifact_cmd_and_envs_from_home():
    mock_env = {}
    with mock.patch.dict("os.environ", mock_env), mock.patch(
        "posixpath.exists", return_value=True
    ), mock.patch("posixpath.expanduser", return_value="mock_volume"):
        cmds, envs = _get_docker_artifact_storage_cmd_and_envs("s3://mock_bucket")
        assert cmds == ["-v", "mock_volume:/.aws"]
        assert envs == mock_env


def test_docker_wasbs_artifact_cmd_and_envs_from_home():
    # pylint: disable=unused-import, unused-variable
    from azure.storage.blob import BlobServiceClient

    mock_env = {
        "AZURE_STORAGE_CONNECTION_STRING": "mock_connection_string",
        "AZURE_STORAGE_ACCESS_KEY": "mock_access_key",
    }
    wasbs_uri = "wasbs://container@account.blob.core.windows.net/some/path"
    with mock.patch.dict("os.environ", mock_env), mock.patch(
        "azure.storage.blob.BlobServiceClient"
    ):
        cmds, envs = _get_docker_artifact_storage_cmd_and_envs(wasbs_uri)
        assert cmds == []
        assert envs == mock_env


def test_docker_gcs_artifact_cmd_and_envs_from_home():
    mock_env = {
        "GOOGLE_APPLICATION_CREDENTIALS": "mock_credentials_path",
    }
    gs_uri = "gs://mock_bucket"
    with mock.patch.dict("os.environ", mock_env, clear=True):
        cmds, envs = _get_docker_artifact_storage_cmd_and_envs(gs_uri)
        assert cmds == ["-v", "mock_credentials_path:/.gcs"]
        assert envs == {"GOOGLE_APPLICATION_CREDENTIALS": "/.gcs"}


def test_docker_hdfs_artifact_cmd_and_envs_from_home():
    mock_env = {
        "MLFLOW_KERBEROS_TICKET_CACHE": "/mock_ticket_cache",
        "MLFLOW_KERBEROS_USER": "mock_krb_user",
        "MLFLOW_PYARROW_EXTRA_CONF": "mock_pyarrow_extra_conf",
    }
    hdfs_uri = "hdfs://host:8020/path"
    with mock.patch.dict("os.environ", mock_env, clear=True):
        cmds, envs = _get_docker_artifact_storage_cmd_and_envs(hdfs_uri)
        assert cmds == ["-v", "/mock_ticket_cache:/mock_ticket_cache"]
        assert envs == mock_env


def test_docker_local_artifact_cmd_and_envs():
    host_path_expected = os.path.abspath("./mlruns")
    container_path_expected = "/mlflow/projects/code/mlruns"
    cmds, envs = _get_docker_artifact_storage_cmd_and_envs("file:./mlruns")
    assert cmds == ["-v", "{}:{}".format(host_path_expected, container_path_expected)]
    assert envs == {}


def test_docker_unknown_uri_artifact_cmd_and_envs():
    cmd, envs = _get_docker_artifact_storage_cmd_and_envs("file-plugin://some_path")
    assert cmd == []
    assert envs == {}
