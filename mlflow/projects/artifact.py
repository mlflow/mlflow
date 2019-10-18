import os
import posixpath

from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository


_MLFLOW_DOCKER_WORKDIR_PATH = "/mlflow/projects/code/"


def _get_local_artifact_cmd_and_envs(artifact_repo):
    artifact_dir = artifact_repo.artifact_dir
    container_path = artifact_dir
    if not os.path.isabs(container_path):
        container_path = os.path.join(_MLFLOW_DOCKER_WORKDIR_PATH, container_path)
        container_path = os.path.normpath(container_path)
    abs_artifact_dir = os.path.abspath(artifact_dir)
    return ["-v", "%s:%s" % (abs_artifact_dir, container_path)], {}


def _get_s3_artifact_cmd_and_envs(artifact_repo):
    # pylint: disable=unused-argument
    aws_path = posixpath.expanduser("~/.aws")

    volumes = []
    if posixpath.exists(aws_path):
        volumes = ["-v", "%s:%s" % (str(aws_path), "/.aws")]
    envs = {
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    }
    envs = dict((k, v) for k, v in envs.items() if v is not None)
    return volumes, envs


def _get_azure_blob_artifact_cmd_and_envs(artifact_repo):
    # pylint: disable=unused-argument
    envs = {
        "AZURE_STORAGE_CONNECTION_STRING": os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        "AZURE_STORAGE_ACCESS_KEY": os.environ.get("AZURE_STORAGE_ACCESS_KEY")
    }
    envs = dict((k, v) for k, v in envs.items() if v is not None)
    return [], envs


def _get_gcs_artifact_cmd_and_envs(artifact_repo):
    # pylint: disable=unused-argument
    cmds = []
    envs = {}

    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        cmds = ["-v", "{}:/.gcs".format(credentials_path)]
        envs["GOOGLE_APPLICATION_CREDENTIALS"] = "/.gcs"
    return cmds, envs


def _get_hdfs_artifact_cmd_and_envs(artifact_repo):
    # pylint: disable=unused-argument
    cmds = []
    envs = {
        "MLFLOW_HDFS_DRIVER": os.environ.get("MLFLOW_HDFS_DRIVER"),
        "MLFLOW_KERBEROS_TICKET_CACHE": os.environ.get("MLFLOW_KERBEROS_TICKET_CACHE"),
        "MLFLOW_KERBEROS_USER": os.environ.get("MLFLOW_KERBEROS_USER"),
        "MLFLOW_PYARROW_EXTRA_CONF": os.environ.get("MLFLOW_PYARROW_EXTRA_CONF")
    }
    envs = dict((k, v) for k, v in envs.items() if v is not None)

    if "MLFLOW_KERBEROS_TICKET_CACHE" in envs:
        ticket_cache = envs["MLFLOW_KERBEROS_TICKET_CACHE"]
        cmds = ["-v", "{}:{}".format(ticket_cache, ticket_cache)]
    return cmds, envs


_artifact_storages = {
    LocalArtifactRepository: _get_local_artifact_cmd_and_envs,
    S3ArtifactRepository: _get_s3_artifact_cmd_and_envs,
    AzureBlobArtifactRepository: _get_azure_blob_artifact_cmd_and_envs,
    HdfsArtifactRepository: _get_hdfs_artifact_cmd_and_envs,
    GCSArtifactRepository: _get_gcs_artifact_cmd_and_envs,
}
