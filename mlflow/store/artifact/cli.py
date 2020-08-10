import logging
import os
import sys

import click

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking import _get_store
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository, archive_artifacts, \
    remove_folder

_logger = logging.getLogger(__name__)

ARTIFACT_NAME = "artifacts.har"


@click.group("artifacts")
def commands():
    """
    Upload, list, and download artifacts from an MLflow artifact repository.

    To manage artifacts for a run associated with a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("log-artifact")
@click.option("--local-file", "-l", required=True, help="Local path to artifact to log")
@click.option("--run-id", "-r", required=True, help="Run ID into which we should log the artifact.")
@click.option(
    "--artifact-path",
    "-a",
    help="If specified, we will log the artifact into this subdirectory of the "
    + "run's artifact directory.",
)
def log_artifact(local_file, run_id, artifact_path):
    """
    Log a local file as an artifact of a run, optionally within a run-specific
    artifact path. Run artifacts can be organized into directories, so you can
    place the artifact in a directory this way.
    """
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = get_artifact_repository(artifact_uri)
    artifact_repo.log_artifact(local_file, artifact_path)
    _logger.info(
        "Logged artifact from local file %s to artifact_path=%s", local_file, artifact_path
    )


@commands.command("log-artifacts")
@click.option("--local-dir", "-l", required=True, help="Directory of local artifacts to log")
@click.option("--run-id", "-r", required=True, help="Run ID into which we should log the artifact.")
@click.option(
    "--artifact-path",
    "-a",
    help="If specified, we will log the artifact into this subdirectory of the "
    + "run's artifact directory.",
)
def log_artifacts(local_dir, run_id, artifact_path):
    """
    Log the files within a local directory as an artifact of a run, optionally
    within a run-specific artifact path. Run artifacts can be organized into
    directories, so you can place the artifact in a directory this way.
    """
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = get_artifact_repository(artifact_uri)
    artifact_repo.log_artifacts(local_dir, artifact_path)
    _logger.info("Logged artifact from local dir %s to artifact_path=%s", local_dir, artifact_path)


@commands.command("list")
@click.option("--run-id", "-r", required=True, help="Run ID to be listed")
@click.option(
    "--artifact-path",
    "-a",
    help="If specified, a path relative to the run's root directory to list.",
)
def list_artifacts(run_id, artifact_path):
    """
    Return all the artifacts directly under run's root artifact directory,
    or a sub-directory. The output is a JSON-formatted list.
    """
    artifact_path = artifact_path if artifact_path is not None else ""
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = get_artifact_repository(artifact_uri)
    file_infos = artifact_repo.list_artifacts(artifact_path)
    print(_file_infos_to_json(file_infos))


def _file_infos_to_json(file_infos):
    json_list = [message_to_json(file_info.to_proto()) for file_info in file_infos]
    return "[" + ", ".join(json_list) + "]"


@commands.command("download")
@click.option("--run-id", "-r", help="Run ID from which to download")
@click.option(
    "--artifact-path",
    "-a",
    help="For use with Run ID: if specified, a path relative to the run's root "
    "directory to download",
)
@click.option(
    "--artifact-uri",
    "-u",
    help="URI pointing to the artifact file or artifacts directory; use as an "
    "alternative to specifying --run_id and --artifact-path",
)
def download_artifacts(run_id, artifact_path, artifact_uri):
    """
    Download an artifact file or directory to a local directory.
    The output is the name of the file or directory on the local disk.

    Either ``--run-id`` or ``--artifact-uri`` must be provided.
    """
    if run_id is None and artifact_uri is None:
        _logger.error("Either ``--run-id`` or ``--artifact-uri`` must be provided.")
        sys.exit(1)

    if artifact_uri is not None:
        print(_download_artifact_from_uri(artifact_uri))
        return

    artifact_path = artifact_path if artifact_path is not None else ""
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = get_artifact_repository(artifact_uri)
    artifact_location = artifact_repo.download_artifacts(artifact_path)
    print(artifact_location)


@commands.command("archive-hdfs-artifacts")
@click.option("--run-ids", "-r", required=True, help="Comma separated list of Run IDs for which "
                                                     "we will archive the artifacts")
def archive_hdfs_artifacts(run_ids):
    """
    Pack into an hadoop archive a folder on HDFS. Only HdfsArtifactStore supported
    """
    store = _get_store()
    run_ids = run_ids.split(',')
    for run_id in run_ids:
        artifact_uri = store.get_run(run_id).info.artifact_uri
        artifact_repo = get_artifact_repository(artifact_uri)
        if not isinstance(artifact_repo, HdfsArtifactRepository):
            _logger.error("Artifacts store must be Hdfs")
            sys.exit(1)
        parent_dir = os.path.dirname(artifact_uri)
        new_artifact_path = archive_artifacts(artifact_repo, parent_dir, ARTIFACT_NAME)
        _logger.debug("Update database: artifact_uri for run (run_id = %s) to %s",
                      run_id, new_artifact_path)
        store.update_artifacts_location(run_id, new_artifact_path)
        remove_folder(artifact_uri)


if __name__ == '__main__':
    commands()
