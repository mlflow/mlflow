from mlflow.utils.logging_utils import eprint

import click
import sys

from mlflow.tracking import _get_store
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.proto_json_utils import message_to_json

RUN_ID_HELP = (
    "Run ID of MLflow run, which will use this to find the base artifact URI to use."
    " Exactly one of --run-id and --base-artifact-uri is required."
)

BASE_ARTIFACT_URI_HELP = (
    "Base artifact URI for an MLflow Run (e.g., s3://bucket/0/12345/artifacts)."
    " Exactly one of --base-artifact-uri and --run-id and is required."
)

@click.group("artifacts")
def commands():
    """Upload, list, and download artifacts from an MLflow artifact repository."""
    pass


@commands.command("log-artifact")
@click.option("--local-file", "-l", required=True,
              help="Local path to artifact to log")
@click.option("--run-id", "-r", help=RUN_ID_HELP)
@click.option("--base-artifact-uri", "-b", help=BASE_ARTIFACT_URI_HELP)
@click.option("--artifact-path", "-a",
              help="If specified, we will log the artifact into this subdirectory of the " +
                   "run's artifact directory.")
def log_artifact(local_file, run_id, base_artifact_uri, artifact_path):
    """
    Logs a local file as an artifact of a run, optionally within a run-specific
    artifact path. Run artifacts can be organized into directories, so you can
    place the artifact in a directory this way.
    """
    store = _get_store()
    artifact_uri = _get_artifact_uri_or_exit(store, run_id, base_artifact_uri)
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    artifact_repo.log_artifact(local_file, artifact_path)
    eprint("Logged artifact from local file %s to artifact_path=%s" % (local_file, artifact_path))


@commands.command("log-artifacts")
@click.option("--local-dir", "-l", required=True,
              help="Directory of local artifacts to log")
@click.option("--run-id", "-r", help=RUN_ID_HELP)
@click.option("--base-artifact-uri", "-b", help=BASE_ARTIFACT_URI_HELP)
@click.option("--artifact-path", "-a",
              help="If specified, we will log the artifact into this subdirectory of the " +
                   "run's artifact directory.")
def log_artifacts(local_dir, run_id, base_artifact_uri, artifact_path):
    """
    Logs the files within a local directory as an artifact of a run, optionally
    within a run-specific artifact path. Run artifacts can be organized into
    directories, so you can place the artifact in a directory this way.
    """
    store = _get_store()
    artifact_uri = _get_artifact_uri_or_exit(store, run_id, base_artifact_uri)
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    artifact_repo.log_artifacts(local_dir, artifact_path)
    eprint("Logged artifact from local dir %s to artifact_path=%s" % (local_dir, artifact_path))


@commands.command("list")
@click.option("--run-id", "-r", help=RUN_ID_HELP)
@click.option("--base-artifact-uri", "-b", help=BASE_ARTIFACT_URI_HELP)
@click.option("--artifact-path", "-a",
              help="If specified, a path relative to the run's root directory to list.")
def list_artifacts(run_id, base_artifact_uri, artifact_path):
    """
    Return all the artifacts directly under run's root artifact directory,
    or a sub-directory. The output is a JSON-formatted list.
    """
    artifact_path = artifact_path if artifact_path is not None else ""
    store = _get_store()
    artifact_uri = _get_artifact_uri_or_exit(store, run_id, base_artifact_uri)
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    file_infos = artifact_repo.list_artifacts(artifact_path)
    print(_file_infos_to_json(file_infos))


def _file_infos_to_json(file_infos):
    json_list = [message_to_json(file_info.to_proto()) for file_info in file_infos]
    return "[" + ", ".join(json_list) + "]"


@commands.command("download")
@click.option("--run-id", "-r", help=RUN_ID_HELP)
@click.option("--base-artifact-uri", "-b", help=BASE_ARTIFACT_URI_HELP)
@click.option("--artifact-path", "-a",
              help="If specified, a path relative to the run's root directory to download")
def download_artifacts(run_id, base_artifact_uri, artifact_path):
    """
    Download an artifact file or directory to a local directory.
    The output is the name of the file or directory on the local disk.
    """
    artifact_path = artifact_path if artifact_path is not None else ""
    store = _get_store()
    artifact_uri = _get_artifact_uri_or_exit(store, run_id, base_artifact_uri)
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    artifact_location = artifact_repo.download_artifacts(artifact_path)
    print(artifact_location)

def _get_artifact_uri_or_exit(store, run_id, base_artifact_uri):
    """Returns an appropriate artifact_uri by looking at the run_id or provided URI."""
    if run_id and base_artifact_uri:
      eprint("Only one of --run-id and --base-artifact-uri may be provided")
      sys.exit(1)
    if not run_id and not base_artifact_uri:
      eprint("One of --run-id and --base-artifact-uri must be provided")
      sys.exit(1)
    if base_artifact_uri:
      return base_artifact_uri
    return store.get_run(run_id).info.artifact_uri
