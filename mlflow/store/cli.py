from mlflow.utils.logging_utils import eprint

import json
import os
import tempfile

import click

from tabulate import tabulate

from mlflow.data import is_uri
from mlflow.entities import ViewType
from mlflow.tracking import _get_store
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.proto_json_utils import message_to_json


@click.group("artifacts")
def commands():
    """Manage experiments."""
    pass


@commands.command("log-artifact")
@click.option("--local-file", "-l", required=True,
              help="Base location for runs to store artifact results. Artifacts will be stored ")
@click.option("--run-id", "-r", required=True)
@click.option("--artifact-path", "-a",
              help="Base location for runs to store artifact results. Artifacts will be stored ")
def log_artifact(local_file, run_id, artifact_path):
    """
    Creates a new experiment in the configured tracking server.
    """
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    artifact_repo.log_artifact(local_file, artifact_path)
    eprint("Logged artifact from local file '%s' to '%s'" % (local_file, artifact_path))


@commands.command("log-artifacts")
@click.option("--local-dir", "-l", required=True,
              help="Base location for runs to store artifact results. Artifacts will be stored ")
@click.option("--run-id", "-r", required=True)
@click.option("--artifact-path", "-a",
              help="Base location for runs to store artifact results. Artifacts will be stored ")
def log_artifacts(local_dir, run_id, artifact_path):
    """
    Creates a new experiment in the configured tracking server.
    """
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    artifact_repo.log_artifacts(local_dir, artifact_path)
    eprint("Logged artifact from local file '%s' to '%s'" % (local_dir, artifact_path))


@commands.command("list")
@click.option("--run-id", "-r", required=True)
@click.option("--artifact-path", "-a",
              help="Base location for runs to store artifact results. Artifacts will be stored ")
def list_artifacts(run_id, artifact_path):
    """
    Creates a new experiment in the configured tracking server.
    """
    artifact_path = artifact_path if artifact_path is not None else ""
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    file_infos = artifact_repo.list_artifacts(artifact_path)
    print(_file_infos_to_json(file_infos))


def _file_infos_to_json(file_infos):
  json_list = [message_to_json(file_info.to_proto()) for file_info in file_infos]
  return "[" + ", ".join(json_list) + "]"


@commands.command("download-artifacts")
@click.option("--run-id", "-r", required=True)
@click.option("--artifact-path", "-a",
              help="Base location for runs to store artifact results. Artifacts will be stored ")
def download_artifacts(run_id, artifact_path):
    """
    Creates a new experiment in the configured tracking server.
    """
    artifact_path = artifact_path if artifact_path is not None else ""
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, store)
    artifact_location = artifact_repo.download_artifacts(artifact_path)
    print(artifact_location)
