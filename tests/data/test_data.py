from contextlib import contextmanager
import os
import shutil
import tempfile

import mock
import pytest

from mlflow.data import is_uri, download_uri
from mlflow.projects import _project_spec

TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")


def load_project():
    return _project_spec.load_project(directory=TEST_PROJECT_DIR)


def test_is_uri():
    assert is_uri("s3://some/s3/path")
    assert is_uri("dbfs:/some/dbfs/path")
    assert is_uri("file://some/local/path")
    assert not is_uri("/tmp/some/local/path")


def test_download_uri_uses_artifact_repo_to_download_artifacts(tmpdir):
    uri_artifact_repo_map = {
        "/path/to/file": "mlflow.store.local_artifact_repo.LocalArtifactRepository",
        "s3://path/to/file": "mlflow.store.s3_artifact_repo.S3ArtifactRepository",
        "gs://path/to/file": "mlflow.store.gcs_artifact_repo.GCSArtifactRepository",
        "dbfs://path/to/file": "mlflow.store.dbfs_artifact_repo.DbfsArtifactRepository",
    }

    for uri, artifact_repo_module in uri_artifact_repo_map.items():
        with mock.patch("{artifact_repo_module}.download_artifacts".format(
                artifact_repo_module=artifact_repo_module)) as artifact_repo_download_mock:
            download_uri(uri=uri, output_path=str(tmpdir))
            artifact_repo_download_mock.assert_called_once()
            # assert artifact_repo_download_mock.call_args.artifact_path in uri
