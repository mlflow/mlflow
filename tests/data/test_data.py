from contextlib import contextmanager
import os
import shutil
import tempfile

import mock
import pytest

from mlflow.data import is_uri, download_uri, DownloadException
from mlflow.projects import _project_spec

TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")


@contextmanager
def temp_directory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def load_project():
    return _project_spec.load_project(directory=TEST_PROJECT_DIR)


def test_is_uri():
    assert is_uri("s3://some/s3/path")
    assert is_uri("gs://some/gs/path")
    assert is_uri("dbfs:/some/dbfs/path")
    assert is_uri("file://some/local/path")
    assert is_uri("hdfs://some/hdfs/path")
    assert is_uri("viewfs://some/viewfs/path")
    assert is_uri("har://some/viewfs/path")
    assert not is_uri("/tmp/some/local/path")


def test_download_uri():
    # Verify downloading from DBFS & S3 urls calls the corresponding helper functions
    prefix_to_mock = {
        "dbfs:/": "mlflow.data._fetch_dbfs",
        "s3://": "mlflow.data._fetch_s3",
        "gs://": "mlflow.data._fetch_gs",
        "hdfs://": "mlflow.data._fetch_hdfs",
        "viewfs://": "mlflow.data._fetch_hdfs",
    }
    for prefix, fn_name in prefix_to_mock.items():
        with mock.patch(fn_name) as mocked_fn, temp_directory() as dst_dir:
            download_uri(
                uri=os.path.join(prefix, "some/path"), output_path=os.path.join(dst_dir, "tmp-file")
            )
            assert mocked_fn.call_count == 1
    # Verify exceptions are thrown when downloading from unsupported/invalid URIs
    invalid_prefixes = ["file://", "/tmp"]
    for prefix in invalid_prefixes:
        with temp_directory() as dst_dir, pytest.raises(DownloadException):
            download_uri(
                uri=os.path.join(prefix, "some/path"), output_path=os.path.join(dst_dir, "tmp-file")
            )
