from contextlib import contextmanager
import os
import shutil
import tempfile

from mlflow.data import is_uri
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
    assert not is_uri("/tmp/some/local/path")
