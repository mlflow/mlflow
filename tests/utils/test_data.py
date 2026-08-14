import os

from mlflow.projects import _project_spec
from mlflow.utils.data_utils import is_uri

TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")


def load_project():
    return _project_spec.load_project(directory=TEST_PROJECT_DIR)


def test_is_uri():
    assert is_uri("s3://some/s3/path")
    assert is_uri("gs://some/gs/path")
    assert is_uri("dbfs:/some/dbfs/path")
    assert is_uri("file://some/local/path")
    assert not is_uri("/tmp/some/local/path")
