import mock
import os

import pytest
import yaml

from mlflow.entities.run_status import RunStatus
from mlflow.projects import Project


TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")
GIT_PROJECT_URI = "https://github.com/databricks/mlflow-example"


def load_project():
    """ Loads an example project for use in tests, returning an in-memory `Project` object. """
    with open(os.path.join(TEST_PROJECT_DIR, "MLproject")) as mlproject_file:
        project_yaml = yaml.safe_load(mlproject_file.read())
    return Project(uri=TEST_PROJECT_DIR, yaml_obj=project_yaml)


def validate_exit_status(status_str, expected):
    assert RunStatus.from_string(status_str) == expected


@pytest.fixture()
def tracking_uri_mock(tmpdir):
    with mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = str(tmpdir)
        yield get_tracking_uri_mock
