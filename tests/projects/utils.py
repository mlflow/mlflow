import filecmp
import os

import pytest

import mlflow
from mlflow.entities import RunStatus
from mlflow.projects import _project_spec


TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")
TEST_PROJECT_NAME = "example_project"
TEST_NO_SPEC_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project_no_spec")
GIT_PROJECT_URI = "https://github.com/mlflow/mlflow-example"
SSH_PROJECT_URI = "git@github.com:mlflow/mlflow-example.git"


def load_project():
    """ Loads an example project for use in tests, returning an in-memory `Project` object. """
    return _project_spec.load_project(TEST_PROJECT_DIR)


def validate_exit_status(status_str, expected):
    assert RunStatus.from_string(status_str) == expected


def assert_dirs_equal(expected, actual):
    dir_comparison = filecmp.dircmp(expected, actual)
    assert len(dir_comparison.left_only) == 0
    assert len(dir_comparison.right_only) == 0
    assert len(dir_comparison.diff_files) == 0
    assert len(dir_comparison.funny_files) == 0


@pytest.fixture()
def tracking_uri_mock(tmpdir):
    try:
        mlflow.set_tracking_uri(os.path.join(tmpdir.strpath, 'mlruns'))
        yield tmpdir
    finally:
        mlflow.set_tracking_uri(None)
