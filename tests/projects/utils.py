import filecmp
import os


import pytest

from mlflow.utils.file_utils import TempDir, _copy_project

from mlflow.entities import RunStatus
from mlflow.projects import _project_spec


TEST_DIR = "tests"
TEST_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project")
TEST_DOCKER_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_docker_project")
TEST_PROJECT_NAME = "example_project"
TEST_NO_SPEC_PROJECT_DIR = os.path.join(TEST_DIR, "resources", "example_project_no_spec")
GIT_PROJECT_URI = "https://github.com/mlflow/mlflow-example"
GIT_PROJECT_BRANCH = "test-branch"
SSH_PROJECT_URI = "git@github.com:mlflow/mlflow-example.git"


def load_project():
    """Loads an example project for use in tests, returning an in-memory `Project` object."""
    return _project_spec.load_project(TEST_PROJECT_DIR)


def validate_exit_status(status_str, expected):
    assert RunStatus.from_string(status_str) == expected


def assert_dirs_equal(expected, actual):
    dir_comparison = filecmp.dircmp(expected, actual)
    assert len(dir_comparison.left_only) == 0
    assert len(dir_comparison.right_only) == 0
    assert len(dir_comparison.diff_files) == 0
    assert len(dir_comparison.funny_files) == 0


@pytest.fixture(scope="session")
def docker_example_base_image():
    import docker
    from docker.errors import BuildError, APIError

    mlflow_home = os.environ.get("MLFLOW_HOME", None)
    if not mlflow_home:
        raise Exception(
            "MLFLOW_HOME environment variable is not set. Please set the variable to "
            "point to your mlflow dev root."
        )
    with TempDir() as tmp:
        cwd = tmp.path()
        mlflow_dir = _copy_project(src_path=mlflow_home, dst_path=cwd)
        import shutil

        shutil.copy(os.path.join(TEST_DOCKER_PROJECT_DIR, "Dockerfile"), tmp.path("Dockerfile"))
        with open(tmp.path("Dockerfile"), "a") as f:
            f.write(
                ("COPY {mlflow_dir} /opt/mlflow\n" "RUN pip install -U -e /opt/mlflow\n").format(
                    mlflow_dir=mlflow_dir
                )
            )

        client = docker.from_env()
        try:
            client.images.build(
                tag="mlflow-docker-example",
                forcerm=True,
                nocache=True,
                dockerfile="Dockerfile",
                path=cwd,
            )
        except BuildError as build_error:
            for chunk in build_error.build_log:
                print(chunk)
            raise build_error
        except APIError as api_error:
            print(api_error.explanation)
            raise api_error
