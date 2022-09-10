import os
import pathlib
import shutil

import pytest

from mlflow.exceptions import MlflowException
from mlflow.pipelines.utils import (
    get_pipeline_root_path,
    get_pipeline_name,
    get_pipeline_config,
    get_default_profile,
)
from mlflow.utils.file_utils import write_yaml

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
    enter_test_pipeline_directory,
)  # pylint: enable=unused-import
from tests.pipelines.helper_functions import chdir
from unittest import mock


def test_get_pipeline_root_path_returns_correctly_when_inside_pipeline_directory(
    enter_pipeline_example_directory,
):
    pipeline_root_path = enter_pipeline_example_directory
    assert get_pipeline_root_path() == str(pipeline_root_path)
    os.chdir(pathlib.Path.cwd() / "notebooks")
    assert get_pipeline_root_path() == str(enter_pipeline_example_directory)


def test_get_pipeline_root_path_throws_outside_pipeline_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"), chdir(tmp_path):
        get_pipeline_root_path()


def test_get_pipeline_name_returns_correctly_for_valid_pipeline_directory(
    enter_pipeline_example_directory, tmp_path
):
    pipeline_root_path = enter_pipeline_example_directory
    assert pathlib.Path.cwd() == pipeline_root_path
    assert get_pipeline_name() == "sklearn_regression"

    with chdir(tmp_path):
        assert get_pipeline_name(pipeline_root_path=pipeline_root_path) == "sklearn_regression"


def test_get_pipeline_name_throws_for_invalid_pipeline_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"), chdir(tmp_path):
        get_pipeline_name()

    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"):
        get_pipeline_name(pipeline_root_path=tmp_path)


def test_get_pipeline_config_returns_correctly_for_valid_pipeline_directory(
    enter_pipeline_example_directory, tmp_path
):
    pipeline_root_path = enter_pipeline_example_directory
    test_pipeline_root_path = tmp_path / "test_pipeline"
    shutil.copytree(pipeline_root_path, test_pipeline_root_path)

    test_pipeline_config = {
        "config1": 10,
        "config2": {
            "subconfig": ["A"],
        },
        "config3": "3",
    }
    write_yaml(test_pipeline_root_path, "pipeline.yaml", test_pipeline_config, overwrite=True)

    with chdir(test_pipeline_root_path):
        assert pathlib.Path.cwd() == test_pipeline_root_path
        assert get_pipeline_config() == test_pipeline_config

    with chdir(tmp_path):
        assert (
            get_pipeline_config(pipeline_root_path=test_pipeline_root_path) == test_pipeline_config
        )


def test_get_pipeline_config_throws_for_invalid_pipeline_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"), chdir(tmp_path):
        get_pipeline_config()

    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"):
        get_pipeline_config(pipeline_root_path=tmp_path)


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_get_pipeline_config_throws_for_nonexistent_profile():
    with pytest.raises(MlflowException, match="Did not find the YAML configuration.*badprofile"):
        get_pipeline_config(profile="badprofile")


def test_get_default_profile_works():
    assert get_default_profile() == "local"
    with mock.patch(
        "mlflow.pipelines.utils.is_in_databricks_runtime", return_value=True
    ) as patched_is_in_databricks_runtime:
        assert get_default_profile() == "databricks"
        patched_is_in_databricks_runtime.assert_called_once()


@pytest.mark.usefixtures("enter_test_pipeline_directory")
@pytest.mark.parametrize("get_config_from_subdirectory", [False, True])
def test_get_pipeline_config_succeeds_for_valid_profile(get_config_from_subdirectory):
    if get_config_from_subdirectory:
        with chdir("notebooks"):
            get_pipeline_config(profile="local")
    else:
        get_pipeline_config(profile="local")


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_get_pipeline_config_throws_clear_error_for_invalid_profile():
    with open("profiles/badcontent.yaml", "w") as f:
        f.write(r"\BAD")

    with pytest.raises(
        MlflowException,
        match="Failed to read pipeline configuration.*verify.*syntactically correct",
    ):
        get_pipeline_config(profile="badcontent")
