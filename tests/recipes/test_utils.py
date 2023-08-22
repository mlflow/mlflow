import json
import os
import pathlib
import shutil
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.recipes.utils import (
    get_default_profile,
    get_recipe_config,
    get_recipe_name,
    get_recipe_root_path,
)
from mlflow.utils.file_utils import write_yaml

from tests.recipes.helper_functions import chdir


def test_get_recipe_root_path_returns_correctly_when_inside_recipe_directory(
    enter_recipe_example_directory,
):
    recipe_root_path = enter_recipe_example_directory
    assert get_recipe_root_path() == str(recipe_root_path)
    os.chdir(pathlib.Path.cwd() / "notebooks")
    assert get_recipe_root_path() == str(enter_recipe_example_directory)


def test_get_recipe_root_path_throws_outside_recipe_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find recipe.yaml"), chdir(tmp_path):
        get_recipe_root_path()


def test_get_recipe_name_returns_correctly_for_valid_recipe_directory(
    enter_recipe_example_directory, tmp_path
):
    recipe_root_path = enter_recipe_example_directory
    assert pathlib.Path.cwd() == recipe_root_path
    assert get_recipe_name() == "regression"

    with chdir(tmp_path):
        assert get_recipe_name(recipe_root_path=recipe_root_path) == "regression"


def test_get_recipe_name_throws_for_invalid_recipe_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find recipe.yaml"), chdir(tmp_path):
        get_recipe_name()

    with pytest.raises(MlflowException, match="Failed to find recipe.yaml"):
        get_recipe_name(recipe_root_path=tmp_path)


def test_get_recipe_config_returns_correctly_for_valid_recipe_directory(
    enter_recipe_example_directory, tmp_path
):
    recipe_root_path = enter_recipe_example_directory
    test_recipe_root_path = tmp_path / "test_recipe"
    shutil.copytree(recipe_root_path, test_recipe_root_path)

    test_recipe_config = {
        "config1": 10,
        "config2": {
            "subconfig": ["A"],
        },
        "config3": "3",
    }
    write_yaml(test_recipe_root_path, "recipe.yaml", test_recipe_config, overwrite=True)

    with chdir(test_recipe_root_path):
        assert pathlib.Path.cwd() == test_recipe_root_path
        assert get_recipe_config() == test_recipe_config

    with chdir(tmp_path):
        assert get_recipe_config(recipe_root_path=test_recipe_root_path) == test_recipe_config


def test_get_recipe_config_for_recipe_directory_referencing_external_json(
    enter_recipe_example_directory, tmp_path
):
    recipe_root_path = enter_recipe_example_directory
    test_recipe_root_path = tmp_path / "test_recipe"
    shutil.copytree(recipe_root_path, test_recipe_root_path)

    test_recipe_config = {
        "config1": 10,
        "config2": {
            "subconfig": ["A"],
        },
        "config3": "3",
        "config_from_profile": "{{PROFILE_CONFIG}}",
    }
    write_yaml(test_recipe_root_path, "recipe.yaml", test_recipe_config, overwrite=True)

    # Write a profiles/test.yaml file that references keys in an external JSON file
    test_profile_name = "test"
    profile_contents = {
        "PROFILE_CONFIG": """{{ ("example-json.json" | from_json)["my-key"]["nested-key"] }}"""
    }
    profiles_dir = os.path.join(test_recipe_root_path, "profiles")
    if not os.path.exists(profiles_dir):
        os.mkdir(profiles_dir)
    write_yaml(
        test_recipe_root_path,
        f"profiles/{test_profile_name}.yaml",
        profile_contents,
        overwrite=True,
    )

    # Write the external JSON file
    value_from_json = "my-value-from-external-json"
    example_json = {
        "my-key": {
            "nested-key": value_from_json,
        },
    }
    with open(os.path.join(test_recipe_root_path, "example-json.json"), "w") as handle:
        handle.write(json.dumps(example_json))
    # Load recipe config and assert correctness
    expected_config = {}
    expected_config.update(test_recipe_config)
    expected_config.update(
        {
            "config_from_profile": value_from_json,
            "PROFILE_CONFIG": value_from_json,
        }
    )
    with chdir(test_recipe_root_path):
        assert get_recipe_config(profile=test_profile_name) == expected_config


def test_get_recipe_config_throws_for_invalid_recipe_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find recipe.yaml"), chdir(tmp_path):
        get_recipe_config()

    with pytest.raises(MlflowException, match="Failed to find recipe.yaml"):
        get_recipe_config(recipe_root_path=tmp_path)


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_get_recipe_config_throws_for_nonexistent_profile():
    with pytest.raises(MlflowException, match="Did not find the YAML configuration.*badprofile"):
        get_recipe_config(profile="badprofile")


def test_get_default_profile_works():
    assert get_default_profile() == "local"
    with mock.patch(
        "mlflow.recipes.utils.is_in_databricks_runtime", return_value=True
    ) as patched_is_in_databricks_runtime:
        assert get_default_profile() == "databricks"
        patched_is_in_databricks_runtime.assert_called_once()


@pytest.mark.usefixtures("enter_test_recipe_directory")
@pytest.mark.parametrize("get_config_from_subdirectory", [False, True])
def test_get_recipe_config_succeeds_for_valid_profile(get_config_from_subdirectory):
    if get_config_from_subdirectory:
        with chdir("notebooks"):
            get_recipe_config(profile="local")
    else:
        get_recipe_config(profile="local")


@pytest.mark.usefixtures("enter_test_recipe_directory")
def test_get_recipe_config_throws_clear_error_for_invalid_profile():
    with open("profiles/badcontent.yaml", "w") as f:
        f.write(r"\BAD")

    with pytest.raises(
        MlflowException,
        match="Failed to read recipe configuration.*verify.*syntactically correct",
    ):
        get_recipe_config(profile="badcontent")
