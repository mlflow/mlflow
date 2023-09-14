import os
import pathlib
import shutil
import sys
from pathlib import Path

import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_DIRECTORY
from mlflow.utils.file_utils import TempDir

from tests.recipes.helper_functions import (
    RECIPE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS,
    RECIPE_EXAMPLE_PATH_FROM_MLFLOW_ROOT,
    chdir,
)


@pytest.fixture
def enter_recipe_example_directory():
    recipe_example_path = os.environ.get(RECIPE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS)
    if recipe_example_path is None:
        mlflow_repo_root_directory = pathlib.Path(mlflow.__file__).parent.parent
        recipe_example_path = mlflow_repo_root_directory / RECIPE_EXAMPLE_PATH_FROM_MLFLOW_ROOT

    with chdir(recipe_example_path):
        yield recipe_example_path


@pytest.fixture
def enter_test_recipe_directory(enter_recipe_example_directory):
    recipe_example_root_path = enter_recipe_example_directory

    with TempDir(chdr=True) as tmp:
        test_recipe_path = tmp.path("test_recipe")
        shutil.copytree(recipe_example_root_path, test_recipe_path)
        os.chdir(test_recipe_path)
        yield os.getcwd()


@pytest.fixture
def tmp_recipe_exec_path(monkeypatch, tmp_path) -> Path:
    path = tmp_path.joinpath("recipe_execution")
    path.mkdir(parents=True)
    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(path))
    yield path
    shutil.rmtree(path)


@pytest.fixture
def tmp_recipe_root_path(tmp_path) -> Path:
    path = tmp_path.joinpath("recipe_root")
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def clear_custom_metrics_module_cache():
    key = "steps.custom_metrics"
    if key in sys.modules:
        del sys.modules[key]


@pytest.fixture
def registry_uri_path(tmp_path) -> Path:
    path = tmp_path.joinpath("registry.db")
    db_url = f"sqlite:///{path}"
    yield db_url
    mlflow.set_registry_uri("")
