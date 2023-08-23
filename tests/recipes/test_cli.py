import pytest
from click.testing import CliRunner

import mlflow.recipes.cli as recipes_cli
from mlflow.environment_variables import MLFLOW_RECIPES_PROFILE
from mlflow.exceptions import MlflowException

_STEP_NAMES = ["ingest", "split", "train", "transform", "evaluate", "register"]


@pytest.fixture
def clean_up_recipe():
    try:
        yield
    finally:
        CliRunner().invoke(recipes_cli.clean, env={MLFLOW_RECIPES_PROFILE.name: "local"})


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
@pytest.mark.parametrize("step", _STEP_NAMES)
def test_recipes_cli_step_works(step):
    for command in [recipes_cli.clean, recipes_cli.inspect, recipes_cli.run]:
        assert (
            CliRunner()
            .invoke(command, args=f"--step {step}", env={MLFLOW_RECIPES_PROFILE.name: "local"})
            .exit_code
            == 0
        )
        assert CliRunner().invoke(command, args=f"--step {step} --profile=local").exit_code == 0


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_flow_completes_successfully():
    for command in [recipes_cli.clean, recipes_cli.inspect, recipes_cli.run]:
        assert (
            CliRunner().invoke(command, env={MLFLOW_RECIPES_PROFILE.name: "local"}).exit_code == 0
        )


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_fails_without_profile():
    for command in [recipes_cli.clean, recipes_cli.inspect, recipes_cli.run]:
        assert CliRunner().invoke(command, env={MLFLOW_RECIPES_PROFILE.name: ""}).exit_code != 0
        assert CliRunner().invoke(command).exit_code != 0


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_fails_with_illegal_profile():
    result = CliRunner().invoke(
        recipes_cli.clean, env={MLFLOW_RECIPES_PROFILE.name: "illegal_profile"}
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, MlflowException)


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_cli_works_with_non_default_profile():
    result = CliRunner().invoke(recipes_cli.clean, env={MLFLOW_RECIPES_PROFILE.name: "databricks"})
    assert "with profile: 'databricks'" in result.output
    assert result.exit_code == 0


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_get_artifact_works():
    result = CliRunner().invoke(
        recipes_cli.get_artifact,
        args="--artifact model",
        env={MLFLOW_RECIPES_PROFILE.name: "local"},
    )
    assert result.exit_code == 0
    assert result.output is not None


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_get_artifact_with_bad_artifact_name_fails():
    result = CliRunner().invoke(
        recipes_cli.get_artifact,
        args="--artifact foo",
        env={MLFLOW_RECIPES_PROFILE.name: "local"},
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, MlflowException)


@pytest.mark.usefixtures("enter_recipe_example_directory", "clean_up_recipe")
def test_recipes_get_artifact_with_no_artifact_name_fails():
    result = CliRunner().invoke(
        recipes_cli.get_artifact,
        env={MLFLOW_RECIPES_PROFILE.name: "local"},
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
