import os
import pathlib
import re
import shutil
from unittest import mock

import pandas as pd
import pytest
import yaml

import mlflow
from mlflow.entities import Run, SourceType
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.recipes.recipe import Recipe
from mlflow.recipes.step import BaseStep
from mlflow.recipes.utils.execution import (
    _MAKEFILE_FORMAT_STRING,
    _get_execution_directory_basename,
    get_step_output_path,
)
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.mlflow_tags import (
    LEGACY_MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
)

from tests.recipes.helper_functions import chdir, list_all_artifacts

# _STEP_NAMES must contain all step names that are expected to be executed when
# `recipe.run(step=None)` is called
_STEP_NAMES = ["ingest", "split", "transform", "train", "evaluate", "register"]


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_create_recipe_fails_with_invalid_profile():
    with pytest.raises(
        MlflowException,
        match=r"(Failed to find|Did not find the YAML configuration)",
    ):
        Recipe(profile="local123")


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_create_recipe_and_clean_works():
    r = Recipe(profile="local")
    r.clean()


@pytest.mark.usefixtures("enter_recipe_example_directory")
@pytest.mark.parametrize("empty_profile", [None, ""])
def test_create_recipe_fails_with_empty_profile_name(empty_profile):
    with pytest.raises(MlflowException, match="A profile name must be provided"):
        Recipe(profile=empty_profile)


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_create_recipe_fails_with_path_containing_space(tmp_path):
    space_parent = tmp_path / "space parent"
    space_path = space_parent / "child"
    os.makedirs(space_parent, exist_ok=True)
    os.makedirs(space_path, exist_ok=True)
    shutil.copytree(src=os.getcwd(), dst=str(space_path), dirs_exist_ok=True)

    with chdir(space_path), pytest.raises(
        MlflowException, match="Recipe directory path cannot contain spaces"
    ):
        Recipe(profile="local")


@pytest.mark.usefixtures("enter_recipe_example_directory")
@pytest.mark.parametrize("custom_execution_directory", [None, "custom"])
def test_recipes_execution_directory_is_managed_as_expected(
    custom_execution_directory, enter_recipe_example_directory, tmp_path
):
    if custom_execution_directory is not None:
        custom_execution_directory = tmp_path / custom_execution_directory

    if custom_execution_directory is not None:
        os.environ["MLFLOW_RECIPES_EXECUTION_DIRECTORY"] = str(custom_execution_directory)

    expected_execution_directory_location = (
        pathlib.Path(custom_execution_directory)
        if custom_execution_directory
        else pathlib.Path.home()
        / ".mlflow"
        / "recipes"
        / _get_execution_directory_basename(enter_recipe_example_directory)
    )

    # Run the full recipe and verify that outputs for each step were written to the expected
    # execution directory locations
    r = Recipe(profile="local")
    r.run()
    assert (expected_execution_directory_location / "Makefile").exists()
    assert (expected_execution_directory_location / "steps").exists()
    for step_name in _STEP_NAMES:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert step_outputs_path.exists()
        first_output = next(step_outputs_path.iterdir(), None)
        assert first_output is not None

    # Clean the recipe and verify that all step outputs have been removed
    r.clean()
    for step_name in _STEP_NAMES:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert not list(step_outputs_path.iterdir())


@pytest.mark.usefixtures("enter_test_recipe_directory")
def test_recipes_log_to_expected_mlflow_backend_with_expected_run_tags_once_on_reruns(
    tmp_path,
):
    experiment_name = "my_test_exp"
    tracking_uri = "sqlite:///" + str((tmp_path / "tracking_dst.db").resolve())
    artifact_location = str((tmp_path / "mlartifacts").resolve())

    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path) as f:
        profile_contents = yaml.safe_load(f)

    profile_contents["experiment"]["name"] = experiment_name
    profile_contents["experiment"]["tracking_uri"] = tracking_uri
    profile_contents["experiment"]["artifact_location"] = path_to_local_file_uri(artifact_location)

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    mlflow.set_tracking_uri(tracking_uri)
    recipe = Recipe(profile="local")
    recipe.clean()
    recipe.run()

    logged_runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
    assert len(logged_runs) == 1
    logged_run = logged_runs[0]
    assert logged_run.info.artifact_uri == path_to_local_file_uri(
        str((pathlib.Path(artifact_location) / logged_run.info.run_id / "artifacts").resolve())
    )
    assert "test_r2_score" in logged_run.data.metrics
    artifacts = MlflowClient(tracking_uri).list_artifacts(
        run_id=logged_run.info.run_id, path="train"
    )
    assert {artifact.path for artifact in artifacts} == {
        "train/best_parameters.yaml",
        "train/card.html",
        "train/estimator",
        "train/model",
    }
    run_tags = MlflowClient(tracking_uri).get_run(run_id=logged_run.info.run_id).data.tags
    recipe_source_tag = {MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.RECIPE)}
    assert resolve_tags(recipe_source_tag).items() <= run_tags.items()

    recipe.run()
    logged_runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
    assert len(logged_runs) == 1


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_recipes_run_sets_mlflow_git_tags():
    recipe = Recipe(profile="local")
    recipe.clean()
    recipe.run(step="train")

    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path) as f:
        profile_contents = yaml.safe_load(f)

    tracking_uri = profile_contents["experiment"]["tracking_uri"]
    experiment_name = profile_contents["experiment"]["name"]

    mlflow.set_tracking_uri(tracking_uri)
    logged_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        output_format="list",
        order_by=["attributes.start_time DESC"],
    )
    logged_run = logged_runs[0]

    run_tags = MlflowClient(tracking_uri).get_run(run_id=logged_run.info.run_id).data.tags
    assert {
        MLFLOW_SOURCE_NAME,
        MLFLOW_GIT_COMMIT,
        MLFLOW_GIT_REPO_URL,
        LEGACY_MLFLOW_GIT_REPO_URL,
    } < run_tags.keys()
    assert run_tags[MLFLOW_SOURCE_NAME] == run_tags[MLFLOW_GIT_REPO_URL]


@pytest.mark.usefixtures("enter_test_recipe_directory")
def test_recipes_run_throws_exception_and_produces_failure_card_when_step_fails():
    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path) as f:
        profile_contents = yaml.safe_load(f)

    profile_contents["INGEST_CONFIG"] = {"using": "parquet", "location": "a bad location"}

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    recipe = Recipe(profile="local")
    recipe.clean()
    with pytest.raises(
        MlflowException, match="Failed to run recipe.*test_recipe.*\n.*Step:ingest.*"
    ):
        recipe.run()
    with pytest.raises(MlflowException, match="Failed to run step.*split.*\n.*Step:ingest.*"):
        recipe.run(step="split")

    step_card_path = get_step_output_path(
        recipe_root_path=recipe._recipe_root_path,
        step_name="ingest",
        relative_path="card.html",
    )
    with open(step_card_path) as f:
        card_content = f.read()

    assert "Ingest" in card_content
    assert "Failed" in card_content
    assert "Stacktrace" in card_content


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_test_step_logs_step_cards_as_artifacts():
    recipe = Recipe(profile="local")
    recipe.clean()
    recipe.run()

    tracking_uri = recipe._get_step("train").tracking_config.tracking_uri
    local_run_id_path = get_step_output_path(
        recipe_root_path=recipe._recipe_root_path,
        step_name="train",
        relative_path="run_id",
    )
    run_id = pathlib.Path(local_run_id_path).read_text()
    artifacts = set(list_all_artifacts(tracking_uri, run_id))
    assert artifacts.issuperset(
        {
            "ingest/card.html",
            "split/card.html",
            "transform/card.html",
            "train/card.html",
            "evaluate/card.html",
            "register/card.html",
        }
    )


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_recipe_get_artifacts():
    recipe = Recipe(profile="local")
    recipe.clean()

    recipe.run("ingest")
    recipe.run("split")
    recipe.run("transform")
    recipe.run("train")
    recipe.run("register")

    assert isinstance(recipe.get_artifact("ingested_data"), pd.DataFrame)
    assert isinstance(recipe.get_artifact("training_data"), pd.DataFrame)
    assert isinstance(recipe.get_artifact("validation_data"), pd.DataFrame)
    assert isinstance(recipe.get_artifact("test_data"), pd.DataFrame)
    assert isinstance(recipe.get_artifact("transformed_training_data"), pd.DataFrame)
    assert isinstance(recipe.get_artifact("transformed_validation_data"), pd.DataFrame)
    assert hasattr(recipe.get_artifact("transformer"), "transform")
    assert isinstance(recipe.get_artifact("model"), mlflow.pyfunc.PyFuncModel)
    assert isinstance(recipe.get_artifact("run"), Run)
    assert isinstance(recipe.get_artifact("registered_model_version"), ModelVersion)

    with pytest.raises(MlflowException, match="The artifact with name 'abcde' is not supported."):
        recipe.get_artifact("abcde")

    recipe.clean()
    assert not recipe.get_artifact("ingested_data")


def test_generate_worst_examples_dataframe():
    test_df = pd.DataFrame(
        {
            "a": [3, 2, 5],
            "b": [6, 9, 1],
        }
    )
    target_col = "b"
    predictions = [5, 3, 4]

    result_df = BaseStep._generate_worst_examples_dataframe(
        test_df, predictions, predictions - test_df[target_col].to_numpy(), target_col, worst_k=2
    )

    def assert_result_correct(df):
        assert df.columns.tolist() == ["absolute_error", "prediction", "b", "a"]
        assert df.absolute_error.tolist() == [6, 3]
        assert df.prediction.tolist() == [3, 4]
        assert df.b.tolist() == [9, 1]
        assert df.a.tolist() == [2, 5]
        assert df.index.tolist() == [0, 1]

    assert_result_correct(result_df)

    test_df2 = test_df.set_axis([2, 1, 0], axis="index")
    result_df2 = BaseStep._generate_worst_examples_dataframe(
        test_df2, predictions, predictions - test_df2[target_col].to_numpy(), target_col, worst_k=2
    )
    assert_result_correct(result_df2)


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_print_cached_steps_and_running_steps(capsys):
    recipe = Recipe(profile="local")
    recipe.clean()
    recipe.run()
    captured = capsys.readouterr()
    output_info = captured.out
    run_step_pattern = "Running step {step}..."
    for step in _STEP_NAMES:
        # Check for printed message when every step is actually executed
        assert re.search(run_step_pattern.format(step=step), output_info) is not None

    recipe.run()  # cached
    captured = capsys.readouterr()
    output_info = captured.err
    cached_step_pattern = "{step}: No changes. Skipping."
    cached_steps = ", ".join(_STEP_NAMES)
    # Check for printed message when steps are cached
    assert re.search(cached_step_pattern.format(step=cached_steps), output_info) is not None


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_make_dry_run_error_does_not_print_cached_steps_messages(capsys):
    malformed_makefile = _MAKEFILE_FORMAT_STRING + "non_existing_cmd"
    with mock.patch(
        "mlflow.recipes.utils.execution._MAKEFILE_FORMAT_STRING",
        new=malformed_makefile,
    ):
        r = Recipe(profile="local")
        r.clean()
        try:
            r.run()
        except MlflowException:
            pass
        captured = capsys.readouterr()
        output_info = captured.out
        assert re.search(r"\*\*\* missing separator.  Stop.", output_info) is not None

        output_info = captured.err
        cached_step_pattern = "{step}: No changes. Skipping."
        for step in _STEP_NAMES:
            assert re.search(cached_step_pattern.format(step=step), output_info) is None


@pytest.mark.usefixtures("enter_recipe_example_directory")
def test_makefile_with_runtime_error_print_cached_steps_messages(capsys):
    split = 'echo "Run MLflow Recipe step: split"'
    tokens = _MAKEFILE_FORMAT_STRING.split(split)
    assert len(tokens) == 2
    tokens[1] = "\n\tnon-existing-cmd" + tokens[1]
    malformed_makefile_rte = split.join(tokens)

    with mock.patch(
        "mlflow.recipes.utils.execution._MAKEFILE_FORMAT_STRING",
        new=malformed_makefile_rte,
    ):
        r = Recipe(profile="local")
        r.clean()
        try:
            r.run(step="split")
        except MlflowException:
            pass
        captured = capsys.readouterr()
        output_info = captured.out
        # Runtime error occurs
        assert re.search(r"\*\*\*.+Error", output_info) is not None
        # ingest step is executed
        assert re.search("Running step ingest...", output_info) is not None
        # split step is not executed
        assert re.search("Running step split...", output_info) is None

        try:
            r.run(step="split")
        except MlflowException:
            pass
        captured = capsys.readouterr()
        output_info = captured.out
        # Runtime error occurs
        assert re.search(r"\*\*\*.+Error", output_info) is not None
        output_info = captured.err
        # ingest step is cached
        assert re.search("ingest: No changes. Skipping.", output_info) is not None
        # split step is not cached
        assert re.search("split: No changes. Skipping.", output_info) is None
