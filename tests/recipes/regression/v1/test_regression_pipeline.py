import os

import pytest

from mlflow.exceptions import MlflowException
from mlflow.recipes.regression.v1.recipe import RegressionRecipe


@pytest.fixture
def create_recipe(enter_recipe_example_directory):
    recipe_root_path = enter_recipe_example_directory
    profile = "local"
    return RegressionRecipe(recipe_root_path=recipe_root_path, profile=profile)


def test_create_recipe_works(enter_recipe_example_directory):
    recipe_root_path = enter_recipe_example_directory
    recipe_name = os.path.basename(recipe_root_path)
    profile = "local"
    r = RegressionRecipe(recipe_root_path=recipe_root_path, profile=profile)
    assert r.name == recipe_name
    assert r.profile == profile


@pytest.mark.parametrize(
    ("recipe_name", "profile"),
    [("name_a", "local"), ("", "local"), ("sklearn_regression_example", "profile_a")],
)
def test_create_recipe_fails_with_invalid_input(
    recipe_name, profile, enter_recipe_example_directory
):
    recipe_root_path = os.path.join(os.path.dirname(enter_recipe_example_directory), recipe_name)
    with pytest.raises(
        MlflowException,
        match=r"(Failed to find|Did not find the YAML configuration)",
    ):
        RegressionRecipe(recipe_root_path=recipe_root_path, profile=profile)


def test_recipe_run_and_clean_the_whole_recipe_works(create_recipe):
    r = create_recipe
    r.run()
    r.clean()


@pytest.mark.parametrize("step", ["ingest", "split", "transform", "train", "evaluate", "register"])
def test_recipe_run_and_clean_individual_step_works(step, create_recipe):
    r = create_recipe
    r.run(step)
    r.clean(step)


def test_get_subgraph_for_target_step(create_recipe):
    r = create_recipe
    train_subgraph = r._get_subgraph_for_target_step(r._get_step("ingest"))
    for step, expected_step_name in zip(
        train_subgraph, ["ingest", "split", "transform", "train", "evaluate", "register"]
    ):
        assert step.name == expected_step_name
    for step in ["split", "transform", "train", "evaluate", "register"]:
        target_step = r._get_step(step)
        assert r._get_subgraph_for_target_step(target_step) == train_subgraph

    scoring_subgraph = r._get_subgraph_for_target_step(r._get_step("ingest_scoring"))
    for step, expected_step_name in zip(scoring_subgraph, ["ingest_scoring", "predict"]):
        assert step.name == expected_step_name
    for step in ["predict"]:
        target_step = r._get_step(step)
        assert r._get_subgraph_for_target_step(target_step) == scoring_subgraph
