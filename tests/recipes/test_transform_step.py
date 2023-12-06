import os
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_DIRECTORY
from mlflow.exceptions import MlflowException
from mlflow.recipes.steps.transform import TransformStep, _validate_user_code_output
from mlflow.recipes.utils import _RECIPE_CONFIG_FILE_NAME
from mlflow.utils.file_utils import read_yaml


@pytest.fixture(autouse=True)
def dummy_transform_step(tmp_recipe_root_path, monkeypatch):
    # `mock.patch("steps.transform.transformer_fn", ...)` would fail without this fixture
    steps = tmp_recipe_root_path / "steps"
    steps.mkdir(exist_ok=True)
    steps.joinpath("transform.py").write_text(
        """
def transformer_fn(estimator_params=None):
    return None
"""
    )
    monkeypatch.syspath_prepend(str(tmp_recipe_root_path))


# Sets up the transform step and returns the constructed TransformStep instance and step output dir
def set_up_transform_step(recipe_root: Path, transform_user_module):
    split_step_output_dir = recipe_root.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)

    transform_step_output_dir = recipe_root.joinpath("steps", "transform", "outputs")
    transform_step_output_dir.mkdir(parents=True)

    # use for train and validation, also for split
    dataset = pd.DataFrame(
        {
            "a": list(range(0, 5)),
            "b": list(range(5, 10)),
            "y": [float(i % 2) for i in range(5)],
        }
    )
    dataset.to_parquet(str(split_step_output_dir / "validation.parquet"))
    dataset.to_parquet(str(split_step_output_dir / "train.parquet"))

    recipe_yaml = recipe_root.joinpath(_RECIPE_CONFIG_FILE_NAME)
    experiment_name = "demo"
    MlflowClient().create_experiment(experiment_name)

    recipe_yaml.write_text(
        f"""
        recipe: "regression/v1"
        target_col: "y"
        experiment:
          name: {experiment_name}
          tracking_uri: {mlflow.get_tracking_uri()}
        steps:
          transform:
            using: custom
            transformer_method: {transform_user_module}
        """
    )
    recipe_config = read_yaml(recipe_root, _RECIPE_CONFIG_FILE_NAME)
    transform_step = TransformStep.from_recipe_config(recipe_config, str(recipe_root))
    return transform_step, transform_step_output_dir, split_step_output_dir


def test_transform_step_writes_onehot_encoded_dataframe_and_transformer_pkl(
    tmp_recipe_root_path, monkeypatch
):
    from sklearn.preprocessing import StandardScaler

    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(tmp_recipe_root_path))
    with mock.patch("steps.transform.transformer_fn", lambda: StandardScaler()):
        transform_step, transform_step_output_dir, _ = set_up_transform_step(
            tmp_recipe_root_path, "transformer_fn"
        )
        transform_step.run(str(transform_step_output_dir))

    assert os.path.exists(transform_step_output_dir / "transformed_training_data.parquet")
    transformed = pd.read_parquet(transform_step_output_dir / "transformed_training_data.parquet")
    assert len(transformed.columns) == 3
    assert os.path.exists(transform_step_output_dir / "transformer.pkl")


@pytest.mark.parametrize("recipe", ["regression/v1", "classification/v1"])
def test_transform_steps_work_without_step_config(tmp_recipe_root_path, recipe):
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    experiment_name = "demo"
    MlflowClient().create_experiment(experiment_name)

    recipe_yaml.write_text(
        """
        recipe: {recipe}
        target_col: "y"
        {positive_class}
        experiment:
          name: {experiment_name}
          tracking_uri: {tracking_uri}
        steps:
          fakestep:
            something: else
        """.format(
            tracking_uri=mlflow.get_tracking_uri(),
            experiment_name=experiment_name,
            recipe=recipe,
            positive_class='positive_class: "a"' if recipe == "regression/v1" else "",
        )
    )
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    transform_step = TransformStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    transform_step._validate_and_apply_step_config()


def test_transform_empty_step(tmp_recipe_root_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_DIRECTORY.name, str(tmp_recipe_root_path))
    with mock.patch("steps.transform.transformer_fn", return_value=None):
        transform_step, transform_step_output_dir, split_step_output_dir = set_up_transform_step(
            tmp_recipe_root_path, "transformer_fn"
        )
        transform_step.run(str(transform_step_output_dir))

    assert os.path.exists(transform_step_output_dir / "transformed_training_data.parquet")
    train_transformed = pd.read_parquet(
        transform_step_output_dir / "transformed_training_data.parquet"
    )
    train_split = pd.read_parquet(split_step_output_dir / "train.parquet")

    assert train_transformed.equals(train_split) is True
    assert os.path.exists(transform_step_output_dir / "transformer.pkl")


def test_validate_method_validates_the_transformer():
    class Transformer:
        def fit(self):
            return "fit"

        def transform(self):
            return "transform"

    transformer = Transformer()

    def correct_transformer():
        return transformer

    validated_transformer = _validate_user_code_output(correct_transformer)
    assert transformer == validated_transformer

    class InCorrectFitTransformer:
        def pick(self):
            return "pick"

        def transform(self):
            return "transform"

    in_correct_fit_transformer = InCorrectFitTransformer()

    def incorrect__fit_transformer():
        return in_correct_fit_transformer

    with pytest.raises(
        MlflowException,
        match="The transformer provided doesn't have a fit method.",
    ):
        validated_transformer = _validate_user_code_output(incorrect__fit_transformer)

    class InCorrectTransformer:
        def pick(self):
            return "pick"

        def fit(self):
            return "fit"

    inCorrectTransformer = InCorrectTransformer()

    def incorrect_transformer():
        return inCorrectTransformer

    with pytest.raises(
        MlflowException,
        match="The transformer provided doesn't have a transform method.",
    ):
        validated_transformer = _validate_user_code_output(incorrect_transformer)
