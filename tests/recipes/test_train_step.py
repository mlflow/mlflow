import os
import sys
import cloudpickle
from pathlib import Path
import pytest

import pandas as pd

import mlflow
import sklearn.compose
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import read_yaml
from mlflow.recipes.utils.execution import (
    _MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR,
    _MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME_ENV_VAR,
)
from mlflow.recipes.utils import _RECIPE_CONFIG_FILE_NAME
from mlflow.recipes.steps.train import TrainStep
from unittest import mock
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_TYPE,
    MLFLOW_RECIPE_TEMPLATE_NAME,
    MLFLOW_RECIPE_PROFILE_NAME,
    MLFLOW_RECIPE_STEP_NAME,
)

# pylint: disable=unused-import
from tests.recipes.helper_functions import tmp_recipe_root_path

# pylint: enable=unused-import

# Sets up the train step output dir
def setup_train_dataset(recipe_root: Path, recipe: str = "regression"):
    split_step_output_dir = recipe_root.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)

    transform_step_output_dir = recipe_root.joinpath("steps", "transform", "outputs")
    transform_step_output_dir.mkdir(parents=True)

    train_step_output_dir = recipe_root.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)

    transformer = sklearn.preprocessing.FunctionTransformer(func=(lambda x: x))
    with open(os.path.join(transform_step_output_dir, "transformer.pkl"), "wb") as f:
        cloudpickle.dump(transformer, f)

    num_rows = 100
    # use for train and validation, also for split
    if recipe == "regression":
        transformed_dataset = pd.DataFrame(
            {
                "a": list(range(num_rows)),
                "b": list(range(num_rows)),
                "y": [float(i % 2) for i in range(num_rows)],
            }
        )
    else:
        import math
        import random

        minority_class_cnt = math.ceil(0.1 * num_rows)
        majority_class_cnt = num_rows - minority_class_cnt
        y = ["a"] * minority_class_cnt + ["b"] * majority_class_cnt
        random.shuffle(y)

        transformed_dataset = pd.DataFrame(
            {
                "a": list(range(num_rows)),
                "b": list(range(num_rows)),
                "y": y,
            }
        )
    transformed_dataset.to_parquet(
        str(transform_step_output_dir / "transformed_training_data.parquet")
    )
    transformed_dataset.to_parquet(
        str(transform_step_output_dir / "transformed_validation_data.parquet")
    )
    transformed_dataset.to_parquet(str(split_step_output_dir / "validation.parquet"))
    transformed_dataset.to_parquet(str(split_step_output_dir / "train.parquet"))

    return train_step_output_dir


# Sets up the constructed TrainStep instance
def setup_train_step_with_tuning(
    recipe_root: Path, use_tuning: bool, with_hardcoded_params: bool = True
):
    recipe_yaml = recipe_root.joinpath(_RECIPE_CONFIG_FILE_NAME)
    if with_hardcoded_params:
        estimator_params = """
                    estimator_params:
                        alpha: 0.1
                        penalty: l1
                        eta0: 0.1
                        fit_intercept: true
        """
    else:
        estimator_params = ""
    if use_tuning:
        recipe_yaml.write_text(
            """
            recipe: "regression/v1"
            target_col: "y"
            profile: "test_profile"
            run_args:
                step: "train"
            experiment:
                name: "demo"
                tracking_uri: {tracking_uri}
            steps:
                train:
                    using: custom
                    estimator_method: estimator_fn
                    {estimator_params}
                    tuning:
                        enabled: true
                        max_trials: 2
                        sample_fraction: 0.5
                        early_stop_fn: tests.recipes.test_train_step.early_stop_fn
                        parameters:
                            alpha:
                                distribution: "uniform"
                                low: 0.0
                                high: 0.01
                            penalty:
                                values: ["l2", "l1"]
                            eta0:
                                distribution: "normal"
                                mu: 0.01
                                sigma: 0.0001
            """.format(
                tracking_uri=mlflow.get_tracking_uri(), estimator_params=estimator_params
            )
        )
    else:
        recipe_yaml.write_text(
            """
            recipe: "regression/v1"
            target_col: "y"
            profile: "test_profile"
            run_args:
                step: "train"
            experiment:
                name: "demo"
                tracking_uri: {tracking_uri}
            steps:
                train:
                    using: custom
                    estimator_method: estimator_fn
                    tuning:
                        enabled: false
            """.format(
                tracking_uri=mlflow.get_tracking_uri()
            )
        )
    recipe_config = read_yaml(recipe_root, _RECIPE_CONFIG_FILE_NAME)
    train_step = TrainStep.from_recipe_config(recipe_config, str(recipe_root))
    return train_step


def test_train_step(tmp_recipe_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
        recipe_yaml.write_text(
            """
            recipe: "regression/v1"
            target_col: "y"
            profile: "test_profile"
            run_args:
                step: "train"
            experiment:
                name: "demo"
                tracking_uri: {tracking_uri}
            steps:
                train:
                    using: custom
                    estimator_method: estimator_fn
                    tuning:
                        enabled: false
            """.format(
                tracking_uri=mlflow.get_tracking_uri()
            )
        )
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
            train_step.run(str(train_step_output_dir))

    run_id = train_step_output_dir.joinpath("run_id").read_text()
    metrics = MlflowClient().get_run(run_id).data.metrics
    assert "val_mean_squared_error" in metrics
    assert "training_mean_squared_error" in metrics


@mock.patch("mlflow.recipes.steps.train._REBALANCING_CUTOFF", 50)
def test_train_step_imbalanced_data(tmp_recipe_root_path, capsys):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path, recipe="classification")
        recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
        recipe_yaml.write_text(
            """
            recipe: "classification/v1"
            target_col: "y"
            primary_metric: "f1_score"
            profile: "test_profile"
            positive_class: "a"
            run_args:
                step: "train"
            experiment:
                name: "demo"
                tracking_uri: {tracking_uri}
            steps:
                train:
                    using: custom
                    estimator_method: estimator_fn
                    tuning:
                        enabled: false
            """.format(
                tracking_uri=mlflow.get_tracking_uri()
            )
        )
        with mock.patch(
            "steps.train.estimator_fn",
            classifier_estimator_fn,
        ):
            recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
            train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
            train_step.run(str(train_step_output_dir))

    captured = capsys.readouterr()
    assert "Detected class imbalance" in captured.err
    assert "After downsampling: minority class percentage is 0.30" in captured.err

    run_id = train_step_output_dir.joinpath("run_id").read_text()
    metrics = MlflowClient().get_run(run_id).data.metrics
    assert "val_f1_score" in metrics


def test_train_step_classifier_automl(tmp_recipe_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path, recipe="classification")
        recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
        recipe_yaml.write_text(
            """
            recipe: "classification/v1"
            target_col: "y"
            primary_metric: "f1_score"
            profile: "test_profile"
            positive_class: "a"
            run_args:
                step: "train"
            experiment:
                name: "demo"
                tracking_uri: {tracking_uri}
            steps:
                train:
                    using: automl/flaml
                    time_budget_secs: 20
                    flaml_params:
                        estimator_list:
                        - rf
                        - lgbm
            """.format(
                tracking_uri=mlflow.get_tracking_uri()
            )
        )
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        train_step.run(str(train_step_output_dir))

    run_id = train_step_output_dir.joinpath("run_id").read_text()
    metrics = MlflowClient().get_run(run_id).data.metrics
    assert "val_f1_score" in metrics


def setup_train_step_with_automl(
    recipe_root: Path,
    primary_metric: str = "root_mean_squared_error",
    generate_custom_metrics: bool = False,
):
    recipe_yaml = recipe_root.joinpath(_RECIPE_CONFIG_FILE_NAME)

    custom_metric = """
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
            """
    recipe_yaml.write_text(
        f"""
  recipe: regression/v1
  target_col: y
  primary_metric: { primary_metric }
  profile: test_profile
  run_args:
    step: train
  experiment:
    name: demo
    tracking_uri: { mlflow.get_tracking_uri() }
  steps:
    train:
      using: automl/flaml
      time_budget_secs: 20
      flaml_params:
        estimator_list:
          - xgboost
          - rf
          - lgbm
  custom_metrics:
    { custom_metric if generate_custom_metrics else "" }
        """
    )
    recipe_config = read_yaml(recipe_root, _RECIPE_CONFIG_FILE_NAME)
    train_step = TrainStep.from_recipe_config(recipe_config, str(recipe_root))
    return train_step


def estimator_fn(estimator_params=None):
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor(random_state=42, **(estimator_params or {}))


def classifier_estimator_fn(estimator_params=None):
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(random_state=42, **(estimator_params or {}))


def early_stop_fn(trial, count=0):  # pylint: disable=unused-argument
    return count + 1 <= 2, [count + 1]


@pytest.mark.parametrize("use_tuning", [True, False])
def test_train_steps_writes_model_pkl_and_card(tmp_recipe_root_path, use_tuning):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step.run(str(train_step_output_dir))

    assert (train_step_output_dir / "model/model.pkl").exists()
    assert (train_step_output_dir / "card.html").exists()
    assert (train_step_output_dir / "predicted_training_data.parquet").exists()


@pytest.mark.parametrize("use_tuning", [True, False])
def test_train_steps_writes_card_with_model_and_run_links_on_databricks(
    monkeypatch, tmp_recipe_root_path, use_tuning
):
    workspace_host = "https://dev.databricks.com"
    workspace_id = 123456
    workspace_url = f"{workspace_host}?o={workspace_id}"

    monkeypatch.setenv("_DATABRICKS_WORKSPACE_HOST", workspace_host)
    monkeypatch.setenv("_DATABRICKS_WORKSPACE_ID", workspace_id)
    monkeypatch.setenv(_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR, str(tmp_recipe_root_path))

    train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
    train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
    with mock.patch(
        "steps.train.estimator_fn",
        estimator_fn,
    ):
        train_step.run(str(train_step_output_dir))

    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    assert (train_step_output_dir / "card.html").exists()
    with open(train_step_output_dir / "card.html", "r") as f:
        step_card_content = f.read()

    assert f"<a href={workspace_url}#mlflow/experiments/1/runs/{run_id}>" in step_card_content
    assert (
        f"<a href={workspace_url}#mlflow/experiments/1/runs/{run_id}/artifactPath/train/model>"
        in step_card_content
    )


@pytest.mark.parametrize("use_tuning", [True, False])
def test_train_steps_autologs(tmp_recipe_root_path, use_tuning):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step.run(str(train_step_output_dir))

    assert os.path.exists(train_step_output_dir / "run_id")

    # assert eval output exists
    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    metrics = MlflowClient().get_run(run_id).data.metrics
    params = MlflowClient().get_run(run_id).data.params
    assert "training_score" in metrics
    assert "epsilon" in params


@pytest.mark.parametrize("use_tuning", [True, False])
def test_train_steps_with_correct_tags(tmp_recipe_root_path, use_tuning):
    with mock.patch.dict(
        os.environ,
        {
            _MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path),
            _MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME_ENV_VAR: "train",
        },
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step.run(str(train_step_output_dir))

    assert os.path.exists(train_step_output_dir / "run_id")

    # assert eval output exists
    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    tags = MlflowClient().get_run(run_id).data.tags
    assert tags[MLFLOW_SOURCE_TYPE] == "RECIPE"
    assert tags[MLFLOW_RECIPE_TEMPLATE_NAME] == "regression/v1"
    assert tags[MLFLOW_RECIPE_STEP_NAME] == "train"
    assert tags[MLFLOW_RECIPE_PROFILE_NAME] == "test_profile"


def test_train_step_with_tuning_best_parameters(tmp_recipe_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning=True)
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step.run(str(train_step_output_dir))
    assert (train_step_output_dir / "best_parameters.yaml").exists()

    best_params_yaml = read_yaml(train_step_output_dir, "best_parameters.yaml")
    assert "alpha" in best_params_yaml
    assert "penalty" in best_params_yaml
    assert "eta0" in best_params_yaml

    run_id = open(train_step_output_dir / "run_id").read()
    parent_run_params = MlflowClient().get_run(run_id).data.params
    assert "alpha" in parent_run_params
    assert "penalty" in parent_run_params
    assert "eta0" in parent_run_params


@pytest.mark.parametrize(
    "with_hardcoded_params, expected_num_tuned, expected_num_hardcoded, num_sections",
    [(True, 3, 1, 3), (False, 3, 0, 2)],
)
def test_train_step_with_tuning_output_yaml_correct(
    tmp_recipe_root_path,
    with_hardcoded_params,
    expected_num_tuned,
    expected_num_hardcoded,
    num_sections,
):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        train_step = setup_train_step_with_tuning(
            tmp_recipe_root_path, use_tuning=True, with_hardcoded_params=with_hardcoded_params
        )
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step.run(str(train_step_output_dir))
    assert (train_step_output_dir / "best_parameters.yaml").exists()

    with open(os.path.join(train_step_output_dir, "best_parameters.yaml")) as f:
        lines = f.readlines()
        assert lines[0] == "# tuned hyperparameters \n"
        if with_hardcoded_params:
            assert lines[expected_num_tuned + 2] == "# hardcoded parameters \n"
            assert (
                lines[expected_num_tuned + expected_num_hardcoded + 4] == "# default parameters \n"
            )
        else:
            assert lines[expected_num_tuned + 2] == "# default parameters \n"
        assert len(lines) == 19 + num_sections * 2


def test_train_step_with_tuning_child_runs_and_early_stop(tmp_recipe_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path)}
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning=True)
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step.run(str(train_step_output_dir))

    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    run = MlflowClient().get_run(run_id)
    child_runs = train_step._get_tuning_df(run, params=["alpha", "penalty", "eta0"])
    assert len(child_runs) == 2
    assert "params.alpha" in child_runs.columns
    assert "params.penalty" in child_runs.columns
    assert "params.eta0" in child_runs.columns

    ordered_metrics = list(child_runs["root_mean_squared_error"])
    assert ordered_metrics == sorted(ordered_metrics)


@pytest.mark.skipif("hyperopt" not in sys.modules, reason="requires hyperopt to be installed")
def test_search_space(tmp_recipe_root_path):
    tuning_params_yaml = tmp_recipe_root_path.joinpath("tuning_params.yaml")
    tuning_params_yaml.write_text(
        """
        parameters:
            alpha:
                distribution: "uniform"
                low: 0.0
                high: 0.01
        """
    )
    tuning_params = read_yaml(tmp_recipe_root_path, "tuning_params.yaml")
    search_space = TrainStep.construct_search_space_from_yaml(tuning_params["parameters"])
    assert "alpha" in search_space


@pytest.mark.parametrize("tuning_param,logged_param", [(1, "1"), (1.0, "1.0"), ("a", " a ")])
def test_tuning_param_equal(tuning_param, logged_param):
    assert TrainStep.is_tuning_param_equal(tuning_param, logged_param)


@pytest.mark.parametrize(
    ("automl", "primary_metric", "generate_custom_metrics"),
    [
        (False, "root_mean_squared_error", False),
        (True, "root_mean_squared_error", False),
        (True, "weighted_mean_squared_error", True),
    ],
)
def test_automl(tmp_recipe_root_path, automl, primary_metric, generate_custom_metrics):
    with mock.patch.dict(
        os.environ,
        {
            _MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_recipe_root_path),
            _MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME_ENV_VAR: "train",
        },
    ):
        train_step_output_dir = setup_train_dataset(tmp_recipe_root_path)
        if generate_custom_metrics:
            recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
            recipe_steps_dir.joinpath("custom_metrics.py").write_text(
                """
def weighted_mean_squared_error(eval_df, builtin_metrics):
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(
        eval_df["prediction"],
        eval_df["target"],
        sample_weight=1 / eval_df["prediction"].values,
    )
        """
            )
        if automl:
            train_step = setup_train_step_with_automl(
                tmp_recipe_root_path,
                primary_metric=primary_metric,
                generate_custom_metrics=generate_custom_metrics,
            )
        else:
            train_step = setup_train_step_with_tuning(
                tmp_recipe_root_path,
                use_tuning=True,
                with_hardcoded_params=False,
            )
        with mock.patch(
            "steps.train.estimator_fn",
            estimator_fn,
        ):
            train_step._validate_and_apply_step_config()
            train_step._run(str(train_step_output_dir))

        with open(train_step_output_dir / "run_id") as f:
            run_id = f.read()

        metrics = MlflowClient().get_run(run_id).data.metrics
        assert f"training_{primary_metric}" in metrics
