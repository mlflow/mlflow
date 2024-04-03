import importlib
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock, Mock

import cloudpickle
import pandas as pd
import pytest
import sklearn.compose

import mlflow
from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.utils import _RECIPE_CONFIG_FILE_NAME
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import read_yaml
from mlflow.utils.mlflow_tags import (
    MLFLOW_RECIPE_PROFILE_NAME,
    MLFLOW_RECIPE_STEP_NAME,
    MLFLOW_RECIPE_TEMPLATE_NAME,
    MLFLOW_SOURCE_TYPE,
)
from mlflow.utils.os import is_windows


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
        if recipe == "classification/binary":
            minority_class_cnt = math.ceil(0.1 * num_rows)
            majority_class_cnt = num_rows - minority_class_cnt
            y = ["a"] * minority_class_cnt + ["b"] * majority_class_cnt
        else:
            minority_class_cnt1 = math.ceil(0.05 * num_rows)
            minority_class_cnt2 = math.ceil(0.03 * num_rows)
            minority_class_cnt3 = math.ceil(0.07 * num_rows)
            majority_class_cnt = (
                num_rows - minority_class_cnt1 - minority_class_cnt2 - minority_class_cnt3
            )
            y = (
                ["a1"] * minority_class_cnt1
                + ["a2"] * minority_class_cnt2
                + ["a3"] * minority_class_cnt3
                + ["b"] * majority_class_cnt
            )
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
    recipe_root: Path,
    use_tuning: bool,
    with_hardcoded_params: bool = True,
    recipe: str = "regression",
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
    estimator_fn = "estimator_fn" if recipe == "regression" else "classifier_estimator_fn"
    if use_tuning:
        recipe_yaml.write_text(
            f"""
            recipe: "{recipe}/v1"
            target_col: "y"
            profile: "test_profile"
            run_args:
                step: "train"
            experiment:
                name: "demo"
                tracking_uri: {mlflow.get_tracking_uri()}
            steps:
                train:
                    using: custom
                    estimator_method: {estimator_fn}
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
            """
        )
    else:
        recipe_yaml.write_text(
            f"""
            recipe: "{recipe}/v1"
            target_col: "y"
            profile: "test_profile"
            run_args:
                step: "train"
            experiment:
                name: "demo"
                tracking_uri: {mlflow.get_tracking_uri()}
            steps:
                train:
                    using: custom
                    estimator_method: {estimator_fn}
                    tuning:
                        enabled: false
            """
        )
    recipe_config = read_yaml(recipe_root, _RECIPE_CONFIG_FILE_NAME)
    return TrainStep.from_recipe_config(recipe_config, str(recipe_root))


def test_train_step(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "regression/v1"
        target_col: "y"
        profile: "test_profile"
        run_args:
            step: "train"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            train:
                using: custom
                estimator_method: estimator_fn
                tuning:
                    enabled: false
        """
    )
    m_train = Mock()
    m_train.estimator_fn = estimator_fn

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    with mock.patch("steps.train.estimator_fn", estimator_fn):
        train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        train_step.run(str(train_step_output_dir))

    run_id = train_step_output_dir.joinpath("run_id").read_text()
    metrics = MlflowClient().get_run(run_id).data.metrics
    assert "val_mean_squared_error" in metrics
    assert "training_mean_squared_error" in metrics


@pytest.fixture(autouse=True)
def dummy_train_step(tmp_recipe_root_path, monkeypatch):
    # `mock.patch("steps.train.estimator_fn", ...)` would fail without this fixture
    steps = tmp_recipe_root_path / "steps"
    steps.mkdir(exist_ok=True)
    steps.joinpath("train.py").write_text(
        """
def estimator_fn(estimator_params=None):
    return None
"""
    )
    monkeypatch.syspath_prepend(str(tmp_recipe_root_path))


@mock.patch("mlflow.recipes.steps.train._REBALANCING_CUTOFF", 50)
def test_train_step_imbalanced_data(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path):
    train_step_output_dir = setup_train_dataset(
        tmp_recipe_exec_path, recipe="classification/multiclass"
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "classification/v1"
        target_col: "y"
        primary_metric: "f1_score"
        profile: "test_profile"
        positive_class: "a"
        run_args:
            step: "train"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            train:
                using: custom
                estimator_method: estimator_fn
                tuning:
                    enabled: false
        """
    )
    with mock.patch("steps.train.estimator_fn", classifier_estimator_fn):
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        train_step.run(str(train_step_output_dir))

    # captured = capsys.readouterr()
    # assert "Detected class imbalance" in captured.err
    # assert "After downsampling: minority class percentage is 0.30" in captured.err

    run_id = train_step_output_dir.joinpath("run_id").read_text()
    metrics = MlflowClient().get_run(run_id).data.metrics
    assert "val_f1_score" in metrics


@pytest.mark.parametrize("recipe", ["classification/binary", "classification/multiclass"])
def test_train_step_classifier_automl(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, recipe
):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path, recipe=recipe)
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
        recipe: "classification/v1"
        target_col: "y"
        primary_metric: "roc_auc"
        profile: "test_profile"
        {positive_class}
        run_args:
            step: "train"
        experiment:
            name: "demo"
            tracking_uri: {tracking_uri}
        steps:
            train:
                using: automl/flaml
                time_budget_secs: 5
                flaml_params:
                    estimator_list:
                    - rf
                    - lgbm
        """.format(
            tracking_uri=mlflow.get_tracking_uri(),
            positive_class='positive_class: "a"' if recipe == "classification/binary" else "",
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
  primary_metric: {primary_metric}
  profile: test_profile
  run_args:
    step: train
  experiment:
    name: demo
    tracking_uri: {mlflow.get_tracking_uri()}
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
    {custom_metric if generate_custom_metrics else ""}
        """
    )
    recipe_config = read_yaml(recipe_root, _RECIPE_CONFIG_FILE_NAME)
    return TrainStep.from_recipe_config(recipe_config, str(recipe_root))


def estimator_fn(estimator_params=None):
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor(random_state=42, **(estimator_params or {}))


def classifier_estimator_fn(estimator_params=None):
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(random_state=42, **(estimator_params or {}))


def sklearn_logistic_regression():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression()


def xgb_classifier():
    import xgboost as xgb

    return xgb.XGBClassifier()


def early_stop_fn(trial, count=0):
    return count + 1 <= 2, [count + 1]


@pytest.mark.parametrize("use_tuning", [True, False])
def test_train_steps_writes_model_pkl_and_card(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, use_tuning
):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
    with mock.patch("steps.train.estimator_fn", estimator_fn):
        train_step.run(str(train_step_output_dir))

    assert (train_step_output_dir / "model/python_model.pkl").exists()
    assert (train_step_output_dir / "card.html").exists()
    assert (train_step_output_dir / "predicted_training_data.parquet").exists()


@pytest.mark.parametrize("use_tuning", [True, False])
def test_train_steps_writes_card_with_model_and_run_links_on_databricks(
    monkeypatch, tmp_recipe_exec_path: Path, tmp_recipe_root_path: Path, use_tuning
):
    workspace_host = "https://dev.databricks.com"
    workspace_id = 123456
    workspace_url = f"{workspace_host}?o={workspace_id}"

    monkeypatch.setenv("_DATABRICKS_WORKSPACE_HOST", workspace_host)
    monkeypatch.setenv("_DATABRICKS_WORKSPACE_ID", workspace_id)

    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
    with mock.patch("steps.train.estimator_fn", estimator_fn):
        train_step.run(str(train_step_output_dir))

    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    assert (train_step_output_dir / "card.html").exists()
    with open(train_step_output_dir / "card.html") as f:
        step_card_content = f.read()

    assert f"<a href={workspace_url}#mlflow/experiments/1/runs/{run_id}>" in step_card_content
    assert (
        f"<a href={workspace_url}#mlflow/experiments/1/runs/{run_id}/artifactPath/train/model>"
        in step_card_content
    )


@pytest.mark.parametrize("use_tuning", [True, False])
def test_train_steps_autologs(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, use_tuning):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
    m_train = Mock()
    m_train.estimator_fn = estimator_fn
    with mock.patch("steps.train.estimator_fn", estimator_fn):
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
def test_train_steps_with_correct_tags(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, use_tuning, monkeypatch
):
    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME.name, "train")
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning)
    m_train = Mock()
    m_train.estimator_fn = estimator_fn
    monkeypatch.setitem(sys.modules, "steps.train", m_train)
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


def test_train_step_with_tuning_best_parameters(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning=True)
    with mock.patch("steps.train.estimator_fn", estimator_fn):
        train_step.run(str(train_step_output_dir))

    assert (train_step_output_dir / "best_parameters.yaml").exists()

    best_params_yaml = read_yaml(train_step_output_dir, "best_parameters.yaml")
    assert "alpha" in best_params_yaml
    assert "penalty" in best_params_yaml
    assert "eta0" in best_params_yaml

    run_id = train_step_output_dir.joinpath("run_id").read_text()
    parent_run_params = MlflowClient().get_run(run_id).data.params
    assert "alpha" in parent_run_params
    assert "penalty" in parent_run_params
    assert "eta0" in parent_run_params


@pytest.mark.parametrize(
    ("with_hardcoded_params", "expected_num_tuned", "expected_num_hardcoded", "num_sections"),
    [(True, 3, 1, 3), (False, 3, 0, 2)],
)
def test_train_step_with_tuning_output_yaml_correct(
    tmp_recipe_root_path: Path,
    tmp_recipe_exec_path: Path,
    with_hardcoded_params,
    expected_num_tuned,
    expected_num_hardcoded,
    num_sections,
):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    train_step = setup_train_step_with_tuning(
        tmp_recipe_root_path, use_tuning=True, with_hardcoded_params=with_hardcoded_params
    )
    with mock.patch("steps.train.estimator_fn", estimator_fn):
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


@pytest.mark.parametrize("with_hardcoded_params", [(True), (False)])
def test_train_step_with_tuning_trials_card_tab(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, with_hardcoded_params
):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True, exist_ok=True)
    train_step = setup_train_step_with_tuning(
        tmp_recipe_root_path, use_tuning=True, with_hardcoded_params=with_hardcoded_params
    )
    m_train = Mock()
    m_train.estimator_fn = estimator_fn
    with mock.patch.dict("sys.modules", {"steps.train": m_train}):
        train_step.run(str(train_step_output_dir))
    assert (train_step_output_dir / "card.html").exists()
    with open(train_step_output_dir / "card.html") as f:
        step_card_content = f.read()

    assert "Tuning Trials" in step_card_content


def test_train_step_with_tuning_child_runs_and_early_stop(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    train_step = setup_train_step_with_tuning(tmp_recipe_root_path, use_tuning=True)
    with mock.patch("steps.train.estimator_fn", estimator_fn):
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


@pytest.mark.parametrize(("tuning_param", "logged_param"), [(1, "1"), (1.0, "1.0"), ("a", " a ")])
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
def test_automl(
    tmp_recipe_root_path: Path,
    tmp_recipe_exec_path: Path,
    automl,
    primary_metric,
    generate_custom_metrics,
    monkeypatch,
):
    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME.name, "train")
    train_step_output_dir = setup_train_dataset(tmp_recipe_exec_path)
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(exist_ok=True)
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

    with mock.patch("steps.train.estimator_fn", estimator_fn):
        train_step._validate_and_apply_step_config()
        train_step._run(str(train_step_output_dir))

    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    metrics = MlflowClient().get_run(run_id).data.metrics
    assert f"training_{primary_metric}" in metrics


def test_tuning_multiclass(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, monkeypatch):
    monkeypatch.setenv(MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME.name, "train")
    train_step_output_dir = setup_train_dataset(
        tmp_recipe_exec_path, recipe="classification/multiclass"
    )

    train_step = setup_train_step_with_tuning(
        tmp_recipe_root_path,
        use_tuning=True,
        with_hardcoded_params=False,
        recipe="classification",
    )

    _old_import_module = importlib.import_module

    def _import_module(name: str, package: Optional[str] = None):
        if "steps" in name:
            return _old_import_module("tests.recipes.test_train_step")
        else:
            return _old_import_module(name, package)

    imp_pkg = MagicMock(name="api")
    imp_pkg.side_effect = _import_module
    with mock.patch("importlib.import_module", new=imp_pkg):
        train_step._validate_and_apply_step_config()
        train_step._run(str(train_step_output_dir))

    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    metrics = MlflowClient().get_run(run_id).data.metrics
    assert "training_f1_score" in metrics


def test_train_step_with_predict_probability(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    train_step_output_dir = setup_train_dataset(
        tmp_recipe_exec_path, recipe="classification/binary"
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "classification/v1"
        target_col: "y"
        primary_metric: "f1_score"
        profile: "test_profile"
        positive_class: "a"
        run_args:
            step: "train"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            train:
                using: custom
                estimator_method: estimator_fn
                tuning:
                    enabled: false
        """
    )
    with mock.patch("steps.train.estimator_fn", return_value=sklearn_logistic_regression()):
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        train_step.run(str(train_step_output_dir))

    model_uri = get_step_output_path(
        recipe_root_path=tmp_recipe_root_path,
        step_name="train",
        relative_path=TrainStep.MODEL_ARTIFACT_RELATIVE_PATH,
    )
    sk_model_uri = get_step_output_path(
        recipe_root_path=tmp_recipe_root_path,
        step_name="train",
        relative_path=TrainStep.SKLEARN_MODEL_ARTIFACT_RELATIVE_PATH,
    )
    mlflow.sklearn.load_model(sk_model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    transform_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "transform", "outputs")

    validation_dataset = pd.read_parquet(
        str(transform_step_output_dir / "transformed_validation_data.parquet")
    )

    output = model.predict(validation_dataset.drop("y", axis=1))

    assert list(output.columns) == [
        "predicted_score_a",
        "predicted_score_b",
        "predicted_score",
        "predicted_label",
    ]

    import numpy as np

    assert np.array_equal(
        output.head(5)["predicted_score"].tolist(),
        np.max(output[["predicted_score_a", "predicted_score_b"]].head(5).values, axis=1),
    )


def test_train_step_with_predict_probability_with_custom_prefix(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    train_step_output_dir = setup_train_dataset(
        tmp_recipe_exec_path, recipe="classification/binary"
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "classification/v1"
        target_col: "y"
        primary_metric: "f1_score"
        profile: "test_profile"
        positive_class: "a"
        run_args:
            step: "train"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            train:
                using: custom
                estimator_method: estimator_fn
                predict_prefix: "custom_prefix_"
                tuning:
                    enabled: false
        """
    )
    with mock.patch("steps.train.estimator_fn", return_value=sklearn_logistic_regression()):
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        train_step.run(str(train_step_output_dir))

    model_uri = get_step_output_path(
        recipe_root_path=tmp_recipe_root_path,
        step_name="train",
        relative_path=TrainStep.MODEL_ARTIFACT_RELATIVE_PATH,
    )
    model = mlflow.pyfunc.load_model(model_uri)
    transform_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "transform", "outputs")

    validation_dataset = pd.read_parquet(
        str(transform_step_output_dir / "transformed_validation_data.parquet")
    )

    output = model.predict(validation_dataset.drop("y", axis=1))

    assert list(output.columns) == [
        "custom_prefix_score_a",
        "custom_prefix_score_b",
        "custom_prefix_score",
        "custom_prefix_label",
    ]


def test_train_step_with_label_encoding(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path):
    train_step_output_dir = setup_train_dataset(
        tmp_recipe_exec_path, recipe="classification/multiclass"
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "classification/v1"
        target_col: "y"
        profile: "test_profile"
        run_args:
            step: "train"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            train:
                using: custom
                estimator_method: estimator_fn
                tuning:
                    enabled: false
        """
    )
    with mock.patch("steps.train.estimator_fn", return_value=xgb_classifier()):
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        train_step.run(str(train_step_output_dir))

    model_uri = get_step_output_path(
        recipe_root_path=tmp_recipe_root_path,
        step_name="train",
        relative_path=TrainStep.MODEL_ARTIFACT_RELATIVE_PATH,
    )
    model = mlflow.pyfunc.load_model(model_uri)
    transform_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "transform", "outputs")

    validation_dataset = pd.read_parquet(
        str(transform_step_output_dir / "transformed_validation_data.parquet")
    )

    predicted_output = model.predict(validation_dataset.drop("y", axis=1))
    predicted_label = predicted_output["predicted_label"]

    import numpy as np

    assert np.array_equal(np.unique(predicted_label), np.array(["a1", "a2", "a3", "b"]))


@pytest.mark.skipif(
    is_windows(),
    reason="Flaky on windows, sometimes fails with `(sqlite3.OperationalError) database is locked`",
)
def test_train_step_with_probability_calibration(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    train_step_output_dir = setup_train_dataset(
        tmp_recipe_exec_path, recipe="classification/binary"
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
        recipe: "classification/v1"
        target_col: "y"
        primary_metric: "f1_score"
        profile: "test_profile"
        positive_class: "a"
        run_args:
            step: "train"
        experiment:
            name: "demo"
            tracking_uri: {mlflow.get_tracking_uri()}
        steps:
            train:
                using: custom
                estimator_method: estimator_fn
                calibrate_proba: isotonic
                tuning:
                    enabled: false
        """
    )
    with mock.patch("steps.train.estimator_fn", return_value=sklearn_logistic_regression()):
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        train_step = TrainStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        train_step.run(str(train_step_output_dir))

    model_uri = get_step_output_path(
        recipe_root_path=tmp_recipe_root_path,
        step_name="train",
        relative_path=TrainStep.MODEL_ARTIFACT_RELATIVE_PATH,
    )
    model = mlflow.pyfunc.load_model(model_uri)

    from sklearn.calibration import CalibratedClassifierCV

    assert isinstance(
        model._model_impl.python_model._classifier.named_steps["calibratedclassifiercv"],
        CalibratedClassifierCV,
    )

    assert (train_step_output_dir / "card.html").exists()
    with open(train_step_output_dir / "card.html") as f:
        step_card_content = f.read()

    assert "Prob. Calibration" in step_card_content
