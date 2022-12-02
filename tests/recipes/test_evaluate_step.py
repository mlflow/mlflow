from unittest import mock
from pathlib import Path

import os
import pytest
import shutil
from sklearn.datasets import load_diabetes, load_iris

import mlflow
from mlflow.utils.file_utils import read_yaml
from mlflow.recipes.utils import _RECIPE_CONFIG_FILE_NAME
from mlflow.recipes.steps.split import _OUTPUT_TEST_FILE_NAME, _OUTPUT_VALIDATION_FILE_NAME
from mlflow.recipes.steps.evaluate import EvaluateStep
from mlflow.exceptions import MlflowException

# pylint: disable=unused-import
from tests.recipes.helper_functions import (
    clear_custom_metrics_module_cache,
    tmp_recipe_exec_path,
    tmp_recipe_root_path,
    train_and_log_model,
    train_and_log_classification_model,
)  # pylint: enable=unused-import


@pytest.fixture(autouse=True)
def evaluation_inputs(request, tmp_recipe_exec_path):
    split_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    if "classification" in request.keywords:
        X, y = load_iris(as_frame=True, return_X_y=True)
    else:
        X, y = load_diabetes(as_frame=True, return_X_y=True)
    validation_df = X.assign(y=y).sample(n=50, random_state=9)
    validation_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_VALIDATION_FILE_NAME))
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    if "classification" in request.keywords:
        run_id, model = train_and_log_classification_model()
    else:
        run_id, model = train_and_log_model()
    train_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)
    output_model_path = train_step_output_dir.joinpath("model")
    if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
        shutil.rmtree(output_model_path)
    mlflow.sklearn.save_model(model, output_model_path)


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
@pytest.mark.parametrize("mae_threshold", [-1, 1_000_000])
def test_evaluate_step_run(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, mae_threshold: int
):
    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 1_000_000
      - metric: mean_absolute_error
        threshold: {mae_threshold}
      - metric: weighted_mean_squared_error
        threshold: 1_000_000
custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error_func
    greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri(),
            mae_threshold=mae_threshold,
        )
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
    recipe_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def weighted_mean_squared_error_func(eval_df, builtin_metrics):
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(
        eval_df["prediction"],
        eval_df["target"],
        sample_weight=1 / eval_df["prediction"].values,
    )
"""
    )
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))

    logged_metrics = (
        mlflow.tracking.MlflowClient().get_run(mlflow.last_active_run().info.run_id).data.metrics
    )
    assert "test_weighted_mean_squared_error" in logged_metrics
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    expected_status = "REJECTED" if mae_threshold < 0 else "VALIDATED"
    assert model_validation_status_path.read_text() == expected_status


@pytest.mark.classification
@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
def test_evaluate_produces_expected_step_card(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "classification/v1"
positive_class: "Iris-setosa"
target_col: "y"
primary_metric: "f1_score"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: f1_score
        threshold: 10
""".format(
            tracking_uri=mlflow.get_tracking_uri(),
        )
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))

    with open(evaluate_step_output_dir / "card.html", "r", errors="ignore") as f:
        step_card_content = f.read()

    assert "Model Validation" in step_card_content
    assert "Model Performance Plots" in step_card_content
    assert "Warning Logs" in step_card_content
    assert "Run Summary" in step_card_content


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
def test_no_validation_criteria(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path):
    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))

    logged_metrics = (
        mlflow.tracking.MlflowClient().get_run(mlflow.last_active_run().info.run_id).data.metrics
    )
    assert "test_mean_squared_error" in logged_metrics
    assert "test_root_mean_squared_error" in logged_metrics
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    assert model_validation_status_path.read_text() == "UNKNOWN"


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_recipe_exec_path")
def test_validation_criteria_contain_undefined_metrics(tmp_recipe_root_path: Path):
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 100
      - metric: undefined_metric
        threshold: 100
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step._validate_and_apply_step_config()
    with pytest.raises(
        MlflowException,
        match=r"Validation criteria contain undefined metrics: \['undefined_metric'\]",
    ):
        evaluate_step._validate_validation_criteria()


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_recipe_exec_path")
def test_custom_metric_function_does_not_exist(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: weighted_mean_squared_error
        threshold: 100
custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
    recipe_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def one(eval_df, builtin_metrics):
    return {"one": 1}
"""
    )
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)
    with pytest.raises(MlflowException, match="Failed to load custom metric functions") as exc:
        evaluate_step.run(str(evaluate_step_output_dir))
    assert isinstance(exc.value.__cause__, AttributeError)
    assert "weighted_mean_squared_error" in str(exc.value.__cause__)


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_recipe_exec_path")
def test_custom_metrics_module_does_not_exist(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: weighted_mean_squared_error
        threshold: 100
custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)
    with pytest.raises(MlflowException, match="Failed to load custom metric functions") as exc:
        evaluate_step.run(str(evaluate_step_output_dir))
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)
    assert "No module named 'steps.custom_metrics'" in str(exc.value.__cause__)


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_recipe_exec_path")
def test_custom_metrics_override_builtin_metrics(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 10
      - metric: mean_absolute_error
        threshold: 10
custom_metrics:
  - name: mean_absolute_error
    function: mean_absolute_error
    greater_is_better: False
  - name: root_mean_squared_error
    function: root_mean_squared_error
    greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
    recipe_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def mean_absolute_error(eval_df, builtin_metrics):
    return 1

def root_mean_squared_error(eval_df, builtin_metrics):
    return 1
"""
    )
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)

    with mock.patch("mlflow.recipes.utils.metrics._logger.warning") as mock_warning:
        evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
        evaluate_step.run(str(evaluate_step_output_dir))
        mock_warning.assert_called_once_with(
            "Custom metrics override the following built-in metrics: %s",
            ["mean_absolute_error", "root_mean_squared_error"],
        )
    logged_metrics = (
        mlflow.tracking.MlflowClient().get_run(mlflow.last_active_run().info.run_id).data.metrics
    )
    assert "test_root_mean_squared_error" in logged_metrics
    assert logged_metrics["test_root_mean_squared_error"] == 1
    assert "test_mean_absolute_error" in logged_metrics
    assert logged_metrics["test_mean_absolute_error"] == 1
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    assert model_validation_status_path.read_text() == "VALIDATED"


def test_evaluate_step_writes_card_with_model_and_run_links_on_databricks(
    monkeypatch, tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    workspace_host = "https://dev.databricks.com"
    workspace_id = 123456
    workspace_url = f"{workspace_host}?o={workspace_id}"

    monkeypatch.setenv("_DATABRICKS_WORKSPACE_HOST", workspace_host)
    monkeypatch.setenv("_DATABRICKS_WORKSPACE_ID", workspace_id)

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        """
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 1_000_000
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )

    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))

    train_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "train", "outputs")
    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    assert (evaluate_step_output_dir / "card.html").exists()
    with open(evaluate_step_output_dir / "card.html", "r") as f:
        step_card_content = f.read()

    assert f"<a href={workspace_url}#mlflow/experiments/1/runs/{run_id}>" in step_card_content
    assert (
        f"<a href={workspace_url}#mlflow/experiments/1/runs/{run_id}/artifactPath/train/model>"
        in step_card_content
    )
