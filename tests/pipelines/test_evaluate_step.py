from unittest import mock
from pathlib import Path

import os
import pytest
import shutil
from sklearn.datasets import load_diabetes

import mlflow
from mlflow.utils.file_utils import read_yaml
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.steps.split import _OUTPUT_TEST_FILE_NAME, _OUTPUT_VALIDATION_FILE_NAME
from mlflow.pipelines.steps.evaluate import EvaluateStep
from mlflow.exceptions import MlflowException

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    clear_custom_metrics_module_cache,
    tmp_pipeline_exec_path,
    tmp_pipeline_root_path,
    train_and_log_model,
)  # pylint: enable=unused-import


@pytest.fixture(autouse=True)
def evaluation_inputs(tmp_pipeline_exec_path):
    split_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    validation_df = X.assign(y=y).sample(n=50, random_state=9)
    validation_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_VALIDATION_FILE_NAME))
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    run_id, model = train_and_log_model()
    train_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)
    output_model_path = train_step_output_dir.joinpath("model")
    if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
        shutil.rmtree(output_model_path)
    mlflow.sklearn.save_model(model, output_model_path)


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
@pytest.mark.parametrize("mae_threshold", [-1, 1_000_000])
def test_evaluate_step_run(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path, mae_threshold: int
):
    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
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
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri(),
            mae_threshold=mae_threshold,
        )
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def weighted_mean_squared_error(eval_df, builtin_metrics):
    from sklearn.metrics import mean_squared_error

    return {
        "weighted_mean_squared_error": mean_squared_error(
            eval_df["prediction"],
            eval_df["target"],
            sample_weight=1 / eval_df["prediction"].values,
        )
    }
"""
    )
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    evaluate_step._run(str(evaluate_step_output_dir))

    logged_metrics = (
        mlflow.tracking.MlflowClient().get_run(mlflow.last_active_run().info.run_id).data.metrics
    )
    logged_metrics = {k.replace("_on_data_test", ""): v for k, v in logged_metrics.items()}
    assert "weighted_mean_squared_error" in logged_metrics
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    expected_status = "REJECTED" if mae_threshold < 0 else "VALIDATED"
    assert model_validation_status_path.read_text() == expected_status


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
def test_no_validation_criteria(tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path):
    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    evaluate_step._run(str(evaluate_step_output_dir))

    logged_metrics = (
        mlflow.tracking.MlflowClient().get_run(mlflow.last_active_run().info.run_id).data.metrics
    )
    logged_metrics = {k.replace("_on_data_test", ""): v for k, v in logged_metrics.items()}
    assert "mean_squared_error" in logged_metrics
    assert "root_mean_squared_error" in logged_metrics
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    assert model_validation_status_path.read_text() == "UNKNOWN"


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_pipeline_exec_path")
def test_validation_criteria_contain_undefined_metrics(tmp_pipeline_root_path: Path):
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
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
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)

    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    with pytest.raises(
        MlflowException,
        match=r"Validation criteria contain undefined metrics: \['undefined_metric'\]",
    ):
        evaluate_step._validate_validation_criteria()


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_pipeline_exec_path")
def test_custom_metric_function_does_not_exist(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path
):
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: weighted_mean_squared_error
        threshold: 100
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def one(eval_df, builtin_metrics):
    return {"one": 1}
"""
    )
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)
    with pytest.raises(MlflowException, match="Failed to load custom metric functions") as exc:
        evaluate_step._run(str(evaluate_step_output_dir))
    assert isinstance(exc.value.__cause__, AttributeError)
    assert "weighted_mean_squared_error" in str(exc.value.__cause__)


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_pipeline_exec_path")
def test_custom_metrics_module_does_not_exist(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path
):
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
    validation_criteria:
      - metric: weighted_mean_squared_error
        threshold: 100
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)

    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)
    with pytest.raises(MlflowException, match="Failed to load custom metric functions") as exc:
        evaluate_step._run(str(evaluate_step_output_dir))
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)
    assert "No module named 'steps.custom_metrics'" in str(exc.value.__cause__)


@pytest.mark.usefixtures("clear_custom_metrics_module_cache", "tmp_pipeline_exec_path")
def test_custom_metrics_override_builtin_metrics(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path
):
    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
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
metrics:
  custom:
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
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def mean_absolute_error(eval_df, builtin_metrics):
    return {"mean_absolute_error": 1}

def root_mean_squared_error(eval_df, builtin_metrics):
    return {"root_mean_squared_error": 1}
"""
    )
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)

    with mock.patch("mlflow.pipelines.utils.metrics._logger.warning") as mock_warning:
        evaluate_step = EvaluateStep.from_pipeline_config(
            pipeline_config, str(tmp_pipeline_root_path)
        )
        evaluate_step._run(str(evaluate_step_output_dir))
        mock_warning.assert_called_once_with(
            "Custom metrics override the following built-in metrics: %s",
            ["mean_absolute_error", "root_mean_squared_error"],
        )
    logged_metrics = (
        mlflow.tracking.MlflowClient().get_run(mlflow.last_active_run().info.run_id).data.metrics
    )
    for dataset in ["validation", "test"]:
        assert f"root_mean_squared_error_on_data_{dataset}" in logged_metrics
        assert logged_metrics[f"root_mean_squared_error_on_data_{dataset}"] == 1
        assert f"mean_absolute_error_on_data_{dataset}" in logged_metrics
        assert logged_metrics[f"mean_absolute_error_on_data_{dataset}"] == 1
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    assert model_validation_status_path.read_text() == "VALIDATED"


def test_evaluate_step_writes_card_with_model_and_run_links_on_databricks(
    monkeypatch, tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path
):
    workspace_host = "https://dev.databricks.com"
    workspace_id = 123456
    workspace_url = f"{workspace_host}?o={workspace_id}"

    monkeypatch.setenv("_DATABRICKS_WORKSPACE_HOST", workspace_host)
    monkeypatch.setenv("_DATABRICKS_WORKSPACE_ID", workspace_id)

    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
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

    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    evaluate_step._run(str(evaluate_step_output_dir))

    train_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "train", "outputs")
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
