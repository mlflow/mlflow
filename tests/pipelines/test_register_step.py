from pathlib import Path

import pytest

import mlflow
from mlflow.utils.file_utils import read_yaml
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.steps.evaluate import EvaluateStep
from mlflow.pipelines.steps.register import RegisterStep

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    clear_custom_metrics_module_cache,
    setup_model_and_evaluate,
    tmp_pipeline_exec_path,
    tmp_pipeline_root_path,
)  # pylint: enable=unused-import


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
@pytest.mark.parametrize(
    "mae_threshold,register_flag",
    [
        (-1, ""),
        (1_000_000, ""),
        (-1, "allow_non_validated_model: true"),
        (1_000_000, "allow_non_validated_model: true"),
    ],
)
def test_register_step_run(
    tmp_pipeline_root_path: Path,
    tmp_pipeline_exec_path: Path,
    mae_threshold: int,
    register_flag: str,
):
    evaluate_step_output_dir, register_step_output_dir = setup_model_and_evaluate(
        tmp_pipeline_exec_path
    )
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
  register:
    model_name: "demo_model"
    {allow_non_validated_model}
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
""".format(
            tracking_uri=mlflow.get_tracking_uri(),
            mae_threshold=mae_threshold,
            allow_non_validated_model=register_flag,
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
    assert len(mlflow.tracking.MlflowClient().list_registered_models()) == 0
    register_step = RegisterStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    register_step._run(str(register_step_output_dir))
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    expected_status = "REJECTED" if mae_threshold < 0 else "VALIDATED"
    assert model_validation_status_path.read_text() == expected_status
    assert len(mlflow.tracking.MlflowClient().list_registered_models()) == (
        1 if expected_status == "VALIDATED" else 0
    )


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
@pytest.mark.parametrize("register_flag", ["", "allow_non_validated_model: true"])
def test_register_with_no_validation_criteria(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path, register_flag: str
):
    evaluate_step_output_dir, register_step_output_dir = setup_model_and_evaluate(
        tmp_pipeline_exec_path
    )
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {tracking_uri}
steps:
  evaluate:
  register:
    model_name: "demo_model"
    {allow_non_validated_model}
""".format(
            tracking_uri=mlflow.get_tracking_uri(),
            allow_non_validated_model=register_flag,
        )
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    evaluate_step._run(str(evaluate_step_output_dir))
    assert len(mlflow.tracking.MlflowClient().list_registered_models()) == 0
    register_step = RegisterStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    register_step._run(str(register_step_output_dir))
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    assert model_validation_status_path.read_text() == "UNKNOWN"
    assert len(mlflow.tracking.MlflowClient().list_registered_models()) == (
        0 if register_flag == "" else 1
    )
