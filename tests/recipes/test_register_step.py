from pathlib import Path

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.recipes.steps.evaluate import EvaluateStep
from mlflow.recipes.steps.register import _REGISTERED_MV_INFO_FILE, RegisterStep
from mlflow.recipes.utils import _RECIPE_CONFIG_FILE_NAME
from mlflow.utils.file_utils import read_yaml
from mlflow.utils.mlflow_tags import (
    MLFLOW_RECIPE_TEMPLATE_NAME,
    MLFLOW_SOURCE_TYPE,
)

from tests.recipes.helper_functions import setup_model_and_evaluate


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
@pytest.mark.parametrize(
    ("mae_threshold", "register_flag"),
    [
        (-1, ""),
        (1_000_000, ""),
        (-1, "allow_non_validated_model: true"),
        (1_000_000, "allow_non_validated_model: true"),
    ],
)
def test_register_step_run(
    tmp_recipe_root_path: Path,
    tmp_recipe_exec_path: Path,
    mae_threshold: int,
    register_flag: str,
):
    evaluate_step_output_dir, register_step_output_dir = setup_model_and_evaluate(
        tmp_recipe_exec_path
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {mlflow.get_tracking_uri()}
model_registry:
  model_name: "demo_model"
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
    {register_flag}
custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
"""
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
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
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))
    assert len(mlflow.tracking.MlflowClient().search_registered_models()) == 0
    register_step = RegisterStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    if mae_threshold < 0:
        with pytest.raises(MlflowException, match=r"Model registration on .* failed"):
            register_step.run(str(register_step_output_dir))
    else:
        register_step.run(str(register_step_output_dir))
        model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
        assert model_validation_status_path.exists()
        assert model_validation_status_path.read_text() == "VALIDATED"
        mv_info_file_path = register_step_output_dir.joinpath(_REGISTERED_MV_INFO_FILE)
        assert mv_info_file_path.exists()
        assert len(mlflow.tracking.MlflowClient().search_registered_models()) == 1


@pytest.mark.usefixtures("clear_custom_metrics_module_cache")
@pytest.mark.parametrize("register_flag", ["", "allow_non_validated_model: true"])
def test_register_with_no_validation_criteria(
    tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path, register_flag: str
):
    evaluate_step_output_dir, register_step_output_dir = setup_model_and_evaluate(
        tmp_recipe_exec_path
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {mlflow.get_tracking_uri()}
model_registry:
  model_name: "demo_model"
steps:
  evaluate:
  register:
    {register_flag}
"""
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))
    assert len(mlflow.tracking.MlflowClient().search_registered_models()) == 0
    register_step = RegisterStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    if register_flag == "":
        with pytest.raises(MlflowException, match=r"Model registration on .* failed"):
            register_step.run(str(register_step_output_dir))
    else:
        register_step.run(str(register_step_output_dir))
        model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
        assert model_validation_status_path.exists()
        assert model_validation_status_path.read_text() == "UNKNOWN"
        mv_info_file_path = register_step_output_dir.joinpath(_REGISTERED_MV_INFO_FILE)
        assert mv_info_file_path.exists()
        assert len(mlflow.tracking.MlflowClient().search_registered_models()) == 1


def test_usage_tracking_correctly_added(
    tmp_recipe_root_path: Path,
    tmp_recipe_exec_path: Path,
):
    evaluate_step_output_dir, register_step_output_dir = setup_model_and_evaluate(
        tmp_recipe_exec_path
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
recipe: "regression/v1"
target_col: "y"
model_registry:
  model_name: "demo_model"
experiment:
  tracking_uri: {mlflow.get_tracking_uri()}
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 1_000_000
      - metric: mean_absolute_error
        threshold: 1_000_000
      - metric: weighted_mean_squared_error
        threshold: 1_000_000
custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
"""
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
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
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))
    register_step = RegisterStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    register_step.run(str(register_step_output_dir))
    registered_models = mlflow.tracking.MlflowClient().search_registered_models()
    latest_tag = registered_models[0].latest_versions[0].tags
    assert latest_tag[MLFLOW_SOURCE_TYPE] == "RECIPE"
    assert latest_tag[MLFLOW_RECIPE_TEMPLATE_NAME] == "regression/v1"


def test_register_uri(
    tmp_recipe_root_path: Path,
    tmp_recipe_exec_path: Path,
    registry_uri_path: Path,
):
    evaluate_step_output_dir, register_step_output_dir = setup_model_and_evaluate(
        tmp_recipe_exec_path
    )
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    registry_uri = registry_uri_path
    recipe_yaml.write_text(
        f"""
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {mlflow.get_tracking_uri()}
model_registry:
  registry_uri: {registry_uri}
  model_name: "demo_model"
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 1_000_000
      - metric: mean_absolute_error
        threshold: 1_000_000
      - metric: weighted_mean_squared_error
        threshold: 1_000_000
custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
"""
    )
    recipe_steps_dir = tmp_recipe_root_path.joinpath("steps")
    recipe_steps_dir.mkdir(parents=True)
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
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))
    register_step = RegisterStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    register_step.run(str(register_step_output_dir))
    assert mlflow.get_registry_uri() == registry_uri


def test_register_step_writes_card_with_model_link_and_version_link_on_databricks(
    monkeypatch, tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path
):
    workspace_host = "https://dev.databricks.com"
    workspace_id = 123456
    workspace_url = f"{workspace_host}?o={workspace_id}"

    monkeypatch.setenv("_DATABRICKS_WORKSPACE_HOST", workspace_host)
    monkeypatch.setenv("_DATABRICKS_WORKSPACE_ID", workspace_id)

    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
recipe: "regression/v1"
target_col: "y"
experiment:
  tracking_uri: {mlflow.get_tracking_uri()}
model_registry:
  model_name: "demo_model"
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 1_000_000
"""
    )

    evaluate_step_output_dir, register_step_output_dir = setup_model_and_evaluate(
        tmp_recipe_exec_path
    )

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    evaluate_step.run(str(evaluate_step_output_dir))

    register_step = RegisterStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    register_step.run(str(register_step_output_dir))

    train_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "train", "outputs")
    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    assert (register_step_output_dir / "card.html").exists()
    with open(register_step_output_dir / "card.html") as f:
        step_card_content = f.read()

    assert f"<a href={workspace_url}#mlflow/models/demo_model/versions/1" in step_card_content
    assert (
        f"<a href={workspace_url}#mlflow/experiments/1/runs/{run_id}/artifactPath/train/model>"
        in step_card_content
    )
