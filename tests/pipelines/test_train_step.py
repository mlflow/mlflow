import os
import cloudpickle
from pathlib import Path

import pandas as pd

import mlflow
import sklearn.compose
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import read_yaml
from mlflow.pipelines.utils.execution import _MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.steps.train import TrainStep
from unittest import mock

# pylint: disable=unused-import
from tests.pipelines.helper_functions import tmp_pipeline_root_path

# pylint: enable=unused-import

# Sets up the train step and returns the constructed TrainStep instance and step output dir
def set_up_train_step(pipeline_root: Path):
    split_step_output_dir = pipeline_root.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)

    transform_step_output_dir = pipeline_root.joinpath("steps", "transform", "outputs")
    transform_step_output_dir.mkdir(parents=True)

    train_step_output_dir = pipeline_root.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)

    transformer = sklearn.preprocessing.FunctionTransformer(func=(lambda x: x))
    with open(os.path.join(transform_step_output_dir, "transformer.pkl"), "wb") as f:
        cloudpickle.dump(transformer, f)

    num_rows = 100
    # use for train and validation, also for split
    transformed_dataset = pd.DataFrame(
        {
            "a": list(range(num_rows)),
            "b": list(range(num_rows)),
            "y": [float(i % 2) for i in range(num_rows)],
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

    pipeline_yaml = pipeline_root.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
        template: "regression/v1"
        target_col: "y"
        profile: "test_profile"
        run_args:
            step: "train"
        experiment:
          name: "demo"
          tracking_uri: {tracking_uri}
        steps:
          train:
            estimator_method: sklearn.linear_model.SGDRegressor
        """.format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    pipeline_config = read_yaml(pipeline_root, _PIPELINE_CONFIG_FILE_NAME)
    train_step = TrainStep.from_pipeline_config(pipeline_config, str(pipeline_root))
    return train_step, train_step_output_dir


def test_train_steps_writes_model_pkl_and_card(tmp_pipeline_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_pipeline_root_path)}
    ):
        train_step, train_step_output_dir = set_up_train_step(tmp_pipeline_root_path)
        train_step._run(str(train_step_output_dir))

    assert (train_step_output_dir / "model/model.pkl").exists()
    assert (train_step_output_dir / "card.html").exists()


def test_train_steps_writes_card_with_model_and_run_links_on_databricks(
    monkeypatch, tmp_pipeline_root_path
):
    workspace_host = "https://dev.databricks.com"
    workspace_id = 123456
    workspace_url = f"{workspace_host}?o={workspace_id}"

    monkeypatch.setenv("_DATABRICKS_WORKSPACE_HOST", workspace_host)
    monkeypatch.setenv("_DATABRICKS_WORKSPACE_ID", workspace_id)
    monkeypatch.setenv(_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR, str(tmp_pipeline_root_path))

    train_step, train_step_output_dir = set_up_train_step(tmp_pipeline_root_path)
    train_step._run(str(train_step_output_dir))

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


def test_train_steps_autologs(tmp_pipeline_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_pipeline_root_path)}
    ):
        train_step, train_step_output_dir = set_up_train_step(tmp_pipeline_root_path)
        train_step._run(str(train_step_output_dir))

    assert os.path.exists(train_step_output_dir / "run_id")

    # assert eval output exists
    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    metrics = MlflowClient().get_run(run_id).data.metrics
    params = MlflowClient().get_run(run_id).data.params
    assert "training_score" in metrics
    assert "epsilon" in params


def test_train_steps_with_correct_tags(tmp_pipeline_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_pipeline_root_path)}
    ):
        train_step, train_step_output_dir = set_up_train_step(tmp_pipeline_root_path)
        train_step._run(str(train_step_output_dir))

    assert os.path.exists(train_step_output_dir / "run_id")

    # assert eval output exists
    with open(train_step_output_dir / "run_id") as f:
        run_id = f.read()

    tags = MlflowClient().get_run(run_id).data.tags
    assert tags["mlflow.source.type"] == "PIPELINE"
    assert tags["mlflow.pipeline.template.name"] == "regression/v1"
    assert tags["mlflow.pipeline.step.name"] == "train"
    assert tags["mlflow.pipeline.profile.name"] == "test_profile"
