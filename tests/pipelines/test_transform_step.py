import os
from pathlib import Path

import pandas as pd

import mlflow
from mlflow.utils.file_utils import read_yaml
from mlflow.pipelines.utils.execution import _MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.steps.transform import TransformStep
from unittest import mock

# pylint: disable=unused-import
from tests.pipelines.helper_functions import tmp_pipeline_root_path

# pylint: enable=unused-import

# Sets up the transform step and returns the constructed TransformStep instance and step output dir
def set_up_transform_step(pipeline_root: Path):
    split_step_output_dir = pipeline_root.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)

    transform_step_output_dir = pipeline_root.joinpath("steps", "transform", "outputs")
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

    pipeline_yaml = pipeline_root.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
        template: "regression/v1"
        target_col: "y"
        experiment:
          name: "demo"
          tracking_uri: {tracking_uri}
        steps:
          transform:
            transformer_method: sklearn.preprocessing.StandardScaler
        """.format(
            tracking_uri=mlflow.get_tracking_uri()
        )
    )
    pipeline_config = read_yaml(pipeline_root, _PIPELINE_CONFIG_FILE_NAME)
    transform_step = TransformStep.from_pipeline_config(pipeline_config, str(pipeline_root))
    return transform_step, transform_step_output_dir


def test_transform_step_writes_onehot_encoded_dataframe_and_transformer_pkl(tmp_pipeline_root_path):
    with mock.patch.dict(
        os.environ, {_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR: str(tmp_pipeline_root_path)}
    ):
        transform_step, transform_step_output_dir = set_up_transform_step(tmp_pipeline_root_path)
        transform_step._run(str(transform_step_output_dir))

    assert os.path.exists(transform_step_output_dir / "transformed_training_data.parquet")
    transformed = pd.read_parquet(transform_step_output_dir / "transformed_training_data.parquet")
    assert len(transformed.columns) == 3
    assert os.path.exists(transform_step_output_dir / "transformer.pkl")
