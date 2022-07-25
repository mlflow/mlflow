import os
from xmlrpc.client import Boolean
import pandas as pd
from pathlib import Path
import pytest
from pyspark.sql import SparkSession
from sklearn.datasets import load_diabetes

from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import read_yaml
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.steps.predict import PredictStep
from mlflow.pipelines.steps.preprocessing import _PREPROCESSED_OUTPUT_FILE_NAME

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_test_pipeline_directory,
    enter_pipeline_example_directory,
    get_random_id,
    tmp_pipeline_exec_path,
    tmp_pipeline_root_path,
    train_and_log_model,
    train_log_and_register_model,
)  # pylint: enable=unused-import


@pytest.fixture(scope="module", autouse=True)
def spark_session():
    session = (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:1.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(autouse=True)
def predict_input(tmp_pipeline_exec_path: Path):
    proprocessing_step_output_dir = tmp_pipeline_exec_path.joinpath(
        "steps", "preprocessing", "outputs"
    )
    proprocessing_step_output_dir.mkdir(parents=True)
    X, _ = load_diabetes(as_frame=True, return_X_y=True)
    X.to_parquet(proprocessing_step_output_dir.joinpath(_PREPROCESSED_OUTPUT_FILE_NAME))


@pytest.mark.parametrize("register_model", [True, False])
def test_predict_step_run(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path, register_model: Boolean
):
    if register_model:
        rm_name = "model_" + get_random_id()
        model_uri = train_log_and_register_model(rm_name)
    else:
        run_id, _ = train_and_log_model()
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id, artifact_path="train/model"
        )

    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "batch_scoring/v1"
steps:
  predict:
    model_uri: {model_uri}
    output_format: parquet
""".format(
            model_uri=model_uri,
        )
    )
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    predict_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "predict", "outputs")
    predict_step_output_dir.mkdir(parents=True)

    predict_step = PredictStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    predict_step._run(str(predict_step_output_dir))

    (predict_step_output_dir / "scored.parquet").exists()
    assert "prediction" in pd.read_parquet(predict_step_output_dir / "scored.parquet")


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_predict_throws_when_improperly_configured():
    with pytest.raises(MlflowException, match="Config for predict step is not found"):
        PredictStep.from_pipeline_config(
            pipeline_config={},
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(
        MlflowException, match="The `output_format` configuration key must be specified"
    ):
        PredictStep.from_pipeline_config(
            pipeline_config={
                "steps": {
                    "predict": {
                        "model_uri": "my model",
                        # Missing output_format
                    }
                }
            },
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(
        MlflowException, match="The `model_uri` configuration key must be specified"
    ):
        PredictStep.from_pipeline_config(
            pipeline_config={
                "steps": {
                    "predict": {
                        # Missing model_uri
                        "output_format": "parquet",
                    }
                }
            },
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(
        MlflowException, match="Invalid `output_format` in predict step configuration"
    ):
        PredictStep.from_pipeline_config(
            pipeline_config={
                "steps": {
                    "predict": {
                        "model_uri": "my model",
                        "output_format": "fancy_format",
                    }
                }
            },
            pipeline_root=os.getcwd(),
        )
