import os
from xmlrpc.client import Boolean
import pandas as pd
from pathlib import Path
import pytest
from pyspark.sql import SparkSession
from sklearn.datasets import load_diabetes

from mlflow.exceptions import MlflowException
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


def prediction_assertions(output_dir: Path, output_format: str, output_name: str, spark):
    if output_format == "table":
        sdf = spark.table(output_name)
        df = sdf.toPandas()
    else:
        file_name = "{}.{}".format(output_name, output_format)
        (output_dir / file_name).exists()
        df = pd.read_parquet(output_dir / file_name)
    assert "prediction" in df
    assert (df.prediction == 42).all()


# Sets up predict step run and returns output directory
@pytest.fixture(autouse=True)
def predict_step_output_dir(tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path):
    proprocessing_step_output_dir = tmp_pipeline_exec_path.joinpath(
        "steps", "preprocessing", "outputs"
    )
    proprocessing_step_output_dir.mkdir(parents=True)
    X, _ = load_diabetes(as_frame=True, return_X_y=True)
    X.to_parquet(proprocessing_step_output_dir.joinpath(_PREPROCESSED_OUTPUT_FILE_NAME))
    predict_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "predict", "outputs")
    predict_step_output_dir.mkdir(parents=True)
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "batch_scoring/v1"
"""
    )
    return predict_step_output_dir


@pytest.mark.parametrize("register_model", [True, False])
def test_predict_step_runs(
    tmp_pipeline_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
    register_model: Boolean,
):
    if register_model:
        rm_name = "model_" + get_random_id()
        model_uri = train_log_and_register_model(rm_name, is_dummy=True)
    else:
        run_id, _ = train_and_log_model(is_dummy=True)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id, artifact_path="train/model"
        )

    predict_step = PredictStep.from_pipeline_config(
        {
            "steps": {
                "predict": {
                    "model_uri": model_uri,
                    "output_format": "parquet",
                    "output_location": str(predict_step_output_dir.joinpath("output.parquet")),
                    "_disable_env_restoration": True,
                }
            }
        },
        str(tmp_pipeline_root_path),
    )
    predict_step._run(str(predict_step_output_dir))

    prediction_assertions(predict_step_output_dir, "parquet", "scored", spark_session)
    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)


@pytest.mark.parametrize("output_format", ["parquet", "delta", "table"])
def test_predict_step_output_formats(
    tmp_pipeline_root_path: Path, predict_step_output_dir: Path, spark_session, output_format: str
):
    rm_name = "model_" + get_random_id()
    output_name = "output_" + get_random_id()
    model_uri = train_log_and_register_model(rm_name, is_dummy=True)

    pipeline_config = {
        "steps": {
            "predict": {
                "model_uri": model_uri,
                "output_format": output_format,
                "_disable_env_restoration": True,
            }
        }
    }
    if output_format == "table":
        pipeline_config["steps"]["predict"]["output_location"] = output_name
    else:
        file_name = "{}.{}".format(output_name, output_format)
        pipeline_config["steps"]["predict"]["output_location"] = str(
            predict_step_output_dir / file_name
        )
    predict_step = PredictStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    predict_step._run(str(predict_step_output_dir))
    prediction_assertions(predict_step_output_dir, output_format, output_name, spark_session)


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_predict_throws_when_improperly_configured():
    with pytest.raises(MlflowException, match="Config for predict step is not found"):
        PredictStep.from_pipeline_config(
            pipeline_config={},
            pipeline_root=os.getcwd(),
        )

    for required_key in ["model_uri", "output_format", "output_location"]:
        pipeline_config = {
            "steps": {
                "predict": {
                    "model_uri": "models:/taxi_fare_regressor/Production",
                    "output_format": "parquet",
                    "output_location": "random/path",
                }
            }
        }
        pipeline_config["steps"]["predict"].pop(required_key)
        with pytest.raises(
            MlflowException, match=f"The `{required_key}` configuration key must be specified"
        ):
            PredictStep.from_pipeline_config(
                pipeline_config=pipeline_config,
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
                        "output_location": "random/path",
                    }
                }
            },
            pipeline_root=os.getcwd(),
        )
