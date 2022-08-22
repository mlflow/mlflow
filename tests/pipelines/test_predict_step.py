import os
import tempfile
import pandas as pd
from pathlib import Path
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from sklearn.datasets import load_diabetes
from unittest import mock

from mlflow.exceptions import MlflowException
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.steps.predict import PredictStep, _SCORED_OUTPUT_FILE_NAME

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
    spark_warehouse_path = os.path.abspath(tempfile.mkdtemp())
    session = (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:1.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.warehouse.dir", str(spark_warehouse_path))
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(autouse=True)
def patch_env_manager():
    with mock.patch("mlflow.pipelines.steps.predict._ENV_MANAGER", "local"):
        yield


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
    ingest_scoring_step_output_dir = tmp_pipeline_exec_path.joinpath(
        "steps", "ingest_scoring", "outputs"
    )
    ingest_scoring_step_output_dir.mkdir(parents=True)
    X, _ = load_diabetes(as_frame=True, return_X_y=True)
    X.to_parquet(ingest_scoring_step_output_dir.joinpath("scoring-dataset.parquet"))
    predict_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "predict", "outputs")
    predict_step_output_dir.mkdir(parents=True)
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "{template_to_use}"
""".format(
            template_to_use="regression/v1",
        )
    )
    return predict_step_output_dir


@pytest.mark.parametrize("register_model", [True, False])
def test_predict_step_runs(
    tmp_pipeline_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
    register_model: bool,
):
    if register_model:
        model_name = "model_" + get_random_id()
        model_uri = train_log_and_register_model(model_name, is_dummy=True)
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
                }
            }
        },
        str(tmp_pipeline_root_path),
    )
    predict_step._run(str(predict_step_output_dir))

    # Test internal predict step output artifact
    artifact_file_name, artifact_file_extension = _SCORED_OUTPUT_FILE_NAME.split(".")
    prediction_assertions(
        predict_step_output_dir, artifact_file_extension, artifact_file_name, spark_session
    )
    # Test user specified output
    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)


def test_predict_step_uses_register_step_model_name(
    tmp_pipeline_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
):
    rm_name = "register_step_model"
    train_log_and_register_model(rm_name, is_dummy=True)

    predict_step = PredictStep.from_pipeline_config(
        {
            "steps": {
                "register": {"model_name": rm_name},
                "predict": {
                    "output_format": "parquet",
                    "output_location": str(predict_step_output_dir.joinpath("output.parquet")),
                },
            }
        },
        str(tmp_pipeline_root_path),
    )
    predict_step._run(str(predict_step_output_dir))

    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)


def test_predict_model_uri_takes_precendence_over_model_name(
    tmp_pipeline_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
):
    # Train a normal model and a dummy model
    register_step_rm_name = "register_step_model"
    train_log_and_register_model(register_step_rm_name)
    model_uri = train_log_and_register_model("predict_step_model", is_dummy=True)

    # Specify the normal model in the register step `model_name` config key and
    # the dummy model in the predict step `model_uri` config key
    predict_step = PredictStep.from_pipeline_config(
        {
            "steps": {
                "register": {"model_name": register_step_rm_name},
                "predict": {
                    "model_uri": model_uri,
                    "output_format": "parquet",
                    "output_location": str(predict_step_output_dir.joinpath("output.parquet")),
                },
            }
        },
        str(tmp_pipeline_root_path),
    )
    predict_step._run(str(predict_step_output_dir))

    # These assertions will only pass if the dummy model was used for scoring
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


@pytest.mark.parametrize("output_format", ["parquet", "delta", "table"])
def test_predict_throws_when_overwriting_data(
    tmp_pipeline_root_path: Path, predict_step_output_dir: Path, spark_session, output_format: str
):
    rm_name = "model_" + get_random_id()
    model_uri = train_log_and_register_model(rm_name, is_dummy=True)
    sdf = spark_session.createDataFrame(
        [
            (0, "a b c d e spark", 1.0),
            (1, "b d", 0.0),
            (2, "spark f g h", 1.0),
            (3, "hadoop mapreduce", 0.0),
        ],
        ["id", "text", "label"],
    )
    if output_format == "table":
        output_path = get_random_id()
        sdf.write.format("delta").saveAsTable(output_path)
    else:
        output_file = "output_{}.{}".format(get_random_id(), output_format)
        output_path = str(predict_step_output_dir / output_file)
        sdf.coalesce(1).write.format(output_format).save(output_path)

    pipeline_config = {
        "steps": {
            "predict": {
                "model_uri": model_uri,
                "output_format": output_format,
                "output_location": output_path,
            }
        }
    }

    predict_step = PredictStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    with pytest.raises(AnalysisException, match="already exists"):
        predict_step._run(str(predict_step_output_dir))


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_predict_throws_when_improperly_configured():
    with pytest.raises(MlflowException, match="Config for predict step is not found"):
        PredictStep.from_pipeline_config(
            pipeline_config={},
            pipeline_root=os.getcwd(),
        )

    for required_key in ["output_format", "output_location"]:
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


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_predict_throws_when_no_model_is_specified():
    pipeline_config = {
        "steps": {
            "predict": {
                "output_format": "parquet",
                "output_location": "random/path",
            }
        }
    }
    with pytest.raises(MlflowException, match="No model specified for batch scoring"):
        PredictStep.from_pipeline_config(
            pipeline_config=pipeline_config,
            pipeline_root=os.getcwd(),
        )


def test_predict_skips_profiling_when_specified(
    tmp_pipeline_root_path: Path,
    predict_step_output_dir: Path,
):
    model_name = "model_" + get_random_id()
    model_uri = train_log_and_register_model(model_name, is_dummy=True)
    with mock.patch("mlflow.pipelines.utils.step.get_pandas_data_profile") as mock_profiling:
        PredictStep.from_pipeline_config(
            {
                "steps": {
                    "predict": {
                        "model_uri": model_uri,
                        "output_format": "parquet",
                        "output_location": str(predict_step_output_dir.joinpath("output.parquet")),
                        "skip_data_profiling": True,
                    }
                }
            },
            str(tmp_pipeline_root_path),
        ).run(str(predict_step_output_dir))

    expected_step_card_path = os.path.join(str(predict_step_output_dir), "card.html")
    with open(expected_step_card_path, "r") as f:
        step_card_html_content = f.read()
    assert "Profile of Scored Dataset" not in step_card_html_content
    mock_profiling.assert_not_called()
