import os
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from sklearn.datasets import load_diabetes

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.recipes.artifacts import RegisteredModelVersionInfo
from mlflow.recipes.steps.predict import _INPUT_FILE_NAME, _SCORED_OUTPUT_FILE_NAME, PredictStep
from mlflow.recipes.steps.register import _REGISTERED_MV_INFO_FILE
from mlflow.recipes.utils import _RECIPE_CONFIG_FILE_NAME
from mlflow.utils.file_utils import read_yaml

from tests.recipes.helper_functions import (
    get_random_id,
    train_and_log_model,
    train_log_and_register_model,
)


@pytest.fixture(scope="module")
def spark_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            SparkSession.builder.master("local[*]")
            .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
            )
            .config("spark.sql.warehouse.dir", str(tmpdir))
            .getOrCreate()
        ) as session:
            yield session


@pytest.fixture(autouse=True)
def patch_env_manager():
    with mock.patch("mlflow.recipes.steps.predict._ENV_MANAGER", "local"):
        yield


def prediction_assertions(output_dir: Path, output_format: str, output_name: str, spark):
    if output_format == "table":
        sdf = spark.table(output_name)
        df = sdf.toPandas()
    else:
        file_name = f"{output_name}.{output_format}"
        (output_dir / file_name).exists()
        df = pd.read_parquet(output_dir / file_name)
    assert "prediction" in df
    assert (df.prediction == 42).all()


# Sets up predict step run and returns output directory
@pytest.fixture(autouse=True)
def predict_step_output_dir(tmp_recipe_root_path: Path, tmp_recipe_exec_path: Path):
    ingest_scoring_step_output_dir = tmp_recipe_exec_path.joinpath(
        "steps", "ingest_scoring", "outputs"
    )
    ingest_scoring_step_output_dir.mkdir(parents=True)
    X, _ = load_diabetes(as_frame=True, return_X_y=True)
    X.to_parquet(ingest_scoring_step_output_dir.joinpath(_INPUT_FILE_NAME))
    predict_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "predict", "outputs")
    predict_step_output_dir.mkdir(parents=True)
    recipe_yaml = tmp_recipe_root_path.joinpath(_RECIPE_CONFIG_FILE_NAME)
    recipe_yaml.write_text(
        f"""
recipe: "regression/v1"
experiment:
  name: "test"
  tracking_uri: {mlflow.get_tracking_uri()}
"""
    )
    return predict_step_output_dir


@pytest.mark.parametrize("register_model", [True, False])
def test_predict_step_runs(
    tmp_recipe_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
    register_model: bool,
):
    if register_model:
        model_name = "model_" + get_random_id()
        model_uri = train_log_and_register_model(model_name, is_dummy=True)
    else:
        run_id, _ = train_and_log_model(is_dummy=True)
        model_uri = f"runs:/{run_id}/train/model"

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    recipe_config.update(
        {
            "steps": {
                "predict": {
                    "output": {
                        "using": "parquet",
                        "location": str(predict_step_output_dir.joinpath("output.parquet")),
                    },
                    "model_uri": model_uri,
                },
            },
        }
    )
    predict_step = PredictStep.from_recipe_config(
        recipe_config,
        str(tmp_recipe_root_path),
    )
    predict_step.run(str(predict_step_output_dir))

    # Test internal predict step output artifact
    artifact_file_name, artifact_file_extension = _SCORED_OUTPUT_FILE_NAME.split(".")
    prediction_assertions(
        predict_step_output_dir, artifact_file_extension, artifact_file_name, spark_session
    )
    # Test user specified output
    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)


def test_predict_step_uses_register_step_model_name(
    tmp_recipe_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
):
    rm_name = "register_step_model"
    train_log_and_register_model(rm_name, is_dummy=True)

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    recipe_config.update(
        {
            "model_registry": {"model_name": rm_name},
            "steps": {
                "predict": {
                    "output": {
                        "using": "parquet",
                        "location": str(predict_step_output_dir.joinpath("output.parquet")),
                    }
                },
            },
        }
    )
    predict_step = PredictStep.from_recipe_config(
        recipe_config,
        str(tmp_recipe_root_path),
    )
    predict_step.run(str(predict_step_output_dir))

    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)


def test_predict_step_uses_register_step_output(
    tmp_recipe_root_path: Path,
    tmp_recipe_exec_path: Path,
    predict_step_output_dir: Path,
    spark_session,
):
    # Create two versions for a registered_model. v1 is a dummy model. v2 is a normal model.
    register_step_rm_name = "register_step_model_" + get_random_id()
    train_log_and_register_model(register_step_rm_name, is_dummy=True)
    train_log_and_register_model(register_step_rm_name)

    # Write v1 to the output directory of the register step
    register_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "register", "outputs")
    register_step_output_dir.mkdir(parents=True)
    registered_model_info = RegisteredModelVersionInfo(name=register_step_rm_name, version=1)
    registered_model_info.to_json(path=str(register_step_output_dir / _REGISTERED_MV_INFO_FILE))

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    recipe_config.update(
        {
            "model_registry": {"model_name": register_step_rm_name},
            "steps": {
                "predict": {
                    "output": {
                        "using": "parquet",
                        "location": str(predict_step_output_dir.joinpath("output.parquet")),
                    }
                },
            },
        }
    )
    predict_step = PredictStep.from_recipe_config(
        recipe_config,
        str(tmp_recipe_root_path),
    )
    predict_step.run(str(predict_step_output_dir))

    # These assertions will only pass if the dummy model was used for scoring
    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)


def test_predict_model_uri_takes_precendence_over_model_name(
    tmp_recipe_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
):
    # Train a normal model and a dummy model
    register_step_rm_name = "register_step_model"
    train_log_and_register_model(register_step_rm_name)
    model_uri = train_log_and_register_model("predict_step_model", is_dummy=True)

    # Specify the normal model in the register step `model_name` config key and
    # the dummy model in the predict step `model_uri` config key
    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    recipe_config.update(
        {
            "model_registry": {"model_name": register_step_rm_name},
            "steps": {
                "predict": {
                    "output": {
                        "using": "parquet",
                        "location": str(predict_step_output_dir.joinpath("output.parquet")),
                    },
                    "model_uri": model_uri,
                },
            },
        }
    )
    predict_step = PredictStep.from_recipe_config(
        recipe_config,
        str(tmp_recipe_root_path),
    )
    predict_step.run(str(predict_step_output_dir))

    # These assertions will only pass if the dummy model was used for scoring
    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)


@pytest.mark.parametrize("output_format", ["parquet", "delta", "table"])
def test_predict_step_output_formats(
    tmp_recipe_root_path: Path, predict_step_output_dir: Path, spark_session, output_format: str
):
    rm_name = "model_" + get_random_id()
    output_name = "output_" + get_random_id()
    model_uri = train_log_and_register_model(rm_name, is_dummy=True)

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    recipe_config.update(
        {
            "steps": {
                "predict": {
                    "output": {
                        "using": output_format,
                    },
                    "model_uri": model_uri,
                },
            },
        }
    )
    if output_format == "table":
        recipe_config["steps"]["predict"]["output"]["location"] = output_name
    else:
        file_name = f"{output_name}.{output_format}"
        recipe_config["steps"]["predict"]["output"]["location"] = str(
            predict_step_output_dir / file_name
        )
    predict_step = PredictStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    predict_step.run(str(predict_step_output_dir))
    prediction_assertions(predict_step_output_dir, output_format, output_name, spark_session)


@pytest.mark.parametrize("using", ["parquet", "delta", "table"])
@pytest.mark.parametrize("save_mode", ["default", "overwrite"])
def test_predict_correctly_handles_save_modes(
    tmp_recipe_root_path: Path,
    predict_step_output_dir: Path,
    spark_session,
    using: str,
    save_mode: str,
):
    rm_name = "model_" + get_random_id()
    model_uri = train_log_and_register_model(rm_name, is_dummy=True)
    # We create a dataframe with the same schema as the expected output dataframe to avoid schema
    # incompatibility during overwrite.
    sdf = spark_session.createDataFrame(
        [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
        ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "prediction"],
    )
    if using == "table":
        output_path = "table_" + get_random_id()
        sdf.write.format("delta").saveAsTable(output_path)
    else:
        output_file = f"output_{get_random_id()}.{using}"
        output_path = str(predict_step_output_dir / output_file)
        sdf.coalesce(1).write.format(using).save(output_path)

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    recipe_config.update(
        {
            "steps": {
                "predict": {
                    "output": {
                        "using": using,
                        "location": output_path,
                    },
                    "model_uri": model_uri,
                    "save_mode": save_mode,
                },
            },
        }
    )

    predict_step = PredictStep.from_recipe_config(recipe_config, str(tmp_recipe_root_path))
    if save_mode == "overwrite":
        predict_step.run(str(predict_step_output_dir))
    else:
        with pytest.raises(MlflowException, match="already populated"):
            predict_step.run(str(predict_step_output_dir))


@pytest.mark.usefixtures("enter_test_recipe_directory")
def test_predict_throws_when_improperly_configured():
    for required_key in ["using", "location"]:
        recipe_config = {
            "steps": {
                "predict": {
                    "output": {
                        "using": "parquet",
                        "location": "random/path",
                    },
                    "model_uri": "models:/taxi_fare_regressor/Production",
                },
            },
        }
        recipe_config["steps"]["predict"]["output"].pop(required_key)
        predict_step = PredictStep.from_recipe_config(
            recipe_config=recipe_config,
            recipe_root=os.getcwd(),
        )
        with pytest.raises(
            MlflowException, match=f"The `{required_key}` configuration key must be specified"
        ):
            predict_step._validate_and_apply_step_config()

    predict_step = PredictStep.from_recipe_config(
        recipe_config={
            "steps": {
                "predict": {
                    "output": {
                        "using": "fancy_format",
                        "location": "random/path",
                    },
                    "model_uri": "my model",
                },
            },
        },
        recipe_root=os.getcwd(),
    )
    with pytest.raises(MlflowException, match="Invalid `using` in predict step configuration"):
        predict_step._validate_and_apply_step_config()


@pytest.mark.usefixtures("enter_test_recipe_directory")
def test_predict_throws_when_no_model_is_specified():
    recipe_config = {
        "steps": {
            "predict": {
                "output": {
                    "using": "parquet",
                    "location": "random/path",
                }
            }
        }
    }
    predict_step = PredictStep.from_recipe_config(
        recipe_config=recipe_config,
        recipe_root=os.getcwd(),
    )
    with pytest.raises(MlflowException, match="No model specified for batch scoring"):
        predict_step._validate_and_apply_step_config()


def test_predict_skips_profiling_when_specified(
    tmp_recipe_root_path: Path,
    predict_step_output_dir: Path,
):
    model_name = "model_" + get_random_id()
    model_uri = train_log_and_register_model(model_name, is_dummy=True)
    with mock.patch("mlflow.recipes.utils.step.get_pandas_data_profiles") as mock_profiling:
        recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
        recipe_config.update(
            {
                "steps": {
                    "predict": {
                        "output": {
                            "using": "parquet",
                            "location": str(predict_step_output_dir.joinpath("output.parquet")),
                        },
                        "model_uri": model_uri,
                        "skip_data_profiling": True,
                    },
                },
            }
        )
        predict_step = PredictStep.from_recipe_config(
            recipe_config,
            str(tmp_recipe_root_path),
        )
        predict_step.run(str(predict_step_output_dir))

    expected_step_card_path = os.path.join(str(predict_step_output_dir), "card.html")
    with open(expected_step_card_path) as f:
        step_card_html_content = f.read()
    assert "Profile of Scored Dataset" not in step_card_html_content
    mock_profiling.assert_not_called()


def test_predict_uses_registry_uri(
    tmp_recipe_root_path: Path,
    predict_step_output_dir: Path,
    registry_uri_path: Path,
):
    registry_uri = registry_uri_path
    model_name = "model_" + get_random_id()
    mlflow.set_registry_uri(registry_uri)
    model_uri = train_log_and_register_model(model_name, is_dummy=True)
    # reset model registry
    mlflow.set_registry_uri("")

    recipe_config = read_yaml(tmp_recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    recipe_config.update({"model_registry": {"registry_uri": str(registry_uri)}})
    recipe_config.update(
        {
            "steps": {
                "predict": {
                    "output": {
                        "using": "parquet",
                        "location": str(predict_step_output_dir.joinpath("output.parquet")),
                    },
                    "model_uri": model_uri,
                },
            },
        }
    )
    PredictStep.from_recipe_config(
        recipe_config,
        str(tmp_recipe_root_path),
    ).run(str(predict_step_output_dir))
    assert mlflow.get_registry_uri() == registry_uri
    prediction_assertions(predict_step_output_dir, "parquet", "output", spark_session)
