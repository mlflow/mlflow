import json
import os
from unittest import mock

import numpy as np
import pyspark
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.wrapper import JavaModel
import pytest

import mlflow
import mlflow.mleap
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.pyfunc.test_spark import get_spark_session


from tests.spark.test_spark_model_export import (  # pylint: disable=unused-import
    model_path,
    iris_df,
    spark_model_iris,
    spark_custom_env,
)


@pytest.fixture(scope="session", autouse=True)
def spark_context():
    conf = pyspark.SparkConf()
    conf.set(
        key="spark.jars.packages",
        value=(
            "ml.combust.mleap:mleap-spark-base_2.11:0.12.0,"
            "ml.combust.mleap:mleap-spark_2.11:0.12.0"
        ),
    )
    spark_session = get_spark_session(conf)
    return spark_session.sparkContext


@pytest.mark.large
def test_model_deployment(spark_model_iris, model_path, spark_custom_env):
    mlflow.spark.save_model(
        spark_model_iris.model,
        path=model_path,
        conda_env=spark_custom_env,
        sample_input=spark_model_iris.spark_df,
    )

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=spark_model_iris.pandas_df.to_json(orient="split"),
        content_type=mlflow.pyfunc.scoring_server.CONTENT_TYPE_JSON,
        flavor=mlflow.mleap.FLAVOR_NAME,
    )
    np.testing.assert_array_almost_equal(
        spark_model_iris.predictions, np.array(json.loads(scoring_response.content)), decimal=4
    )


@pytest.mark.large
def test_mleap_module_model_save_with_relative_path_and_valid_sample_input_produces_mleap_flavor(
    spark_model_iris,
):
    with TempDir(chdr=True) as tmp:
        model_path = os.path.basename(tmp.path("model"))
        mlflow_model = Model()
        mlflow.mleap.save_model(
            spark_model=spark_model_iris.model,
            path=model_path,
            sample_input=spark_model_iris.spark_df,
            mlflow_model=mlflow_model,
        )
        assert mlflow.mleap.FLAVOR_NAME in mlflow_model.flavors

        config_path = os.path.join(model_path, "MLmodel")
        assert os.path.exists(config_path)
        config = Model.load(config_path)
        assert mlflow.mleap.FLAVOR_NAME in config.flavors


@pytest.mark.large
def test_mleap_module_model_save_with_absolute_path_and_valid_sample_input_produces_mleap_flavor(
    spark_model_iris, model_path
):
    model_path = os.path.abspath(model_path)
    mlflow_model = Model()
    mlflow.mleap.save_model(
        spark_model=spark_model_iris.model,
        path=model_path,
        sample_input=spark_model_iris.spark_df,
        mlflow_model=mlflow_model,
    )
    assert mlflow.mleap.FLAVOR_NAME in mlflow_model.flavors

    config_path = os.path.join(model_path, "MLmodel")
    assert os.path.exists(config_path)
    config = Model.load(config_path)
    assert mlflow.mleap.FLAVOR_NAME in config.flavors


@pytest.mark.large
def test_mleap_module_model_save_with_unsupported_transformer_raises_serialization_exception(
    spark_model_iris, model_path
):
    class CustomTransformer(JavaModel):
        def _transform(self, dataset):
            return dataset

    unsupported_pipeline = Pipeline(stages=[CustomTransformer()])
    unsupported_model = unsupported_pipeline.fit(spark_model_iris.spark_df)

    with pytest.raises(mlflow.mleap.MLeapSerializationException):
        mlflow.mleap.save_model(
            spark_model=unsupported_model, path=model_path, sample_input=spark_model_iris.spark_df
        )


@pytest.mark.large
def test_mleap_model_log(spark_model_iris):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.spark.log_model(
            spark_model=spark_model_iris.model,
            sample_input=spark_model_iris.spark_df,
            artifact_path=artifact_path,
            registered_model_name="Model1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "Model1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    config_path = os.path.join(model_path, "MLmodel")
    mlflow_model = Model.load(config_path)
    assert mlflow.spark.FLAVOR_NAME in mlflow_model.flavors
    assert mlflow.mleap.FLAVOR_NAME in mlflow_model.flavors


@pytest.mark.large
def test_spark_module_model_save_with_relative_path_and_valid_sample_input_produces_mleap_flavor(
    spark_model_iris,
):
    with TempDir(chdr=True) as tmp:
        model_path = os.path.basename(tmp.path("model"))
        mlflow_model = Model()
        mlflow.spark.save_model(
            spark_model=spark_model_iris.model,
            path=model_path,
            sample_input=spark_model_iris.spark_df,
            mlflow_model=mlflow_model,
        )
        assert mlflow.mleap.FLAVOR_NAME in mlflow_model.flavors

        config_path = os.path.join(model_path, "MLmodel")
        assert os.path.exists(config_path)
        config = Model.load(config_path)
        assert mlflow.mleap.FLAVOR_NAME in config.flavors
