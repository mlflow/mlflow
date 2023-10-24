import json
import os
import sys
from unittest import mock

import mleap.version
import numpy as np
import pandas as pd
import pyspark
import pytest
from packaging.version import Version
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.wrapper import JavaModel

import mlflow
import mlflow.mleap
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir

from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.pyfunc.test_spark import get_spark_session
from tests.spark.test_spark_model_export import (
    assert_register_model_called_with_local_model_path,
    iris_df,  # noqa: F401
    model_path,  # noqa: F401
    spark_custom_env,  # noqa: F401
    spark_model_iris,  # noqa: F401
)


def get_mleap_jars():
    mleap_ver = Version(
        mleap.version if isinstance(mleap.version, str) else mleap.version.__version__
    )
    scala_ver = "2.11" if mleap_ver < Version("0.18.0") else "2.12"
    jar_ver = f"{mleap_ver.major}.{mleap_ver.minor}.0"
    return ",".join(
        [
            f"ml.combust.mleap:mleap-spark-base_{scala_ver}:{jar_ver}",
            f"ml.combust.mleap:mleap-spark_{scala_ver}:{jar_ver}",
        ]
    )


@pytest.fixture(scope="module")
def spark():
    conf = pyspark.SparkConf()
    conf.set(key="spark.jars.packages", value=get_mleap_jars())
    # Exclude `net.sourceforge.f2j` to avoid `java.io.FileNotFoundException`
    conf.set(key="spark.jars.excludes", value="net.sourceforge.f2j:arpack_combined_all")
    with get_spark_session(conf) as spark_session:
        yield spark_session


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Docker image resolution for non-linux tests is not supported",
)
def test_model_deployment(spark_model_iris, model_path, spark_custom_env):
    mlflow.spark.save_model(
        spark_model_iris.model,
        path=model_path,
        conda_env=spark_custom_env,
        sample_input=spark_model_iris.spark_df,
    )

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=json.dumps(
            {
                "dataframe_split": spark_model_iris.pandas_df.to_dict(orient="split"),
            }
        ),
        content_type=mlflow.pyfunc.scoring_server.CONTENT_TYPE_JSON,
        flavor=mlflow.mleap.FLAVOR_NAME,
    )
    np.testing.assert_array_almost_equal(
        spark_model_iris.predictions,
        np.array(json.loads(scoring_response.content)["predictions"]),
        decimal=4,
    )


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


def test_mleap_module_model_save_with_unsupported_transformer_raises_serialization_exception(
    spark_model_iris, model_path
):
    from pyspark.ml.feature import VectorAssembler

    class CustomTransformer(VectorAssembler):
        def _transform(self, dataset):
            return dataset

    unsupported_pipeline = Pipeline(stages=[CustomTransformer()])
    unsupported_model = unsupported_pipeline.fit(spark_model_iris.spark_df)

    with pytest.raises(
        mlflow.mleap.MLeapSerializationException,
        match="MLeap encountered an error while serializing the model",
    ):
        mlflow.mleap.save_model(
            spark_model=unsupported_model, path=model_path, sample_input=spark_model_iris.spark_df
        )


def test_mleap_model_log(spark_model_iris):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        model_info = mlflow.spark.log_model(
            spark_model=spark_model_iris.model,
            sample_input=spark_model_iris.spark_df,
            artifact_path=artifact_path,
            registered_model_name="Model1",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        assert_register_model_called_with_local_model_path(
            mlflow.tracking._model_registry.fluent._register_model, model_uri, "Model1"
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    config_path = os.path.join(model_path, "MLmodel")
    mlflow_model = Model.load(config_path)
    assert mlflow.spark.FLAVOR_NAME in mlflow_model.flavors
    assert mlflow.mleap.FLAVOR_NAME in mlflow_model.flavors


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


def test_mleap_module_model_save_with_invalid_sample_input_type_raises_exception(
    spark_model_iris, model_path
):
    with pytest.raises(Exception, match="must be a PySpark dataframe"):
        mlflow.spark.save_model(
            spark_model=spark_model_iris.model, path=model_path, sample_input=pd.DataFrame()
        )


def test_spark_module_model_save_with_mleap_and_unsupported_transformer_raises_exception(
    spark_model_iris, model_path
):
    class CustomTransformer(JavaModel):
        def _transform(self, dataset):
            return dataset

    unsupported_pipeline = Pipeline(stages=[CustomTransformer()])
    unsupported_model = unsupported_pipeline.fit(spark_model_iris.spark_df)

    with pytest.raises(ValueError, match="CustomTransformer"):
        mlflow.spark.save_model(
            spark_model=unsupported_model, path=model_path, sample_input=spark_model_iris.spark_df
        )


def test_model_save_load_with_metadata(spark_model_iris, model_path):
    mlflow.mleap.save_model(
        spark_model=spark_model_iris.model,
        path=model_path,
        sample_input=spark_model_iris.spark_df,
        metadata={"metadata_key": "metadata_value"},
    )

    reloaded_model = Model.load(os.path.join(model_path, "MLmodel"))
    assert reloaded_model.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(spark_model_iris):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.mleap.log_model(
            spark_model=spark_model_iris.model,
            artifact_path=artifact_path,
            sample_input=spark_model_iris.spark_df,
            metadata={"metadata_key": "metadata_value"},
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    reloaded_model = Model.load(os.path.join(model_path, "MLmodel"))
    assert reloaded_model.metadata["metadata_key"] == "metadata_value"
