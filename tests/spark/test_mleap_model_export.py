import os

from pyspark.ml.pipeline import Pipeline
from pyspark.ml.wrapper import JavaModel
import pytest

import mlflow
import mlflow.tracking
import mlflow.mleap
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir

from tests.spark.test_spark_model_export import (
    spark_model_iris,
    model_path,
)  # pylint: disable=unused-import


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
