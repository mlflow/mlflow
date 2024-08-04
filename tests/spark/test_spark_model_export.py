import inspect
import json
import logging
import os
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyspark
import pytest
import yaml
from packaging.version import Version
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from sklearn import datasets

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.tracking
import mlflow.utils.file_utils
from mlflow import pyfunc
from mlflow.entities.model_registry import ModelVersion
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import _read_example
from mlflow.spark import _add_code_from_conf_to_system_path
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_conda_env_requirements,
    _compare_logged_code_paths,
    _get_pip_deps,
    _mlflow_major_version_string,
    assert_register_model_called_with_local_model_path,
    score_model_in_sagemaker_docker_container,
)
from tests.pyfunc.test_spark import get_spark_session, score_model_as_udf
from tests.store.artifact.constants import MODELS_ARTIFACT_REPOSITORY

_logger = logging.getLogger(__name__)


@pytest.fixture
def spark_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    additional_pip_deps = ["/opt/mlflow", "pyspark", "pytest"]
    if Version(pyspark.__version__) <= Version("3.3.2"):
        # Versions of PySpark <= 3.3.2 are incompatible with pandas >= 2
        additional_pip_deps.append("pandas<2")
    _mlflow_conda_env(conda_env, additional_pip_deps=additional_pip_deps)
    return conda_env


SparkModelWithData = namedtuple(
    "SparkModelWithData", ["model", "spark_df", "pandas_df", "predictions"]
)


def _get_spark_session_with_retry(max_tries=3):
    conf = pyspark.SparkConf()
    for attempt in range(max_tries):
        try:
            return get_spark_session(conf)
        except Exception as e:
            if attempt >= max_tries - 1:
                raise
            _logger.exception(
                f"Attempt {attempt} to create a SparkSession failed ({e!r}), retrying..."
            )


# Specify `autouse=True` to ensure that a context is created
# before any tests are executed. This ensures that the Hadoop filesystem
# does not create its own SparkContext without the MLeap libraries required by
# other tests.
@pytest.fixture(scope="module")
def spark():
    if Version(pyspark.__version__) < Version("3.1"):
        # A workaround for this issue:
        # https://stackoverflow.com/questions/62109276/errorjava-lang-unsupportedoperationexception-for-pyspark-pandas-udf-documenta
        spark_home = (
            os.environ.get("SPARK_HOME")
            if "SPARK_HOME" in os.environ
            else os.path.dirname(pyspark.__file__)
        )
        conf_dir = os.path.join(spark_home, "conf")
        os.makedirs(conf_dir, exist_ok=True)
        with open(os.path.join(conf_dir, "spark-defaults.conf"), "w") as f:
            conf = """
spark.driver.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"
spark.executor.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"
"""
            f.write(conf)

    with _get_spark_session_with_retry() as spark:
        yield spark


def iris_pandas_df():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = ["0", "1", "2", "3"]
    df = pd.DataFrame(X, columns=feature_names)  # to make spark_udf work
    df["label"] = pd.Series(y)
    return df


@pytest.fixture(scope="module")
def iris_df(spark):
    pdf = iris_pandas_df()
    feature_names = list(pdf.drop("label", axis=1).columns)
    iris_spark_df = spark.createDataFrame(pdf)
    return feature_names, pdf, iris_spark_df


@pytest.fixture(scope="module")
def iris_signature():
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(name="0", type=DataType.double),
                ColSpec(name="1", type=DataType.double),
                ColSpec(name="2", type=DataType.double),
                ColSpec(name="3", type=DataType.double),
            ]
        ),
        outputs=Schema([ColSpec(type=DataType.double)]),
    )


@pytest.fixture(scope="module")
def spark_model_iris(iris_df):
    feature_names, iris_pandas_df, iris_spark_df = iris_df
    assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
    lr = LogisticRegression(maxIter=50, regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[assembler, lr])
    # Fit the model
    model = pipeline.fit(iris_spark_df)
    preds_df = model.transform(iris_spark_df)
    preds = [x.prediction for x in preds_df.select("prediction").collect()]
    return SparkModelWithData(
        model=model, spark_df=iris_spark_df, pandas_df=iris_pandas_df, predictions=preds
    )


@pytest.fixture(scope="module")
def spark_model_transformer(iris_df):
    feature_names, iris_pandas_df, iris_spark_df = iris_df
    assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
    # Fit the model
    preds_df = assembler.transform(iris_spark_df)
    preds = [x.features for x in preds_df.select("features").collect()]
    return SparkModelWithData(
        model=assembler, spark_df=iris_spark_df, pandas_df=iris_pandas_df, predictions=preds
    )


@pytest.fixture(scope="module")
def spark_model_estimator(iris_df):
    feature_names, iris_pandas_df, iris_spark_df = iris_df
    assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
    features_df = assembler.transform(iris_spark_df)
    lr = LogisticRegression(maxIter=50, regParam=0.1, elasticNetParam=0.8)
    # Fit the model
    model = lr.fit(features_df)
    preds_df = model.transform(features_df)
    preds = [x.prediction for x in preds_df.select("prediction").collect()]
    return SparkModelWithData(
        model=model, spark_df=features_df, pandas_df=iris_pandas_df, predictions=preds
    )


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.mark.usefixtures("spark")
def test_hadoop_filesystem(tmp_path):
    # copy local dir to and back from HadoopFS and make sure the results match
    from mlflow.spark import _HadoopFileSystem as FS

    test_dir_0 = os.path.join(tmp_path, "expected")
    test_file_0 = os.path.join(test_dir_0, "root", "file_0")
    test_dir_1 = os.path.join(test_dir_0, "root", "subdir")
    test_file_1 = os.path.join(test_dir_1, "file_1")
    os.makedirs(os.path.dirname(test_file_0))
    with open(test_file_0, "w") as f:
        f.write("test0")
    os.makedirs(os.path.dirname(test_file_1))
    with open(test_file_1, "w") as f:
        f.write("test1")
    remote = "/tmp/mlflow/test0"
    # File should not be copied in this case
    assert os.path.abspath(test_dir_0) == FS.maybe_copy_from_local_file(test_dir_0, remote)
    FS.copy_from_local_file(test_dir_0, remote, remove_src=False)
    local = os.path.join(tmp_path, "actual")
    FS.copy_to_local_file(remote, local, remove_src=True)
    assert sorted(os.listdir(os.path.join(local, "root"))) == sorted(
        ["subdir", "file_0", ".file_0.crc"]
    )
    assert sorted(os.listdir(os.path.join(local, "root", "subdir"))) == sorted(
        ["file_1", ".file_1.crc"]
    )
    # compare the files
    with open(os.path.join(test_dir_0, "root", "file_0")) as expected_f:
        with open(os.path.join(local, "root", "file_0")) as actual_f:
            assert expected_f.read() == actual_f.read()
    with open(os.path.join(test_dir_0, "root", "subdir", "file_1")) as expected_f:
        with open(os.path.join(local, "root", "subdir", "file_1")) as actual_f:
            assert expected_f.read() == actual_f.read()

    # make sure we cleanup
    assert not os.path.exists(FS._remote_path(remote).toString())  # skip file: prefix
    FS.copy_from_local_file(test_dir_0, remote, remove_src=False)
    assert os.path.exists(FS._remote_path(remote).toString())  # skip file: prefix
    FS.delete(remote)
    assert not os.path.exists(FS._remote_path(remote).toString())  # skip file: prefix


def test_model_export(spark_model_iris, model_path, spark_custom_env):
    mlflow.spark.save_model(spark_model_iris.model, path=model_path, conda_env=spark_custom_env)
    # 1. score and compare reloaded sparkml model
    reloaded_model = mlflow.spark.load_model(model_uri=model_path)
    preds_df = reloaded_model.transform(spark_model_iris.spark_df)
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    assert spark_model_iris.predictions == preds1
    m = pyfunc.load_model(model_path)
    # 2. score and compare reloaded pyfunc
    preds2 = m.predict(spark_model_iris.pandas_df)
    assert spark_model_iris.predictions == preds2
    # 3. score and compare reloaded pyfunc Spark udf
    preds3 = score_model_as_udf(model_uri=model_path, pandas_df=spark_model_iris.pandas_df)
    assert spark_model_iris.predictions == preds3
    assert os.path.exists(MLFLOW_DFS_TMP.get())


def test_model_export_with_signature_and_examples(spark_model_iris, iris_signature):
    features_df = spark_model_iris.pandas_df.drop("label", axis=1)
    example_ = features_df.head(3)
    for signature in (None, iris_signature):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.spark.save_model(
                    spark_model_iris.model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                if example is None and signature is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == iris_signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_model_export_raise_when_example_is_spark_dataframe(spark, spark_model_iris, model_path):
    features_df = spark_model_iris.pandas_df.drop("label", axis=1)
    example = spark.createDataFrame(features_df.head(3))
    with pytest.raises(MlflowException, match="Examples can not be provided as Spark Dataframe."):
        mlflow.spark.save_model(spark_model_iris.model, path=model_path, input_example=example)


def test_log_model_with_signature_and_examples(spark_model_iris, iris_signature):
    features_df = spark_model_iris.pandas_df.drop("label", axis=1)
    example_ = features_df.head(3)
    artifact_path = "model"
    for signature in (None, iris_signature):
        for example in (None, example_):
            with mlflow.start_run():
                mlflow.spark.log_model(
                    spark_model_iris.model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=example,
                )
                artifact_uri = mlflow.get_artifact_uri()
                model_path = os.path.join(artifact_uri, artifact_path)
                mlflow_model = Model.load(model_path)
                if example is None and signature is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == iris_signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, model_path) == example).all())


def test_estimator_model_export(spark_model_estimator, model_path, spark_custom_env):
    mlflow.spark.save_model(
        spark_model_estimator.model, path=model_path, conda_env=spark_custom_env
    )
    # score and compare the reloaded sparkml model
    reloaded_model = mlflow.spark.load_model(model_uri=model_path)
    preds_df = reloaded_model.transform(spark_model_estimator.spark_df)
    preds = [x.prediction for x in preds_df.select("prediction").collect()]
    assert spark_model_estimator.predictions == preds
    # 2. score and compare reloaded pyfunc
    m = pyfunc.load_model(model_path)
    preds2 = m.predict(spark_model_estimator.spark_df.toPandas())
    assert spark_model_estimator.predictions == preds2


def test_transformer_model_export(spark_model_transformer, model_path, spark_custom_env):
    mlflow.spark.save_model(
        spark_model_transformer.model, path=model_path, conda_env=spark_custom_env
    )
    # score and compare the reloaded sparkml model
    reloaded_model = mlflow.spark.load_model(model_uri=model_path)
    preds_df = reloaded_model.transform(spark_model_transformer.spark_df)
    preds = [x.features for x in preds_df.select("features").collect()]
    assert spark_model_transformer.predictions == preds
    # 2. score and compare reloaded pyfunc
    m = pyfunc.load_model(model_path)
    preds2 = m.predict(spark_model_transformer.spark_df.toPandas())
    assert spark_model_transformer.predictions == preds2


def test_model_deployment(spark_model_iris, model_path, spark_custom_env):
    mlflow.spark.save_model(
        spark_model_iris.model,
        path=model_path,
        conda_env=spark_custom_env,
    )
    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=spark_model_iris.pandas_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        flavor=mlflow.pyfunc.FLAVOR_NAME,
    )
    from mlflow.deployments import PredictionsResponse

    np.testing.assert_array_almost_equal(
        spark_model_iris.predictions,
        PredictionsResponse.from_json(scoring_response.content).get_predictions(
            predictions_format="ndarray"
        ),
        decimal=4,
    )


@pytest.mark.skipif(
    "dev" in pyspark.__version__,
    reason="The dev version of pyspark built from the source doesn't exist on PyPI or Anaconda",
)
def test_sagemaker_docker_model_scoring_with_default_conda_env(spark_model_iris, model_path):
    mlflow.spark.save_model(
        spark_model_iris.model, path=model_path, extra_pip_requirements=["/opt/mlflow"]
    )

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=spark_model_iris.pandas_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        flavor=mlflow.pyfunc.FLAVOR_NAME,
    )
    deployed_model_preds = np.array(json.loads(scoring_response.content)["predictions"])

    np.testing.assert_array_almost_equal(
        deployed_model_preds, spark_model_iris.predictions, decimal=4
    )


@pytest.mark.parametrize("should_start_run", [False, True])
@pytest.mark.parametrize("use_dfs_tmpdir", [False, True])
def test_sparkml_model_log(tmp_path, spark_model_iris, should_start_run, use_dfs_tmpdir):
    old_tracking_uri = mlflow.get_tracking_uri()
    dfs_tmpdir = None if use_dfs_tmpdir else tmp_path.joinpath("test")

    try:
        tracking_dir = tmp_path.joinpath("mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        if should_start_run:
            mlflow.start_run()
        artifact_path = "model"
        mlflow.spark.log_model(
            artifact_path=artifact_path,
            spark_model=spark_model_iris.model,
            dfs_tmpdir=dfs_tmpdir,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

        reloaded_model = mlflow.spark.load_model(model_uri=model_uri, dfs_tmpdir=dfs_tmpdir)
        preds_df = reloaded_model.transform(spark_model_iris.spark_df)
        preds = [x.prediction for x in preds_df.select("prediction").collect()]
        assert spark_model_iris.predictions == preds
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(old_tracking_uri)


@pytest.mark.parametrize(
    ("registry_uri", "artifact_repo_class"),
    [
        ("databricks-uc", UnityCatalogModelsArtifactRepository),
        ("databricks", DatabricksModelsArtifactRepository),
    ],
)
def test_load_spark_model_from_models_uri(
    tmp_path, spark_model_estimator, registry_uri, artifact_repo_class
):
    model_dir = str(tmp_path.joinpath("spark_model"))
    model_name = "mycatalog.myschema.mymodel"
    fake_model_version = ModelVersion(name=model_name, version=str(3), creation_timestamp=0)

    with mock.patch(
        f"{MODELS_ARTIFACT_REPOSITORY}.get_underlying_uri"
    ) as mock_get_underlying_uri, mock.patch.object(
        artifact_repo_class, "download_artifacts", return_value=model_dir
    ) as mock_download_artifacts, mock.patch(
        "mlflow.get_registry_uri", return_value=registry_uri
    ), mock.patch.object(
        mlflow.tracking.MlflowClient, "get_model_version_by_alias", return_value=fake_model_version
    ) as get_model_version_by_alias_mock:
        mlflow.spark.save_model(
            path=model_dir,
            spark_model=spark_model_estimator.model,
        )
        mock_get_underlying_uri.return_value = "nonexistentscheme://fakeuri"
        mlflow.spark.load_model(f"models:/{model_name}/1")
        # Assert that we downloaded both the MLmodel file and the whole model itself using
        # the models:/ URI
        kwargs = (
            {"lineage_header_info": None}
            if artifact_repo_class is UnityCatalogModelsArtifactRepository
            else {}
        )
        assert mock_download_artifacts.mock_calls == [
            mock.call("MLmodel", None, **kwargs),
            mock.call("", None, **kwargs),
        ]
        mock_download_artifacts.reset_mock()
        mlflow.spark.load_model(f"models:/{model_name}@Champion")
        assert mock_download_artifacts.mock_calls == [
            mock.call("MLmodel", None, **kwargs),
            mock.call("", None, **kwargs),
        ]
        assert get_model_version_by_alias_mock.called_with(model_name, "Champion")


@pytest.mark.parametrize("should_start_run", [False, True])
@pytest.mark.parametrize("use_dfs_tmpdir", [False, True])
def test_sparkml_estimator_model_log(
    tmp_path, spark_model_estimator, should_start_run, use_dfs_tmpdir
):
    old_tracking_uri = mlflow.get_tracking_uri()
    dfs_tmpdir = None if use_dfs_tmpdir else tmp_path.joinpath("test")

    try:
        tracking_dir = tmp_path.joinpath("mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        if should_start_run:
            mlflow.start_run()
        artifact_path = "model"
        mlflow.spark.log_model(
            artifact_path=artifact_path,
            spark_model=spark_model_estimator.model,
            dfs_tmpdir=dfs_tmpdir,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

        reloaded_model = mlflow.spark.load_model(model_uri=model_uri, dfs_tmpdir=dfs_tmpdir)
        preds_df = reloaded_model.transform(spark_model_estimator.spark_df)
        preds = [x.prediction for x in preds_df.select("prediction").collect()]
        assert spark_model_estimator.predictions == preds
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(old_tracking_uri)


def test_log_model_calls_register_model(tmp_path, spark_model_iris):
    artifact_path = "model"
    dfs_tmp_dir = tmp_path.joinpath("test")
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.spark.log_model(
            artifact_path=artifact_path,
            spark_model=spark_model_iris.model,
            dfs_tmpdir=dfs_tmp_dir,
            registered_model_name="AdsModel1",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_uri,
            registered_model_name="AdsModel1",
        )


def test_log_model_no_registered_model_name(tmp_path, spark_model_iris):
    artifact_path = "model"
    dfs_tmp_dir = os.path.join(tmp_path, "test")
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.spark.log_model(
            artifact_path=artifact_path,
            spark_model=spark_model_iris.model,
            dfs_tmpdir=dfs_tmp_dir,
        )
        mlflow.tracking._model_registry.fluent._register_model.assert_not_called()


def test_sparkml_model_load_from_remote_uri_succeeds(spark_model_iris, model_path, mock_s3_bucket):
    mlflow.spark.save_model(spark_model=spark_model_iris.model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.spark.load_model(model_uri=model_uri)
    preds_df = reloaded_model.transform(spark_model_iris.spark_df)
    preds = [x.prediction for x in preds_df.select("prediction").collect()]
    assert spark_model_iris.predictions == preds


def test_sparkml_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    spark_model_iris, model_path, spark_custom_env
):
    mlflow.spark.save_model(
        spark_model=spark_model_iris.model, path=model_path, conda_env=spark_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != spark_custom_env

    with open(spark_custom_env) as f:
        spark_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == spark_custom_env_parsed


def test_sparkml_model_save_persists_requirements_in_mlflow_model_directory(
    spark_model_iris, model_path, spark_custom_env
):
    mlflow.spark.save_model(
        spark_model=spark_model_iris.model, path=model_path, conda_env=spark_custom_env
    )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(spark_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(spark_model_iris, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.spark.log_model(spark_model_iris.model, "model", pip_requirements=str(req_file))
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a"], strict=True
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model_iris.model, "model", pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model_iris.model, "model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(spark_model_iris, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.spark.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model_iris.model, "model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model_iris.model, "model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model_iris.model, "model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_sparkml_model_save_accepts_conda_env_as_dict(spark_model_iris, model_path):
    conda_env = dict(mlflow.spark.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.spark.save_model(
        spark_model=spark_model_iris.model, path=model_path, conda_env=conda_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_sparkml_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    spark_model_iris, model_path, spark_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.spark.log_model(
            spark_model=spark_model_iris.model,
            artifact_path=artifact_path,
            conda_env=spark_custom_env,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != spark_custom_env

    with open(spark_custom_env) as f:
        spark_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == spark_custom_env_parsed


def test_sparkml_model_log_persists_requirements_in_mlflow_model_directory(
    spark_model_iris, model_path, spark_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model=spark_model_iris.model,
            artifact_path=artifact_path,
            conda_env=spark_custom_env,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(spark_custom_env, saved_pip_req_path)


def test_sparkml_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    spark_model_iris, model_path
):
    mlflow.spark.save_model(spark_model=spark_model_iris.model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.spark.get_default_pip_requirements())


def test_sparkml_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    spark_model_iris,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.spark.log_model(spark_model=spark_model_iris.model, artifact_path=artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    _assert_pip_requirements(model_uri, mlflow.spark.get_default_pip_requirements())


def test_pyspark_version_is_logged_without_dev_suffix(spark_model_iris):
    expected_mlflow_version = _mlflow_major_version_string()
    unsuffixed_version = "2.4.0"
    for dev_suffix in [".dev0", ".dev", ".dev1", "dev.a", ".devb"]:
        with mock.patch("importlib_metadata.version", return_value=unsuffixed_version + dev_suffix):
            with mlflow.start_run():
                mlflow.spark.log_model(spark_model=spark_model_iris.model, artifact_path="model")
                model_uri = mlflow.get_artifact_uri("model")
            _assert_pip_requirements(
                model_uri, [expected_mlflow_version, f"pyspark=={unsuffixed_version}"]
            )

    for unaffected_version in ["2.0", "2.3.4", "2"]:
        with mock.patch("importlib_metadata.version", return_value=unaffected_version):
            pip_deps = _get_pip_deps(mlflow.spark.get_default_conda_env())
            assert any(x == f"pyspark=={unaffected_version}" for x in pip_deps)


def test_model_is_recorded_when_using_direct_save(spark_model_iris):
    # Patch `is_local_uri` to enforce direct model serialization to DFS
    with mock.patch("mlflow.spark.is_local_uri", return_value=False):
        with mlflow.start_run():
            mlflow.spark.log_model(spark_model=spark_model_iris.model, artifact_path="model")
            current_tags = mlflow.get_run(mlflow.active_run().info.run_id).data.tags
            assert mlflow.utils.mlflow_tags.MLFLOW_LOGGED_MODELS in current_tags


@pytest.mark.parametrize(
    (
        "artifact_uri",
        "db_runtime_version",
        "mlflowdbfs_disabled",
        "mlflowdbfs_available",
        "dbutils_available",
        "expected_uri",
    ),
    [
        (
            "dbfs:/databricks/mlflow-tracking/a/b",
            "12.0",
            "",
            True,
            True,
            "mlflowdbfs:///artifacts?run_id={}&path=/model/sparkml",
        ),
        (
            "dbfs:/databricks/mlflow-tracking/a/b",
            "12.0",
            "false",
            True,
            True,
            "mlflowdbfs:///artifacts?run_id={}&path=/model/sparkml",
        ),
        (
            "dbfs:/databricks/mlflow-tracking/a/b",
            "12.0",
            "false",
            True,
            False,
            "dbfs:/databricks/mlflow-tracking/a/b/model/sparkml",
        ),
        (
            "dbfs:/databricks/mlflow-tracking/a/b",
            "12.0",
            "",
            False,
            True,
            "dbfs:/databricks/mlflow-tracking/a/b/model/sparkml",
        ),
        (
            "dbfs:/databricks/mlflow-tracking/a/b",
            "",
            "",
            True,
            True,
            "dbfs:/databricks/mlflow-tracking/a/b/model/sparkml",
        ),
        (
            "dbfs:/databricks/mlflow-tracking/a/b",
            "12.0",
            "true",
            True,
            True,
            "dbfs:/databricks/mlflow-tracking/a/b/model/sparkml",
        ),
        ("dbfs:/root/a/b", "12.0", "", True, True, "dbfs:/root/a/b/model/sparkml"),
        ("s3://mybucket/a/b", "12.0", "", True, True, "s3://mybucket/a/b/model/sparkml"),
    ],
)
def test_model_logged_via_mlflowdbfs_when_appropriate(
    monkeypatch,
    spark_model_iris,
    artifact_uri,
    db_runtime_version,
    mlflowdbfs_disabled,
    mlflowdbfs_available,
    dbutils_available,
    expected_uri,
):
    def mock_spark_session_load(path):
        raise Exception("MlflowDbfsClient operation failed!")

    mock_spark_session = mock.Mock()
    mock_read_spark_session = mock.Mock()
    mock_read_spark_session.load = mock_spark_session_load

    from mlflow.utils.databricks_utils import _get_dbutils as og_getdbutils

    def mock_get_dbutils():
        # _get_dbutils is called during run creation and model logging; to avoid breaking run
        # creation, we only mock the output if _get_dbutils is called during spark model logging
        caller_fn_name = inspect.stack()[1].function
        if caller_fn_name == "_should_use_mlflowdbfs":
            if dbutils_available:
                return mock.Mock()
            else:
                raise Exception("dbutils not available")
        else:
            return og_getdbutils()

    with mock.patch(
        "mlflow.utils._spark_utils._get_active_spark_session", return_value=mock_spark_session
    ), mock.patch("mlflow.get_artifact_uri", return_value=artifact_uri), mock.patch(
        "mlflow.spark._HadoopFileSystem.is_filesystem_available", return_value=mlflowdbfs_available
    ), mock.patch(
        "mlflow.utils.databricks_utils.MlflowCredentialContext", autospec=True
    ), mock.patch(
        "mlflow.utils.databricks_utils._get_dbutils", mock_get_dbutils
    ), mock.patch.object(spark_model_iris.model, "save") as mock_save, mock.patch(
        "mlflow.models.infer_pip_requirements", return_value=[]
    ) as mock_infer:
        with mlflow.start_run():
            if db_runtime_version:
                monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", db_runtime_version)
            monkeypatch.setenv("DISABLE_MLFLOWDBFS", mlflowdbfs_disabled)
            mlflow.spark.log_model(spark_model=spark_model_iris.model, artifact_path="model")
            mock_save.assert_called_once_with(expected_uri.format(mlflow.active_run().info.run_id))

            if expected_uri.startswith("mflowdbfs"):
                # If mlflowdbfs is used, infer_pip_requirements should load the model from the
                # remote model path instead of a local tmp path.
                assert (
                    mock_infer.call_args[0][0]
                    == "dbfs:/databricks/mlflow-tracking/a/b/model/sparkml"
                )


@pytest.mark.parametrize("dummy_read_shows_mlflowdbfs_available", [True, False])
def test_model_logging_uses_mlflowdbfs_if_appropriate_when_hdfs_check_fails(
    monkeypatch, spark_model_iris, dummy_read_shows_mlflowdbfs_available
):
    def mock_spark_session_load(path):
        if dummy_read_shows_mlflowdbfs_available:
            raise Exception("MlflowdbfsClient operation failed!")
        else:
            raise Exception("mlflowdbfs filesystem not found")

    mock_read_spark_session = mock.Mock()
    mock_read_spark_session.load = mock_spark_session_load
    mock_spark_session = mock.Mock()
    mock_spark_session.read = mock_read_spark_session

    from mlflow.utils.databricks_utils import _get_dbutils as og_getdbutils

    def mock_get_dbutils():
        # _get_dbutils is called during run creation and model logging; to avoid breaking run
        # creation, we only mock the output if _get_dbutils is called during spark model logging
        caller_fn_name = inspect.stack()[1].function
        if caller_fn_name == "_should_use_mlflowdbfs":
            return mock.Mock()
        else:
            return og_getdbutils()

    with mock.patch(
        "mlflow.utils._spark_utils._get_active_spark_session",
        return_value=mock_spark_session,
    ), mock.patch(
        "mlflow.get_artifact_uri",
        return_value="dbfs:/databricks/mlflow-tracking/a/b",
    ), mock.patch(
        "mlflow.spark._HadoopFileSystem.is_filesystem_available",
        side_effect=Exception("MlflowDbfsClient operation failed!"),
    ), mock.patch(
        "mlflow.utils.databricks_utils.MlflowCredentialContext", autospec=True
    ), mock.patch(
        "mlflow.utils.databricks_utils._get_dbutils",
        mock_get_dbutils,
    ), mock.patch.object(spark_model_iris.model, "save") as mock_save:
        with mlflow.start_run():
            monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "12.0")
            mlflow.spark.log_model(spark_model=spark_model_iris.model, artifact_path="model")
            run_id = mlflow.active_run().info.run_id
            mock_save.assert_called_once_with(
                f"mlflowdbfs:///artifacts?run_id={run_id}&path=/model/sparkml"
                if dummy_read_shows_mlflowdbfs_available
                else "dbfs:/databricks/mlflow-tracking/a/b/model/sparkml"
            )


def test_log_model_with_code_paths(spark_model_iris):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.spark._add_code_from_conf_to_system_path",
        wraps=_add_code_from_conf_to_system_path,
    ) as add_mock:
        mlflow.spark.log_model(
            spark_model=spark_model_iris.model, artifact_path=artifact_path, code_paths=[__file__]
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.spark.FLAVOR_NAME)
        mlflow.spark.load_model(model_uri)
        add_mock.assert_called()


def test_virtualenv_subfield_points_to_correct_path(spark_model_iris, model_path):
    mlflow.spark.save_model(spark_model_iris.model, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


def test_model_save_load_with_metadata(spark_model_iris, model_path):
    mlflow.spark.save_model(
        spark_model_iris.model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(spark_model_iris):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model_iris.model,
            artifact_path=artifact_path,
            metadata={"metadata_key": "metadata_value"},
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


_df_input_example = iris_pandas_df().drop("label", axis=1).iloc[[0]]


@pytest.mark.parametrize(
    "input_example",
    # array and dict input examples are not supported any more as they
    # won't be converted to pandas dataframe when saving example
    [_df_input_example],
)
def test_model_log_with_signature_inference(spark_model_iris, input_example):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.spark.log_model(
            spark_model_iris.model, artifact_path=artifact_path, input_example=input_example
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    mlflow_model = Model.load(model_uri)
    input_columns = mlflow_model.signature.inputs.inputs
    assert all(col.type == DataType.double for col in input_columns)
    column_names = [col.name for col in input_columns]
    if isinstance(input_example, list):
        assert column_names == [0, 1, 2, 3]
    else:
        assert column_names == ["0", "1", "2", "3"]
    assert mlflow_model.signature.outputs == Schema([ColSpec(type=DataType.double)])


def test_log_model_with_vector_input_type_signature(spark, spark_model_estimator):
    from pyspark.ml.functions import vector_to_array

    from mlflow.types.schema import SparkMLVector

    model = spark_model_estimator.model
    with mlflow.start_run():
        model_info = mlflow.spark.log_model(
            model,
            "model",
            signature=ModelSignature(
                inputs=Schema(
                    [
                        ColSpec(name="features", type=SparkMLVector()),
                    ]
                ),
                outputs=Schema([ColSpec(type=DataType.double)]),
            ),
        )

    model_uri = model_info.model_uri
    model_meta = Model.load(model_uri)
    input_type = model_meta.signature.inputs.input_dict()["features"].type
    assert isinstance(input_type, SparkMLVector)

    pyfunc_model = pyfunc.load_model(model_uri)
    infer_data = spark_model_estimator.spark_df.withColumn(
        "features", vector_to_array("features")
    ).toPandas()
    preds = pyfunc_model.predict(infer_data)
    assert spark_model_estimator.predictions == preds
