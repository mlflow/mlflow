import os

import json
import numpy as np
import pandas as pd
import pandas.testing
import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.wrapper import JavaModel
from pyspark.version import __version__ as pyspark_version
from pyspark.sql import SQLContext
from pyspark.sql.types import DateType
import pytest
from sklearn import datasets
import shutil
from collections import namedtuple
import yaml

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.tracking
from mlflow import active_run, pyfunc, mleap
from mlflow import spark as sparkm
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _get_model_log_dir
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.pyfunc.test_spark import score_model_as_udf


@pytest.fixture
def spark_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
            conda_env,
            additional_conda_deps=["pyspark", "pytest"])
    return conda_env


SparkModelWithData = namedtuple("SparkModelWithData",
                                ["model", "spark_df", "pandas_df", "predictions"])


# Specify `autouse=True` to ensure that a context is created
# before any tests are executed. This ensures that the Hadoop filesystem
# does not create its own SparkContext without the MLeap libraries required by
# other tests.
@pytest.fixture(scope="session", autouse=True)
def spark_context():
    conf = pyspark.SparkConf()
    conf.set(key="spark.jars.packages",
             value='ml.combust.mleap:mleap-spark-base_2.11:0.12.0,'
                   'ml.combust.mleap:mleap-spark_2.11:0.12.0')
    conf.set(key="spark_session.python.worker.reuse", value=True)
    spark = pyspark.sql.SparkSession.builder\
        .config(conf=conf)\
        .master("local-cluster[2, 1, 1024]")\
        .getOrCreate()
    sc = spark.sparkContext
    return sc


@pytest.fixture(scope="session")
def spark_model_iris(spark_context):
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    feature_names = ["0", "1", "2", "3"]
    pandas_df = pd.DataFrame(X, columns=feature_names)  # to make spark_udf work
    pandas_df['label'] = pd.Series(y)
    spark_session = pyspark.sql.SparkSession(spark_context)
    spark_df = spark_session.createDataFrame(pandas_df)
    assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
    lr = LogisticRegression(maxIter=50, regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[assembler, lr])
    # Fit the model
    model = pipeline.fit(spark_df)
    preds_df = model.transform(spark_df)
    preds = [x.prediction for x in preds_df.select("prediction").collect()]
    return SparkModelWithData(model=model,
                              spark_df=spark_df,
                              pandas_df=pandas_df,
                              predictions=preds)


@pytest.fixture
def model_path(tmpdir):
    return str(tmpdir.mkdir("model"))


@pytest.mark.large
def test_hadoop_filesystem(tmpdir):
    # copy local dir to and back from HadoopFS and make sure the results match
    from mlflow.spark import _HadoopFileSystem as FS
    test_dir_0 = os.path.join(str(tmpdir), "expected")
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
    local = os.path.join(str(tmpdir), "actual")
    FS.copy_to_local_file(remote, local, remove_src=True)
    assert sorted(os.listdir(os.path.join(local, "root"))) == sorted([
        "subdir", "file_0", ".file_0.crc"])
    assert sorted(os.listdir(os.path.join(local, "root", "subdir"))) == sorted([
        "file_1", ".file_1.crc"])
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


@pytest.mark.large
def test_model_export(spark_model_iris, model_path, spark_custom_env):
    sparkm.save_model(spark_model_iris.model, path=model_path,
                      conda_env=spark_custom_env)
    # 1. score and compare reloaded sparkml model
    reloaded_model = sparkm.load_model(path=model_path)
    preds_df = reloaded_model.transform(spark_model_iris.spark_df)
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    assert spark_model_iris.predictions == preds1
    m = pyfunc.load_pyfunc(model_path)
    # 2. score and compare reloaded pyfunc
    preds2 = m.predict(spark_model_iris.pandas_df)
    assert spark_model_iris.predictions == preds2
    # 3. score and compare reloaded pyfunc Spark udf
    preds3 = score_model_as_udf(model_path, run_id=None, pandas_df=spark_model_iris.pandas_df)
    assert spark_model_iris.predictions == preds3
    assert os.path.exists(sparkm.DFS_TMP)


# TODO(czumar): Remark this test as "large" instead of "release" after SageMaker docker
# container build issues have been debugged
# @pytest.mark.large
@pytest.mark.release
def test_model_deployment(spark_model_iris, model_path, spark_custom_env):
    sparkm.save_model(spark_model_iris.model, path=model_path,
                      conda_env=spark_custom_env,
                      # Test both spark ml and mleap
                      sample_input=spark_model_iris.spark_df)

    # 1. score and compare pyfunc deployed in Sagemaker docker container
    scoring_response_1 = score_model_in_sagemaker_docker_container(
            model_uri=model_path,
            data=spark_model_iris.pandas_df,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
            flavor=mlflow.pyfunc.FLAVOR_NAME)
    np.testing.assert_array_almost_equal(
            spark_model_iris.predictions,
            np.array(json.loads(scoring_response_1.content)),
            decimal=4)
    # 2. score and compare mleap deployed in Sagemaker docker container
    scoring_response_2 = score_model_in_sagemaker_docker_container(
            model_uri=model_path,
            data=spark_model_iris.pandas_df.to_json(orient="split"),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            flavor=mlflow.mleap.FLAVOR_NAME)
    np.testing.assert_array_almost_equal(
            spark_model_iris.predictions,
            np.array(json.loads(scoring_response_2.content)),
            decimal=4)


@pytest.mark.release
def test_sagemaker_docker_model_scoring_with_default_conda_env(spark_model_iris, model_path):
    sparkm.save_model(spark_model_iris.model, path=model_path, conda_env=None)

    scoring_response = score_model_in_sagemaker_docker_container(
            model_path=model_path,
            data=spark_model_iris.pandas_df,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            flavor=mlflow.pyfunc.FLAVOR_NAME)
    deployed_model_preds = json.loads(scoring_response.content)

    np.testing.assert_array_almost_equal(
            deployed_model_preds,
            spark_model_iris.predictions,
            decimal=4)


@pytest.mark.large
def test_sparkml_model_log(tmpdir, spark_model_iris):
    # Print the coefficients and intercept for multinomial logistic regression
    old_tracking_uri = mlflow.get_tracking_uri()
    cnt = 0
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        for dfs_tmp_dir in [None, os.path.join(str(tmpdir), "test")]:
            print("should_start_run =", should_start_run, "dfs_tmp_dir =", dfs_tmp_dir)
            try:
                tracking_dir = os.path.abspath(str(tmpdir.join("mlruns")))
                mlflow.set_tracking_uri("file://%s" % tracking_dir)
                if should_start_run:
                    mlflow.start_run()
                artifact_path = "model%d" % cnt
                cnt += 1
                sparkm.log_model(artifact_path=artifact_path, spark_model=spark_model_iris.model,
                                 dfs_tmpdir=dfs_tmp_dir)
                run_id = active_run().info.run_id
                # test reloaded model
                reloaded_model = sparkm.load_model(artifact_path, run_id=run_id,
                                                   dfs_tmpdir=dfs_tmp_dir)
                preds_df = reloaded_model.transform(spark_model_iris.spark_df)
                preds = [x.prediction for x in preds_df.select("prediction").collect()]
                assert spark_model_iris.predictions == preds
            finally:
                mlflow.end_run()
                mlflow.set_tracking_uri(old_tracking_uri)
                x = dfs_tmp_dir or sparkm.DFS_TMP
                shutil.rmtree(x)
                shutil.rmtree(tracking_dir)


@pytest.mark.large
def test_sparkml_model_save_persists_specified_conda_env_in_mlflow_model_directory(
        spark_model_iris, model_path, spark_custom_env):
    sparkm.save_model(spark_model=spark_model_iris.model,
                      path=model_path,
                      conda_env=spark_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != spark_custom_env

    with open(spark_custom_env, "r") as f:
        spark_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == spark_custom_env_parsed


@pytest.mark.large
def test_sparkml_model_log_invalid_args(spark_model_iris, model_path):
    with pytest.raises(MlflowException) as e:
        sparkm.log_model(
            spark_model=spark_model_iris.model.stages[0],
            artifact_path="model0")
        assert e.message.contains("SparkML can only save PipelineModels")
    with pytest.raises(MlflowException) as e:
        sparkm.log_model(
            spark_model=spark_model_iris.model,
            artifact_path="model1",
            jars=["something.jar"])
        assert e.message.contains("JAR dependencies are not implemented")


@pytest.mark.large
def test_sparkml_model_save_accepts_conda_env_as_dict(spark_model_iris, model_path):
    conda_env = dict(mlflow.spark.DEFAULT_CONDA_ENV)
    conda_env["dependencies"].append("pytest")
    sparkm.save_model(spark_model=spark_model_iris.model,
                      path=model_path,
                      conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_sparkml_model_log_persists_specified_conda_env_in_mlflow_model_directory(
        spark_model_iris, model_path, spark_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        sparkm.log_model(
                spark_model=spark_model_iris.model,
                artifact_path=artifact_path,
                conda_env=spark_custom_env)
        run_id = mlflow.active_run().info.run_id
    model_path = _get_model_log_dir(artifact_path, run_id)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != spark_custom_env

    with open(spark_custom_env, "r") as f:
        spark_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == spark_custom_env_parsed


@pytest.mark.large
def test_sparkml_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        spark_model_iris, model_path):
    sparkm.save_model(spark_model=spark_model_iris.model, path=model_path, conda_env=None)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == sparkm.DEFAULT_CONDA_ENV


@pytest.mark.large
def test_sparkml_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        spark_model_iris):
    artifact_path = "model"
    with mlflow.start_run():
        sparkm.log_model(
                spark_model=spark_model_iris.model, artifact_path=artifact_path, conda_env=None)
        run_id = mlflow.active_run().info.run_id
    model_path = _get_model_log_dir(artifact_path, run_id)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == sparkm.DEFAULT_CONDA_ENV


@pytest.mark.large
def test_mleap_model_log(spark_model_iris):
    artifact_path = "model"
    with mlflow.start_run():
        rid = active_run().info.run_id
        sparkm.log_model(spark_model=spark_model_iris.model,
                         sample_input=spark_model_iris.spark_df,
                         artifact_path=artifact_path)
    model_path = _get_model_log_dir(model_name=artifact_path, run_id=rid)
    config_path = os.path.join(model_path, "MLmodel")
    mlflow_model = Model.load(config_path)
    assert sparkm.FLAVOR_NAME in mlflow_model.flavors
    assert mleap.FLAVOR_NAME in mlflow_model.flavors


@pytest.mark.large
def test_mleap_output_json_format(spark_model_iris, model_path):
    mlflow_model = Model()
    mleap.save_model(spark_model=spark_model_iris.model,
                     path=model_path,
                     sample_input=spark_model_iris.spark_df,
                     mlflow_model=mlflow_model)
    mleap_conf = mlflow_model.flavors[mleap.FLAVOR_NAME]
    schema_path_sub = mleap_conf["input_schema"]
    schema_path_full = os.path.join(model_path, schema_path_sub)
    with open(schema_path_full, "r") as f:
        json_schema = json.load(f)

    assert "fields" in json_schema.keys()
    assert len(json_schema["fields"]) > 0
    assert type(json_schema["fields"][0]) == dict
    assert "name" in json_schema["fields"][0]


@pytest.mark.large
def test_spark_module_model_save_with_mleap_and_unsupported_transformer_raises_exception(
        spark_model_iris, model_path):
    class CustomTransformer(JavaModel):
        def _transform(self, dataset):
            return dataset

    unsupported_pipeline = Pipeline(stages=[CustomTransformer()])
    unsupported_model = unsupported_pipeline.fit(spark_model_iris.spark_df)

    with pytest.raises(ValueError):
        sparkm.save_model(spark_model=unsupported_model,
                          path=model_path,
                          sample_input=spark_model_iris.spark_df)


@pytest.mark.large
def test_spark_module_model_save_with_relative_path_and_valid_sample_input_produces_mleap_flavor(
        spark_model_iris):
    with TempDir(chdr=True) as tmp:
        model_path = os.path.basename(tmp.path("model"))
        mlflow_model = Model()
        sparkm.save_model(spark_model=spark_model_iris.model,
                          path=model_path,
                          sample_input=spark_model_iris.spark_df,
                          mlflow_model=mlflow_model)
        assert mleap.FLAVOR_NAME in mlflow_model.flavors

        config_path = os.path.join(model_path, "MLmodel")
        assert os.path.exists(config_path)
        config = Model.load(config_path)
        assert mleap.FLAVOR_NAME in config.flavors


@pytest.mark.large
def test_mleap_module_model_save_with_relative_path_and_valid_sample_input_produces_mleap_flavor(
        spark_model_iris):
    with TempDir(chdr=True) as tmp:
        model_path = os.path.basename(tmp.path("model"))
        mlflow_model = Model()
        mleap.save_model(spark_model=spark_model_iris.model,
                         path=model_path,
                         sample_input=spark_model_iris.spark_df,
                         mlflow_model=mlflow_model)
        assert mleap.FLAVOR_NAME in mlflow_model.flavors

        config_path = os.path.join(model_path, "MLmodel")
        assert os.path.exists(config_path)
        config = Model.load(config_path)
        assert mleap.FLAVOR_NAME in config.flavors


@pytest.mark.large
def test_mleap_module_model_save_with_absolute_path_and_valid_sample_input_produces_mleap_flavor(
        spark_model_iris, model_path):
    model_path = os.path.abspath(model_path)
    mlflow_model = Model()
    mleap.save_model(spark_model=spark_model_iris.model,
                     path=model_path,
                     sample_input=spark_model_iris.spark_df,
                     mlflow_model=mlflow_model)
    assert mleap.FLAVOR_NAME in mlflow_model.flavors

    config_path = os.path.join(model_path, "MLmodel")
    assert os.path.exists(config_path)
    config = Model.load(config_path)
    assert mleap.FLAVOR_NAME in config.flavors


@pytest.mark.large
def test_mleap_module_model_save_with_invalid_sample_input_type_raises_exception(
        spark_model_iris, model_path):
    with pytest.raises(Exception):
        invalid_input = pd.DataFrame()
        sparkm.save_model(spark_model=spark_model_iris.model,
                          path=model_path,
                          sample_input=invalid_input)


@pytest.mark.large
def test_mleap_module_model_save_with_unsupported_transformer_raises_serialization_exception(
        spark_model_iris, model_path):
    class CustomTransformer(JavaModel):
        def _transform(self, dataset):
            return dataset

    unsupported_pipeline = Pipeline(stages=[CustomTransformer()])
    unsupported_model = unsupported_pipeline.fit(spark_model_iris.spark_df)

    with pytest.raises(mleap.MLeapSerializationException):
        mleap.save_model(spark_model=unsupported_model,
                         path=model_path,
                         sample_input=spark_model_iris.spark_df)


@pytest.mark.large
def test_save_with_sample_input_containing_unsupported_data_type_raises_serialization_exception(
        spark_context, model_path):
    sql_context = SQLContext(spark_context)
    unsupported_df = sql_context.createDataFrame([(1, "2016-09-30"), (2, "2017-02-27")])
    unsupported_df = unsupported_df.withColumn("_2", unsupported_df._2.cast(DateType()))
    pipeline = Pipeline(stages=[])
    model = pipeline.fit(unsupported_df)
    # The Spark `DateType` is not supported by MLeap, so we expect serialization to fail.
    with pytest.raises(mleap.MLeapSerializationException):
        sparkm.save_model(spark_model=model, path=model_path, sample_input=unsupported_df)
