import os

import pandas as pd
import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.version import __version__ as pyspark_version
import pytest
from sklearn import datasets
import shutil
from collections import namedtuple

import mlflow
from mlflow import active_run, pyfunc, mleap
from mlflow import spark as sparkm
from mlflow.models import Model 

from mlflow.utils.environment import _mlflow_conda_env
from tests.helper_functions import score_model_in_sagemaker_docker_container

from tests.pyfunc.test_spark import score_model_as_udf


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
    FS.copy_from_local_file(test_dir_0, remote, removeSrc=False)
    local = os.path.join(str(tmpdir), "actual")
    FS.copy_to_local_file(remote, local, removeSrc=True)
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
    FS.copy_from_local_file(test_dir_0, remote, removeSrc=False)
    assert os.path.exists(FS._remote_path(remote).toString())  # skip file: prefix
    FS.delete(remote)
    assert not os.path.exists(FS._remote_path(remote).toString())  # skip file: prefix


@pytest.fixture
def spark_conda_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["pyspark=={}".format(pyspark_version)])
    return conda_env


SparkModelWithData = namedtuple("SparkModelWithData",
                                ["model", "training_df", "inference_df"])


@pytest.fixture
def spark_model_iris():
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    pandas_df = pd.DataFrame(X, columns=iris.feature_names)
    pandas_df['label'] = pd.Series(y)
    spark_session = pyspark.sql.SparkSession.builder \
        .config(key="spark_session.python.worker.reuse", value=True) \
        .master("local-cluster[2, 1, 1024]") \
        .config(key='spark.jars.packages',
                value='ml.combust.mleap:mleap-spark-base_2.11:0.10.0,'
                      'ml.combust.mleap:mleap-spark_2.11:0.10.0') \
        .getOrCreate()
    spark_df = spark_session.createDataFrame(pandas_df)
    assembler = VectorAssembler(inputCols=iris.feature_names, outputCol="features")
    lr = LogisticRegression(maxIter=50, regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[assembler, lr])
    # Fit the model
    model = pipeline.fit(spark_df)
    return SparkModelWithData(model=model, training_df=spark_df, inference_df=pandas_df)


@pytest.fixture
def model_path(tmpdir):
    return str(tmpdir.mkdir("model"))


@pytest.mark.large
def test_model_export(spark_model_iris, model_path, spark_conda_env):
    preds_df = spark_model_iris.model.transform(spark_model_iris.training_df)
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    sparkm.save_model(spark_model_iris.model, path=model_path,
                      conda_env=spark_conda_env)
    reloaded_model = sparkm.load_model(path=model_path)
    preds_df_1 = reloaded_model.transform(spark_model_iris.training_df)
    preds1_1 = [x.prediction for x in preds_df_1.select("prediction").collect()]
    assert preds1 == preds1_1
    m = pyfunc.load_pyfunc(model_path)
    preds2 = m.predict(spark_model_iris.inference_df)
    assert preds1 == preds2
    preds3 = score_model_in_sagemaker_docker_container(model_path=model_path,
                                                       data=spark_model_iris.inference_df)
    assert preds1 == preds3
    assert os.path.exists(sparkm.DFS_TMP)
    print(os.listdir(sparkm.DFS_TMP))
    assert not os.listdir(sparkm.DFS_TMP)


@pytest.mark.large
def test_model_log(tmpdir, spark_model_iris):
    # Print the coefficients and intercept for multinomial logistic regression
    preds_df = spark_model_iris.model.transform(spark_model_iris.training_df)
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    old_tracking_uri = tracking.get_tracking_uri()
    cnt = 0
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        for dfs_tmp_dir in [None, os.path.join(str(tmpdir), "test")]:
            print("should_start_run =", should_start_run, "dfs_tmp_dir =", dfs_tmp_dir)
            try:
                tracking_dir = os.path.abspath(str(tmpdir.mkdir("mlruns")))
                tracking.set_tracking_uri("file://%s" % tracking_dir)
                if should_start_run:
                    tracking.start_run()
                artifact_path = "model%d" % cnt
                cnt += 1
                if dfs_tmp_dir:
                    sparkm.log_model(artifact_path=artifact_path,
                                     spark_model=spark_model_iris.model,
                                     dfs_tmpdir=dfs_tmp_dir)
                else:
                    sparkm.log_model(artifact_path=artifact_path,
                                     spark_model=spark_model_iris.model)
                run_id = tracking.active_run().info.run_uuid
                # test pyfunc
                x = pyfunc.load_pyfunc(artifact_path, run_id=run_id)
                preds2 = x.predict(spark_model_iris.inference_df)
                assert preds1 == preds2
                # test load model
                reloaded_model = sparkm.load_model(artifact_path, run_id=run_id)
                preds_df_1 = reloaded_model.transform(spark_model_iris.training_df)
                preds3 = [x.prediction for x in preds_df_1.select("prediction").collect()]
                assert preds1 == preds3
                # test spark_udf
                preds4 = score_model_as_udf(artifact_path, run_id, spark_model_iris.inference_df)
                assert preds1 == preds4
                # make sure we did not leave any temp files behind
                x = dfs_tmp_dir or sparkm.DFS_TMP 
                assert os.path.exists(x)
                assert not os.listdir(x)
                shutil.rmtree(x)
            finally:
                tracking.end_run()
                tracking.set_tracking_uri(old_tracking_uri)
                shutil.rmtree(tracking_dir)


@pytest.mark.large
def test_model_log(tmpdir, spark_model_iris):
    # Print the coefficients and intercept for multinomial logistic regression
    preds_df = spark_model_iris.model.transform(spark_model_iris.training_df)
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    old_tracking_uri = tracking.get_tracking_uri()
    cnt = 0
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        for dfs_tmp_dir in [None, os.path.join(str(tmpdir), "test")]:
            print("should_start_run =", should_start_run, "dfs_tmp_dir =", dfs_tmp_dir)
            try:
                tracking_dir = os.path.abspath(str(tmpdir.mkdir("mlruns")))
                tracking.set_tracking_uri("file://%s" % tracking_dir)
                if should_start_run:
                    tracking.start_run()
                artifact_path = "model%d" % cnt
                cnt += 1
                if dfs_tmp_dir:
                    sparkm.log_model(artifact_path=artifact_path,
                                     spark_model=spark_model_iris.model,
                                     dfs_tmpdir=dfs_tmp_dir)
                else:
                    sparkm.log_model(artifact_path=artifact_path,
                                     spark_model=spark_model_iris.model)
                run_id = tracking.active_run().info.run_uuid
                # test pyfunc
                x = pyfunc.load_pyfunc(artifact_path, run_id=run_id)
                preds2 = x.predict(spark_model_iris.inference_df)
                assert preds1 == preds2
                # test load model
                reloaded_model = sparkm.load_model(artifact_path, run_id=run_id)
                preds_df_1 = reloaded_model.transform(spark_model_iris.training_df)
                preds3 = [x.prediction for x in preds_df_1.select("prediction").collect()]
                assert preds1 == preds3
                # test spark_udf
                preds4 = score_model_as_udf(artifact_path, run_id, spark_model_iris.inference_df)
                assert preds1 == preds4
                # make sure we did not leave any temp files behind
                x = dfs_tmp_dir or sparkm.DFS_TMP 
                assert os.path.exists(x)
                assert not os.listdir(x)
                shutil.rmtree(x)
            finally:
                tracking.end_run()
                tracking.set_tracking_uri(old_tracking_uri)
                shutil.rmtree(tracking_dir)


def test_model_save_without_sample_output_produces_sparkml_flavor(spark_model_iris, model_path):
    sparkm.save_model(spark_model=spark_model_iris.model,
                      path=model_path,
                      sample_input=None)
    config_path = os.path.join(model_path, "MLmodel")
    assert os.path.exists(config_path)
    config = Model.load(config_path)
    print(config.flavors)
    assert sparkm.FLAVOR_NAME in config.flavors


def test_model_save_with_sample_output_produces_sparkml_and_mleap_flavors(spark_model_iris,
        model_path):
    sparkm.save_model(spark_model=spark_model_iris.model,
                      path=model_path,
                      sample_input=spark_model_iris.training_df)
    config_path = os.path.join(model_path, "MLmodel")
    assert os.path.exists(config_path)
    config = Model.load(config_path)
    assert sparkm.FLAVOR_NAME in config.flavors
    assert mleap.FLAVOR_NAME in config.flavors
