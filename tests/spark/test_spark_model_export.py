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

from mlflow import pyfunc
from mlflow import spark as sparkm
from mlflow import tracking

from mlflow.utils.environment import _mlflow_conda_env
from tests.helper_functions import score_model_in_sagemaker_docker_container


@pytest.mark.large
def test_model_export(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["pyspark=={}".format(pyspark_version)])
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    pandas_df = pd.DataFrame(X, columns=iris.feature_names)
    pandas_df['label'] = pd.Series(y)
    spark_session = pyspark.sql.SparkSession.builder \
        .config(key="spark_session.python.worker.reuse", value=True) \
        .master("local-cluster[2, 1, 1024]") \
        .getOrCreate()
    spark_df = spark_session.createDataFrame(pandas_df)
    model_path = tmpdir.mkdir("model")
    assembler = VectorAssembler(inputCols=iris.feature_names, outputCol="features")
    lr = LogisticRegression(maxIter=50, regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[assembler, lr])
    # Fit the model
    model = pipeline.fit(spark_df)
    # Print the coefficients and intercept for multinomial logistic regression
    preds_df = model.transform(spark_df)
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    sparkm.save_model(model, path=str(model_path), conda_env=conda_env)
    reloaded_model = sparkm.load_model(path=str(model_path))
    preds_df_1 = reloaded_model.transform(spark_df)
    preds1_1 = [x.prediction for x in preds_df_1.select("prediction").collect()]
    assert preds1 == preds1_1
    m = pyfunc.load_pyfunc(str(model_path))
    preds2 = m.predict(pandas_df)
    assert preds1 == preds2
    preds3 = score_model_in_sagemaker_docker_container(model_path=str(model_path), data=pandas_df)
    assert preds1 == preds3


@pytest.mark.large
def test_model_log(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["pyspark=={}".format(pyspark_version)])
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    pandas_df = pd.DataFrame(X, columns=iris.feature_names)
    pandas_df['label'] = pd.Series(y)
    spark_session = pyspark.sql.SparkSession.builder \
        .config(key="spark_session.python.worker.reuse", value=True) \
        .master("local-cluster[2, 1, 1024]") \
        .getOrCreate()
    spark_df = spark_session.createDataFrame(pandas_df)
    model_path = tmpdir.mkdir("model")
    assembler = VectorAssembler(inputCols=iris.feature_names, outputCol="features")
    lr = LogisticRegression(maxIter=50, regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[assembler, lr])
    # Fit the model
    model = pipeline.fit(spark_df)
    # Print the coefficients and intercept for multinomial logistic regression
    preds_df = model.transform(spark_df)
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    old_tracking_uri = tracking.get_tracking_uri()
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            tracking_dir = os.path.abspath(str(tmpdir.mkdir("mlruns")))
            tracking.set_tracking_uri("file://%s" % tracking_dir)
            if should_start_run:
                tracking.start_run()
            sparkm.log_model(artifact_path="model", spark_model=model)
            run_id = tracking.active_run().info.run_uuid
            x = pyfunc.load_pyfunc("model", run_id=run_id)
            preds2 = x.predict(pandas_df)
            assert preds1 == preds2
            reloaded_model = sparkm.load_model("model", run_id=run_id)
            preds_df_1 = reloaded_model.transform(spark_df)
            preds3 = [x.prediction for x in preds_df_1.select("prediction").collect()]
            assert preds1 == preds3
        finally:
            tracking.end_run()
            tracking.set_tracking_uri(old_tracking_uri)
            shutil.rmtree(tracking_dir)
