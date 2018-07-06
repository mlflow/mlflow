import os

import pandas as pd
import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.version import __version__ as pyspark_version
from sklearn import datasets

from mlflow import pyfunc
from mlflow import spark as sparkm
from mlflow.utils.environment import _mlflow_conda_env
from tests.helper_functions import score_model_in_sagemaker_docker_container


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
