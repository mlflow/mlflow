from pyspark.sql import SparkSession
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

import mlflow
from mlflow.pyfunc import build_model_env

with SparkSession.builder.getOrCreate() as spark:
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    model = KNeighborsClassifier()
    model.fit(X, y)

    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, "model")

    model_uri = model_info.model_uri

    # build a python environment for `model_uri` and save it to `/tmp` directory.
    model_env_path = build_model_env(model_uri, "/tmp")

    infer_spark_df = spark.createDataFrame(X)

    # Setting 'prebuilt_env_path' parameter so that `spark_udf` can use the
    # prebuilt python environment and skip rebuilding python environment.
    pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri, prebuilt_env_path=model_env_path)
    result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

    print(result)
