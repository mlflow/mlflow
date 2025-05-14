"""
This example code shows how to use `mlflow.pyfunc.spark_udf` with Databricks Connect
outside Databricks runtime.
"""

import os

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

import mlflow

ws = WorkspaceClient()

spark = DatabricksSession.builder.remote(
    host=os.environ["DATABRICKS_HOST"],
    token=os.environ["DATABRICKS_TOKEN"],
    cluster_id="<cluster id>",  # get cluster id by spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
).getOrCreate()

X, y = datasets.load_iris(as_frame=True, return_X_y=True)
model = KNeighborsClassifier()
model.fit(X, y)

with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(model, name="model")

model_uri = model_info.model_uri

# The prebuilt model environment archive file path.
# To build the model environment, run the following line code in Databricks runtime:
# `model_env_uc_path = mlflow.pyfunc.build_model_env(model_uri, "/Volumes/...")`
model_env_uc_path = "dbfs:/Volumes/..."

infer_spark_df = spark.createDataFrame(X)

# Setting 'prebuilt_env_uri' parameter so that `spark_udf` can use the
# prebuilt python environment and skip rebuilding python environment.
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri, prebuilt_env_uri=model_env_uc_path)
result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

print(result)
