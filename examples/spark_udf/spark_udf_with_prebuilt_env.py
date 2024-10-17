"""
This example code shows how to use `mlflow.pyfunc.spark_udf` with Databricks Connect
outside Databricks runtime.
"""

import os
import tempfile

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
    model_info = mlflow.sklearn.log_model(model, "model")

model_uri = model_info.model_uri

# The prebuilt model environment archive file path.
# To build the model environment, run the following line code in Databricks runtime:
# `model_env_uc_path = mlflow.pyfunc.build_model_env(model_uri, "/Volumes/...")`
model_env_uc_path = "/Volumes/..."

tmp_dir = tempfile.mkdtemp()
local_model_env_path = os.path.join(tmp_dir, os.path.basename(model_env_uc_path))

# Download model env file from UC volume.
with ws.files.download(model_env_uc_path).contents as rf, open(local_model_env_path, "wb") as wf:
    while chunk := rf.read(4096):
        wf.write(chunk)

infer_spark_df = spark.createDataFrame(X)

# Setting 'prebuilt_env_path' parameter so that `spark_udf` can use the
# prebuilt python environment and skip rebuilding python environment.
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri, prebuilt_env_path=local_model_env_path)
result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

print(result)
