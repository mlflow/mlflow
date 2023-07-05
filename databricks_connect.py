"""
export DATABRICKS_HOST=...
export DATABRICKS_TOKEN=...
export DATABRICKS_CLUSTER_ID=...
"""
import os
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession
import mlflow
import mlflow
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType

assert "DATABRICKS_HOST" in os.environ
assert "DATABRICKS_TOKEN" in os.environ
assert "DATABRICKS_CLUSTER_ID" in os.environ
assert "DATABRICKS_EXPERIMENT_PATH" in os.environ

spark = DatabricksSession.builder.remote(
    host=os.environ["DATABRICKS_HOST"],
    token=os.environ["DATABRICKS_TOKEN"],
    cluster_id=os.environ["DATABRICKS_CLUSTER_ID"],
).getOrCreate()
spark.createDataFrame([(1,)], ["id"]).show()

X, y = datasets.load_iris(as_frame=True, return_X_y=True)
model = KNeighborsClassifier()
model.fit(X, y)
predictions = model.predict(X)
signature = infer_signature(X, predictions)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(os.environ["DATABRICKS_EXPERIMENT_PATH"])
with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

infer_spark_df = spark.createDataFrame(X)
infer_spark_df.limit(5).show()

pyfunc_udf = mlflow.pyfunc.spark_udf(
    spark, model_info.model_uri, env_manager="local", result_type=DoubleType()
)
result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

print(result)
