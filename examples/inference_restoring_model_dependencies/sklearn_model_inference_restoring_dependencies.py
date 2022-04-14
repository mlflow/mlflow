import mlflow
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import tempfile
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

X, y = datasets.load_iris(as_frame=True, return_X_y=True)
model = KNeighborsClassifier()
model.fit(X, y)

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")

infer_spark_df = spark.createDataFrame(X)

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, f"runs:/{run.info.run_id}/model", env_manager="conda")
result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

print(result)

spark.stop()
