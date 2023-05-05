import mlflow
from mlflow.models.signature import infer_signature
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

X, y = datasets.load_iris(as_frame=True, return_X_y=True)
model = KNeighborsClassifier()
model.fit(X, y)
predictions = model.predict(X)
signature = infer_signature(X, predictions)

with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

infer_spark_df = spark.createDataFrame(X)

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, env_manager="conda")
result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

print(result)

spark.stop()
