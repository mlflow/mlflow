import mlflow
import sklearn.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
import tempfile
from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()


X, y = datasets.load_iris(as_frame=True, return_X_y=True)
model = KNeighborsClassifier()
model.fit(X, y)

model_path = tempfile.mkdtemp()

mlflow.sklearn.save_model(model, model_path)

infer_spark_df = spark.createDataFrame(X)

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_path, env_manager="conda")
result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

print(result)

spark.stop()
