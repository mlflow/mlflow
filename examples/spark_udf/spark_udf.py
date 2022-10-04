import mlflow
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from pyspark import SparkConf
from pyspark.sql import SparkSession

conf = SparkConf()
# conf.set(key="spark.python.worker.reuse", value="true")
# conf.set(key="spark.task.maxFailures", value="1")
# conf.set(key="spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled", value="false")
# conf.set(key="spark.sql.pyspark.jvmStacktrace.enabled", value="true")
# conf.set(key="spark.driver.maxResultSize", value="3g")
# conf.set(key="spark.executor.memory", value="3g")

spark = SparkSession.builder.config(conf=conf).master("local[1]").getOrCreate()


X, y = datasets.load_iris(as_frame=True, return_X_y=True)
model = KNeighborsClassifier()
model.fit(X, y)

with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(model, "model")

infer_spark_df = spark.createDataFrame(X).head(5)

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, env_manager="local")
result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()

print(result)

spark.stop()
