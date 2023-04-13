import openai
import mlflow
from pyspark.sql import SparkSession


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        messages=[{"role": "user", "content": "What is MLflow?"}],
        artifact_path="model",
    )

with SparkSession.builder.getOrCreate() as spark:
    spark_udf = mlflow.pyfunc.spark_udf(
        spark=spark, model_uri=model_info.model_uri, result_type="string"
    )
    df = spark.createDataFrame(
        [
            ("user", "What is MLflow?"),
            ("user", "What is Spark?"),
        ],
        ["role", "content"],
    )
    df = df.withColumn("answer", spark_udf("role", "content"))
    df.show()
