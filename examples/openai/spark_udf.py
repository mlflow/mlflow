import os

import openai
from pyspark.sql import SparkSession

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-4o-mini",
        task=openai.chat.completions,
        messages=[{"role": "user", "content": "Tell me a {adjective} joke about {animal}."}],
        name="model",
    )

with SparkSession.builder.getOrCreate() as spark:
    spark_udf = mlflow.pyfunc.spark_udf(
        spark=spark, model_uri=model_info.model_uri, result_type="string"
    )
    df = spark.createDataFrame(
        [
            ("funny", "cats"),
            ("scary", "dogs"),
            ("sad", "rabbits"),
        ],
        ["adjective", "animal"],
    )
    df.withColumn("answer", spark_udf("adjective", "animal")).show()
