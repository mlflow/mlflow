import openai
import mlflow
import pandas as pd
from pyspark.sql import SparkSession


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        messages=[{"role": "user", "content": "You are an MLflow expert!"}],
        artifact_path="model",
    )

with SparkSession.builder.getOrCreate() as spark:
    spark_udf = mlflow.pyfunc.spark_udf(
        spark=spark, model_uri=model_info.model_uri, result_type="string"
    )
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "role": ["user"] * 10,
                "content": [
                    "What is MLflow?",
                    "What are the key components of MLflow?",
                    "How does MLflow enable reproducibility?",
                    "What is MLflow tracking and how does it help?",
                    "How can you compare different ML models using MLflow?",
                    "How can you use MLflow to deploy ML models?",
                    "What are the integrations of MLflow with popular ML libraries?",
                    "How can you use MLflow to automate ML workflows?",
                    "What security and compliance features does MLflow offer?",
                    "Where does MLflow stand in the ML ecosystem?",
                ],
            }
        )
    )
    df.withColumn("answer", spark_udf("role", "content")).show()
