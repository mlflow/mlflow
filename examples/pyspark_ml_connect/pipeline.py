from pyspark.ml.connect.classification import (
    LogisticRegression as LORV2,
)
from pyspark.ml.connect.feature import StandardScaler
from pyspark.ml.connect.pipeline import Pipeline
from pyspark.sql import SparkSession
from sklearn import datasets

import mlflow

spark = SparkSession.builder.remote("local[2]").getOrCreate()

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
lr = LORV2(maxIter=50, numTrainWorkers=2, learningRate=0.001)
pipeline = Pipeline(stages=[scaler, lr])

X, y = datasets.load_iris(return_X_y=True)

spark_df = spark.createDataFrame(list(zip(X, y)), schema="features: array<double>, label: long")

pandas_df = spark_df.toPandas()
pipeline_model = pipeline.fit(spark_df)

model_info = mlflow.spark.log_model(spark_model=pipeline_model, artifact_path="model")
print(f"Model is saved to URI: {model_info.model_uri}")
