from pyspark.ml.connect.classification import LogisticRegression
from pyspark.ml.connect.feature import StandardScaler
from pyspark.ml.connect.pipeline import Pipeline
from pyspark.sql import SparkSession
from sklearn import datasets

import mlflow

spark = SparkSession.builder.remote("local[2]").getOrCreate()

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
lr = LogisticRegression(maxIter=10, numTrainWorkers=2, learningRate=0.001)
pipeline = Pipeline(stages=[scaler, lr])

X, y = datasets.load_iris(return_X_y=True)

spark_df = spark.createDataFrame(zip(X, y), schema="features: array<double>, label: long")

pipeline_model = pipeline.fit(spark_df)

with mlflow.start_run():
    model_info = mlflow.spark.log_model(spark_model=pipeline_model, artifact_path="model")

model_uri = model_info.model_uri
print(f"Model is saved to URI: {model_uri}")

inference_df = spark_df.select("features").toPandas()
loaded_model = mlflow.spark.load_model(model_uri)
inference_result = loaded_model.transform(inference_df.copy(deep=False))

print("Loaded spark model inference result:\n")
print(inference_result)

pyfunc_model = mlflow.pyfunc.load_model(model_uri)
pyfunc_inference_result = pyfunc_model.predict(inference_df.copy(deep=False))
print("Loaded pyfunc model inference result:\n")
print(pyfunc_inference_result)
