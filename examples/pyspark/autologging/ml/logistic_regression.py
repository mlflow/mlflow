from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris

import mlflow


spark = SparkSession.builder.getOrCreate()
X, y = load_iris(return_X_y=True)
df = spark.createDataFrame(
    [(Vectors.dense(features), int(label)) for features, label in zip(X, y)], ["features", "label"]
)
train, test = df.randomSplit([0.8, 0.2])

mlflow.pyspark.ml.autolog()
lor = LogisticRegression(maxIter=5)

with mlflow.start_run():
    lorModel = lor.fit(train)

pred = lorModel.transform(test)
pred.select(lorModel.getPredictionCol()).show(10)
