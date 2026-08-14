from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris

import mlflow

with SparkSession.builder.getOrCreate() as spark:
    df = load_iris(as_frame=True).frame.rename(columns={"target": "label"})
    df = spark.createDataFrame(df)
    df = VectorAssembler(inputCols=df.columns[:-1], outputCol="features").transform(df)
    train, test = df.randomSplit([0.8, 0.2])

    mlflow.pyspark.ml.autolog()
    lor = LogisticRegression(maxIter=5)
    ovr = OneVsRest(classifier=lor)

    with mlflow.start_run():
        ovrModel = ovr.fit(train)

    pred = ovrModel.transform(test)
    pred.select(ovrModel.getPredictionCol()).show(10)
