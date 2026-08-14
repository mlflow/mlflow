from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris

import mlflow

with SparkSession.builder.getOrCreate() as spark:
    mlflow.pyspark.ml.autolog()

    df = load_iris(as_frame=True).frame.rename(columns={"target": "label"})
    df = spark.createDataFrame(df)
    train, test = df.randomSplit([0.8, 0.2])

    assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
    scaler = StandardScaler(inputCol=assembler.getOutputCol(), outputCol="scaledFeatures")
    lor = LogisticRegression(maxIter=5, featuresCol=scaler.getOutputCol())

    # Non-neseted pipeline
    pipeline = Pipeline(stages=[assembler, scaler, lor])
    with mlflow.start_run():
        pipeline_model = pipeline.fit(train)

    columns = ["features", "prediction"]
    pipeline_model.transform(test).select(columns).show()

    # Nested pipeline
    nested_pipeline = Pipeline(stages=[Pipeline(stages=[assembler, scaler]), lor])
    with mlflow.start_run():
        nested_pipeline_model = nested_pipeline.fit(train)

    nested_pipeline_model.transform(test).select(columns).show()
