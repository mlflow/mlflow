import pandas as pd

from sklearn import datasets

import pyspark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import LogisticRegression

from mlflow import sparkml, pyfunc


def test_model_export(tmpdir):
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    pandas_df = pd.DataFrame(X, columns=iris.feature_names)
    pandas_df['label'] = pd.Series(y)
    spark = pyspark.sql.SparkSession.builder \
        .config(key="spark.python.worker.reuse", value=True) \
        .master("local-cluster[2, 1, 1024]") \
        .getOrCreate()
    spark_df = spark.createDataFrame(pandas_df)
    print("")
    print(dir(tmpdir))
    print(pandas_df)
    print(spark_df.show())
    model_path = tmpdir.mkdir("model")
    print("model_path", model_path)
    assembler = VectorAssembler(
        inputCols=iris.feature_names,
        outputCol="features")
    lr = LogisticRegression(maxIter=50, regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[assembler, lr])
    # Fit the model
    model = pipeline.fit(spark_df)
    # Print the coefficients and intercept for multinomial logistic regression
    preds_df = model.transform(spark_df)
    print(preds_df.show())
    preds1 = [x.prediction for x in preds_df.select("prediction").collect()]
    sparkml.save_model(model, path=model_path)
    m = pyfunc.load_pyfunc(model_path)
    preds2 = m.predict(pandas_df)
    print(pd.DataFrame({"preds1": preds1, "preds2": preds2}, columns=("preds1", "preds2")))
    assert preds1 == preds2
    dir(model)
