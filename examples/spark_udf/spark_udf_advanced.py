import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


def print_with_title(title, *args):
    print(f"\n===== {title} =====\n")
    for a in args:
        print(a)


def extract_month(df):
    print_with_title("extract_month input", df.head(), df.dtypes)
    transformed = df.assign(month=df["timestamp"].dt.month)
    print_with_title("extract_month output", transformed.head(), transformed.dtypes)
    return transformed


def main():
    X, y = load_iris(as_frame=True, return_X_y=True)
    # Add a datetime column
    months = X.index.to_series() % 12 + 1
    timestamp = pd.to_datetime(months.map("2022-{:02d}-01 02:03:04".format))
    X = X.assign(timestamp=timestamp), y
    print_with_title("Ran input", X.head(30), X.dtypes)

    signature = mlflow.models.infer_signature(X, y)
    print_with_title("Signature", signature)

    month_extractor = FunctionTransformer(extract_month, validate=False)
    timestamp_remover = ColumnTransformer(
        [("selector", "passthrough", X.columns.drop("timestamp"))], remainder="drop"
    )
    model = Pipeline(
        [
            ("month_extractor", month_extractor),
            ("timestamp_remover", timestamp_remover),
            ("knn", KNeighborsClassifier()),
        ]
    )
    model.fit(X, y)

    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

    with SparkSession.builder.getOrCreate() as spark:
        infer_spark_df = spark.createDataFrame(X.sample(n=10, random_state=42))
        print_with_title(
            "Inference input",
            infer_spark_df._jdf.showString(5, 1, False),
            infer_spark_df._jdf.schema().treeString(),
        )

        pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, env_manager="conda")
        result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()
        print_with_title("Inference result", result)


if __name__ == "__main__":
    main()
