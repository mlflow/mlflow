"""
python examples/databricks/dbconnect.py --cluster-id <cluster-id>
"""

import argparse

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from pyspark.sql.types import DoubleType
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

import mlflow
from mlflow.models import infer_signature


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wc = WorkspaceClient()

    # Train a model
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    model = KNeighborsClassifier().fit(X, y)
    predictions = model.predict(X)
    signature = infer_signature(X, predictions)

    # Log the model
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f"/Users/{wc.current_user.me().user_name}/dbconnect")
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, name="model", signature=signature)

    spark = DatabricksSession.builder.remote(
        host=wc.config.host,
        token=wc.config.token,
        cluster_id=args.cluster_id,
    ).getOrCreate()
    sdf = spark.createDataFrame(X.head(5))
    pyfunc_udf = mlflow.pyfunc.spark_udf(
        spark,
        model_info.model_uri,
        env_manager="local",
        result_type=DoubleType(),
    )
    preds = sdf.select(pyfunc_udf(*X.columns).alias("preds"))
    preds.show()


if __name__ == "__main__":
    main()
