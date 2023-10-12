"""
python examples/databricks/dbconnect.py [--mlflow-req ]
"""
import argparse
import contextlib
import time
import uuid
from typing import Generator

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import (
    ClusterDetails,
    DataSecurityMode,
    Library,
    LibraryFullStatusStatus,
    PythonPyPiLibrary,
)
from pyspark.sql.types import DoubleType
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

import mlflow
from mlflow.models import infer_signature


@contextlib.contextmanager
def start_cluster(
    cluster_name: str, wc: WorkspaceClient, mlflow_req: str
) -> Generator[ClusterDetails, None, None]:
    cluster = None
    try:
        print("Creating new cluster (this may take a few minutes)...")
        cluster = wc.clusters.create_and_wait(
            cluster_name=cluster_name,
            spark_version="13.2.x-scala2.12",
            num_workers=1,
            node_type_id="i3.xlarge",
            autotermination_minutes=10,
            custom_tags={"PythonUDF.enabled": "true"},
        )
        print("Installing mlflow...")
        wc.libraries.install(
            cluster_id=cluster.cluster_id,
            libraries=[Library(pypi=PythonPyPiLibrary(mlflow_req))],
        )
        # clusters-create doesn't support data_security_mode, but clusters-edit does
        print("Updating access mode (this may take a few minutes)...")
        wc.clusters.edit_and_wait(
            cluster_name=cluster.cluster_name,
            cluster_id=cluster.cluster_id,
            spark_version=cluster.spark_version,
            node_type_id=cluster.node_type_id,
            num_workers=cluster.num_workers,
            autotermination_minutes=cluster.autotermination_minutes,
            custom_tags=cluster.custom_tags,
            data_security_mode=DataSecurityMode.SINGLE_USER,
            single_user_name=cluster.creator_user_name,
        )
        for _ in range(30):
            resp = wc.libraries.cluster_status(cluster_id=cluster.cluster_id)
            if any(ls.status == LibraryFullStatusStatus.INSTALLED for ls in resp.library_statuses):
                break
            print("Waiting for mlflow to be installed...")
            time.sleep(10)
        else:
            raise RuntimeError("Failed to install mlflow")

        print("Cluster is ready")
        yield cluster.cluster_id
    finally:
        if cluster:
            print("Deleting cluster...")
            wc.clusters.delete_and_wait(cluster_id=cluster.cluster_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-req", default="mlflow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Train a model
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    model = KNeighborsClassifier().fit(X, y)
    predictions = model.predict(X)
    signature = infer_signature(X, predictions)

    # Log the model
    mlflow.set_tracking_uri("databricks")
    wc = WorkspaceClient()
    mlflow.set_experiment(f"/Users/{wc.current_user.me().user_name}/dbconnect")
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

    # Make predictions
    with start_cluster(
        cluster_name=f"cluster-{uuid.uuid4().hex}", wc=wc, mlflow_req=args.mlflow_req
    ) as cluster_id:
        spark = DatabricksSession.builder.remote(
            host=wc.config.host,
            token=wc.config.token,
            cluster_id=cluster_id,
        ).getOrCreate()
        sdf = spark.createDataFrame(X.head(5))
        pyfunc_udf = mlflow.pyfunc.spark_udf(
            spark,
            model_info.model_uri,
            env_manager="local",
            result_type=DoubleType(),
        )
        result = sdf.select(pyfunc_udf(*X.columns).alias("preds"))
        result.show()


if __name__ == "__main__":
    main()
