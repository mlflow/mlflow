"""
Setup:
- Create a virtualenv with Python >= 3.9.
- Run `pip install databricks-connect databricks-sdk`.
- Set the following environment variables:
  ```
  export DATABRICKS_HOST=<workspace-URL>
  export DATABRICKS_TOKEN=<personal-access-token>
  ```
  or create `~/.databrickscfg`:
  ```
  [DEFAULT]
  host = <workspace-URL>
  token = <personal-access-token>
  ```

Usage:
```
# Create a new cluster (slow)
python examples/dbconnect.py

# Use an existing cluster (fast)
python examples/dbconnect.py --cluster-id <cluster-id>
```
"""
import argparse
import subprocess
import contextlib
import uuid
import re
import time
from typing import Generator, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import (
    DataSecurityMode,
    Library,
    PythonPyPiLibrary,
    ClusterDetails,
    LibraryFullStatusStatus,
)
from databricks.connect import DatabricksSession
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from pyspark.sql.types import DoubleType

import mlflow
from mlflow.models import infer_signature


def get_branch_name() -> Optional[str]:
    res = subprocess.run(
        ["git", "branch", "--show-current"],
        text=True,
        stdout=subprocess.PIPE,
    )
    if res.returncode != 0:
        return None
    return res.stdout


def get_github_username() -> Optional[str]:
    res = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        text=True,
        stdout=subprocess.PIPE,
    )
    if res.returncode != 0:
        return None
    m = re.search(r"([a-zA-Z0-9\-]+)/mlflow(:?\.git)?", res.stdout.strip())
    return m.group(1) if m else None


def get_mlflow_requirement() -> str:
    if (username := get_github_username()) and (branch := get_branch_name()):
        return f"git+https://github.com/{username}/mlflow.git@{branch}"
    return "mlflow"


@contextlib.contextmanager
def start_cluster(cluster_name: str, wc: WorkspaceClient) -> Generator[ClusterDetails, None, None]:
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
            libraries=[Library(pypi=PythonPyPiLibrary(get_mlflow_requirement()))],
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
            print(f"Waiting for mlflow to be installed...")
            time.sleep(10)
        else:
            raise RuntimeError("Failed to install mlflow")

        print("Cluster is ready")
        yield cluster.cluster_id
    finally:
        if cluster:
            print("Deleting cluster...")
            wc.clusters.delete_and_wait(cluster_id=cluster.cluster_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-id", default=None)
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

    ctx = (
        start_cluster(cluster_name=uuid.uuid4().hex, wc=wc)
        if args.cluster_id is None
        else contextlib.nullcontext(args.cluster_id)
    )
    with ctx as cluster_id:
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
