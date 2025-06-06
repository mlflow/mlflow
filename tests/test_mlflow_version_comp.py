import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import sklearn
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression

import mlflow
from mlflow.models import Model


def check(run_id: str, tmp_path: Path) -> None:
    """
    Test various `runs:/<run_id>/model` URI consumers.
    """
    # Model loading
    model_uri = f"runs:/{run_id}/model"
    assert Model.load(model_uri).run_id == run_id
    model = mlflow.sklearn.load_model(model_uri)
    np.testing.assert_array_equal(model.predict([[1, 2]]), [3.0])
    model = mlflow.pyfunc.load_model(model_uri)
    np.testing.assert_array_equal(model.predict([[1, 2]]), [3.0])
    # Model registration
    mv = mlflow.register_model(model_uri, "model")
    mlflow.pyfunc.load_model(f"models:/{mv.name}/{mv.version}")
    # List artifacts
    client = mlflow.MlflowClient()
    assert len(client.list_artifacts(run_id=run_id, path="model")) == 6
    assert len(mlflow.artifacts.list_artifacts(artifact_uri=model_uri)) == 6
    assert len(mlflow.artifacts.list_artifacts(run_id=run_id, path="model")) == 6
    # Download artifacts
    out_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=tmp_path / str(uuid.uuid4())
    )
    assert next(Path(out_path).rglob("MLmodel")) is not None
    out_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model", dst_path=tmp_path / str(uuid.uuid4())
    )
    assert next(Path(out_path).rglob("MLmodel")) is not None
    out_path = client.download_artifacts(
        run_id=run_id, path="model", dst_path=tmp_path / str(uuid.uuid4())
    )
    assert next(Path(out_path).rglob("MLmodel")) is not None
    # Model evaluation
    eval_res = mlflow.models.evaluate(
        model=model_uri,
        data=np.array([[1, 2]]),
        targets=np.array([3]),
        model_type="regressor",
    )
    assert "mean_squared_error" in eval_res.metrics
    # Spark UDF
    if os.name != "nt":
        with SparkSession.builder.getOrCreate() as spark:
            udf = mlflow.pyfunc.spark_udf(
                spark,
                model_uri,
                result_type="double",
                env_manager="local",
            )
            df = spark.createDataFrame([[1, 2]], ["col1", "col2"])
            # This line fails with the following error on Windows:
            #   File ".../pyspark\python\lib\pyspark.zip\pyspark\serializers.py", line 472, in loads
            #     return cloudpickle.loads(obj, encoding=encoding)
            # ModuleNotFoundError: No module named 'pandas'
            pred = df.select(udf("col1", "col2").alias("pred")).collect()
            assert [row.pred for row in pred] == [3.0]


def test_mlflow_2_x_comp(tmp_path: Path) -> None:
    tracking_uri = (tmp_path / "tracking").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    exp_id = mlflow.create_experiment("test", artifact_location=artifact_location)
    mlflow.set_experiment(experiment_id=exp_id)

    out_file = tmp_path / "out.txt"
    # Log a model using MLflow 2.x
    py_ver = ".".join(map(str, sys.version_info[:2]))
    subprocess.check_call(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--index-strategy=unsafe-first-match",
            f"--python={py_ver}",
            # Use mlflow 2.x
            "--with=mlflow<3.0",
            # Pin numpy and sklearn versions to ensure the model can be loaded
            f"--with=numpy=={np.__version__}",
            f"--with=scikit-learn=={sklearn.__version__}",
            "python",
            # Use the isolated mode to ignore mlflow in the repository
            "-I",
            "-c",
            """
import sys
import mlflow
from sklearn.linear_model import LinearRegression

assert mlflow.__version__.startswith("2."), mlflow.__version__

fitted_model= LinearRegression().fit([[1, 2]], [3])
with mlflow.start_run() as run:
    model_info = mlflow.sklearn.log_model(fitted_model, artifact_path="model")
    assert model_info.model_uri.startswith("runs:/")
    out = sys.argv[1]
    with open(out, "w") as f:
        f.write(run.info.run_id)
""",
            out_file,
        ],
    )

    check(run_id=out_file.read_text().strip(), tmp_path=tmp_path)


def test_mlflow_3_x_comp(tmp_path: Path) -> None:
    tracking_uri = (tmp_path / "tracking").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    exp_id = mlflow.create_experiment("test", artifact_location=artifact_location)
    mlflow.set_experiment(experiment_id=exp_id)

    fitted_model = LinearRegression().fit([[1, 2]], [3])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(fitted_model, name="model")

    check(run_id=run.info.run_id, tmp_path=tmp_path)
