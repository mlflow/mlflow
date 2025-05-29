import subprocess
import sys
import uuid
from pathlib import Path

import numpy
import sklearn
from sklearn.linear_model import LinearRegression

import mlflow


def test_mlflow_2_x_comp(tmp_path: Path) -> None:
    """
    Test various `runs:/` model URI consumers with a model logged in MLflow 2.x.
    """
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
            f"--with=numpy=={numpy.__version__}",
            f"--with=scikit-learn=={sklearn.__version__}",
            "python",
            "-I",  # Use the isolated mode to ignore mlflow in the repository
            "-c",
            """
import sys
import mlflow
from sklearn.linear_model import LinearRegression

assert mlflow.__version__.startswith("2."), mlflow.__version__

fitted_model= LinearRegression().fit([[1, 2]], [3])
with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(fitted_model, artifact_path="model")
    assert model_info.model_uri.startswith("runs:/")
    out = sys.argv[1]
    with open(out, "w") as f:
        f.write(model_info.model_uri)
""",
            out_file,
        ],
    )

    model_uri = out_file.read_text().strip()
    # Model loading
    mlflow.sklearn.load_model(model_uri)
    mlflow.pyfunc.load_model(model_uri)
    # Model registration
    mv = mlflow.register_model(model_uri, "model")
    mlflow.pyfunc.load_model(f"models:/{mv.name}/{mv.version}")
    # List artifacts
    run_id = model_uri.split("/")[-2]
    client = mlflow.MlflowClient()
    assert len(client.list_artifacts(run_id=run_id)) == 1
    assert len(client.list_artifacts(run_id=run_id, path="model")) == 6
    assert len(client.list_artifacts(run_id=run_id, path="model/MLmodel")) == 0
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
        data=numpy.array([[1, 2]]),
        targets=numpy.array([3]),
        model_type="regressor",
    )
    assert "mean_squared_error" in eval_res.metrics


def test_mlflow_3_x_comp(tmp_path: Path) -> None:
    """
    Does the same as `test_mlflow_2_x_comp`, but with a model logged in MLflow 3.x.
    """
    tracking_uri = (tmp_path / "tracking").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    exp_id = mlflow.create_experiment("test", artifact_location=artifact_location)
    mlflow.set_experiment(experiment_id=exp_id)

    fitted_model = LinearRegression().fit([[1, 2]], [3])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(fitted_model, name="model")

    # Model loading
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.sklearn.load_model(model_uri)
    mlflow.pyfunc.load_model(model_uri)
    # Model registration
    mv = mlflow.register_model(model_uri, "model")
    mlflow.pyfunc.load_model(f"models:/{mv.name}/{mv.version}")
    # List artifacts
    client = mlflow.MlflowClient()
    # TODO: Add support to list model artifacts
    # assert len(client.list_artifacts(run_id=run_id)) == 1
    # assert len(client.list_artifacts(run_id=run_id, path="model")) == 6
    # assert len(client.list_artifacts(run_id=run_id, path="model/MLmodel")) == 0
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
        data=numpy.array([[1, 2]]),
        targets=numpy.array([3]),
        model_type="regressor",
    )
    assert "mean_squared_error" in eval_res.metrics
