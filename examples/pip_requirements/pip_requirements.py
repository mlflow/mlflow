"""
This example demonstrates how to specify pip requirements using `pip_requirements` and
`extra_pip_requirements` when logging a model via `mlflow.*.log_model`.
"""

import tempfile

import sklearn
from sklearn.datasets import load_iris
import xgboost as xgb
import mlflow


def get_pip_requirements(run_id, artifact_path):
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, f"{artifact_path}/requirements.txt")
    with open(local_path) as f:
        return f.read().splitlines()


def main():
    iris = load_iris()
    dtrain = xgb.DMatrix(iris.data, iris.target)
    model = xgb.train({}, dtrain)

    xgb_req = f"xgboost=={xgb.__version__}"
    sklearn_req = f"scikit-learn=={sklearn.__version__}"

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Default (both `pip_requirements` and `extra_pip_requirements` are unspecified)
        artifact_path = "default"
        mlflow.xgboost.log_model(model, artifact_path)
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert pip_reqs == ["mlflow", xgb_req], pip_reqs

        # Overwrite the default set of pip requirements using `pip_requirements`
        artifact_path = "pip_requirements"
        mlflow.xgboost.log_model(model, artifact_path, pip_requirements=[sklearn_req])
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert pip_reqs == ["mlflow", sklearn_req], pip_reqs

        # Add extra pip requirements on top of the default set of pip requirements
        # using `extra_pip_requirements`
        artifact_path = "extra_pip_requirements"
        mlflow.xgboost.log_model(model, artifact_path, extra_pip_requirements=[sklearn_req])
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert pip_reqs == ["mlflow", xgb_req, sklearn_req], pip_reqs

        # Specify pip requirements using a requirements file
        with tempfile.NamedTemporaryFile("w", suffix=".requirements.txt") as f:
            f.write(sklearn_req)
            f.flush()

            # Path to a pip requirements file
            artifact_path = "requirements_file_path"
            mlflow.xgboost.log_model(model, artifact_path, pip_requirements=f.name)
            pip_reqs = get_pip_requirements(run_id, artifact_path)
            assert pip_reqs == ["mlflow", sklearn_req], pip_reqs

            # List of pip requirement strings
            artifact_path = "requirements_file_list"
            mlflow.xgboost.log_model(
                model, artifact_path, pip_requirements=[xgb_req, f"-r {f.name}"]
            )
            pip_reqs = get_pip_requirements(run_id, artifact_path)
            assert pip_reqs == ["mlflow", xgb_req, sklearn_req], pip_reqs


if __name__ == "__main__":
    main()
