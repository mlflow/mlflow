"""
This example demonstrates how to specify pip requirements using `pip_requirements` and
`extra_pip_requirements` when logging a model via `mlflow.*.log_model`.
"""

import tempfile
from packaging.version import Version

import sklearn
from sklearn.datasets import load_iris
import xgboost as xgb
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.models.signature import infer_signature


def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


def get_pip_requirements(run_id, artifact_path, return_constraints=False):
    req_path = download_artifacts(run_id=run_id, artifact_path=f"{artifact_path}/requirements.txt")
    reqs = read_lines(req_path)

    if return_constraints:
        con_path = download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/constraints.txt"
        )
        cons = read_lines(con_path)
        return set(reqs), set(cons)

    return set(reqs)


def main():
    iris = load_iris()
    dtrain = xgb.DMatrix(iris.data, iris.target)
    model = xgb.train({}, dtrain)
    predictions = model.predict(dtrain)
    signature = infer_signature(dtrain.get_data(), predictions)

    xgb_req = f"xgboost=={xgb.__version__}"
    sklearn_req = f"scikit-learn=={sklearn.__version__}"

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Get the expected mlflow version
        mlflow_version_raw = Version(mlflow.__version__)
        mlflow_version = f"mlflow=={mlflow_version_raw.major}.{mlflow_version_raw.minor}"

        # Default (both `pip_requirements` and `extra_pip_requirements` are unspecified)
        artifact_path = "default"
        mlflow.xgboost.log_model(model, artifact_path, signature=signature)
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert pip_reqs.issuperset([mlflow_version, xgb_req]), pip_reqs

        # Overwrite the default set of pip requirements using `pip_requirements`
        artifact_path = "pip_requirements"
        mlflow.xgboost.log_model(
            model, artifact_path, pip_requirements=[sklearn_req], signature=signature
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert pip_reqs == {mlflow_version, sklearn_req}, pip_reqs

        # Add extra pip requirements on top of the default set of pip requirements
        # using `extra_pip_requirements`
        artifact_path = "extra_pip_requirements"
        mlflow.xgboost.log_model(
            model, artifact_path, extra_pip_requirements=[sklearn_req], signature=signature
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert pip_reqs.issuperset([mlflow_version, xgb_req, sklearn_req]), pip_reqs

        # Specify pip requirements using a requirements file
        with tempfile.NamedTemporaryFile("w", suffix=".requirements.txt") as f:
            f.write(sklearn_req)
            f.flush()

            # Path to a pip requirements file
            artifact_path = "requirements_file_path"
            mlflow.xgboost.log_model(
                model, artifact_path, pip_requirements=f.name, signature=signature
            )
            pip_reqs = get_pip_requirements(run_id, artifact_path)
            assert pip_reqs == {mlflow_version, sklearn_req}, pip_reqs

            # List of pip requirement strings
            artifact_path = "requirements_file_list"
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                pip_requirements=[xgb_req, f"-r {f.name}"],
                signature=signature,
            )
            pip_reqs = get_pip_requirements(run_id, artifact_path)
            assert pip_reqs == {mlflow_version, xgb_req, sklearn_req}, pip_reqs

        # Using a constraints file
        with tempfile.NamedTemporaryFile("w", suffix=".constraints.txt") as f:
            f.write(sklearn_req)
            f.flush()

            artifact_path = "constraints_file"
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                pip_requirements=[xgb_req, f"-c {f.name}"],
                signature=signature,
            )
            pip_reqs, pip_cons = get_pip_requirements(
                run_id, artifact_path, return_constraints=True
            )
            assert pip_reqs == {mlflow_version, xgb_req, "-c constraints.txt"}, pip_reqs
            assert pip_cons == {sklearn_req}, pip_cons


if __name__ == "__main__":
    main()
