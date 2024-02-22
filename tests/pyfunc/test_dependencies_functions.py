from pathlib import Path
from unittest import mock

import pytest
import sklearn
from sklearn.linear_model import LinearRegression

import mlflow.utils.requirements_utils
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import get_model_dependencies
from mlflow.utils import PYTHON_VERSION


def test_get_model_dependencies_read_req_file(tmp_path):
    req_file = tmp_path / "requirements.txt"
    req_file_content = """
mlflow
cloudpickle==2.0.0
scikit-learn==1.0.2"""
    req_file.write_text(req_file_content)

    model_path = str(tmp_path)

    # Test getting pip dependencies
    assert Path(get_model_dependencies(model_path, format="pip")).read_text() == req_file_content

    # Test getting pip dependencies will print instructions
    with mock.patch("mlflow.pyfunc._logger.info") as mock_log_info:
        get_model_dependencies(model_path, format="pip")
        mock_log_info.assert_called_once_with(
            "To install the dependencies that were used to train the model, run the "
            f"following command: 'pip install -r {req_file}'."
        )

        mock_log_info.reset_mock()
        with mock.patch("mlflow.pyfunc._is_in_ipython_notebook", return_value=True):
            get_model_dependencies(model_path, format="pip")
            mock_log_info.assert_called_once_with(
                "To install the dependencies that were used to train the model, run the "
                f"following command: '%pip install -r {req_file}'."
            )

    with pytest.raises(MlflowException, match="Illegal format argument 'abc'"):
        get_model_dependencies(model_path, format="abc")


@pytest.mark.parametrize(
    "ml_model_file_content",
    [
        """
artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: {PYTHON_VERSION}
model_uuid: 722a374a432f48f09ee85da92df13bca
run_id: 765e66a5ba404650be51cb02cda66f35
""",
        f"""
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: {PYTHON_VERSION}
model_uuid: 722a374a432f48f09ee85da92df13bca
run_id: 765e66a5ba404650be51cb02cda66f35
""",
    ],
    ids=["old_env", "new_env"],
)
def test_get_model_dependencies_read_conda_file(ml_model_file_content, tmp_path):
    MLmodel_file = tmp_path / "MLmodel"
    MLmodel_file.write_text(ml_model_file_content)
    conda_yml_file = tmp_path / "conda.yaml"
    conda_yml_file_content = f"""
channels:
- conda-forge
dependencies:
- python={PYTHON_VERSION}
- pip=22.0.3
- scikit-learn=0.22.0
- tensorflow=2.0.0
- pip:
  - mlflow
  - cloudpickle==2.0.0
  - scikit-learn==1.0.1
name: mlflow-env
"""

    conda_yml_file.write_text(conda_yml_file_content)

    model_path = str(tmp_path)

    # Test getting conda environment
    assert (
        Path(get_model_dependencies(model_path, format="conda")).read_text()
        == conda_yml_file_content
    )

    # Test getting pip requirement file failed and fallback to extract pip section from conda.yaml
    with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
        pip_file_path = get_model_dependencies(model_path, format="pip")
        assert (
            Path(pip_file_path).read_text().strip()
            == "mlflow\ncloudpickle==2.0.0\nscikit-learn==1.0.1"
        )
        mock_warning.assert_called_once_with(
            "The following conda dependencies have been excluded from the environment file: "
            f"python={PYTHON_VERSION}, pip=22.0.3, scikit-learn=0.22.0, tensorflow=2.0.0."
        )

    conda_yml_file.write_text(
        f"""
channels:
- conda-forge
dependencies:
- python={PYTHON_VERSION}
- pip=22.0.3
- scikit-learn=0.22.0
- tensorflow=2.0.0
"""
    )

    with pytest.raises(MlflowException, match="No pip section found in conda.yaml file"):
        get_model_dependencies(model_path, format="pip")


def test_get_model_dependencies_with_model_version_uri():
    with mlflow.start_run():
        mlflow.sklearn.log_model(LinearRegression(), "model", registered_model_name="linear")

    deps = get_model_dependencies("models:/linear/1", format="pip")
    assert f"scikit-learn=={sklearn.__version__}" in Path(deps).read_text()
