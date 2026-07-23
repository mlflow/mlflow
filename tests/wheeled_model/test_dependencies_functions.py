import os
import yaml
import random
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models import Model
from mlflow.utils.environment import _is_pip_deps
from mlflow.wheeled_model import (
    _download_wheels,
    _overwrite_pip_requirements_with_wheels,
    _update_model_file,
    _update_conda_file_with_wheels,
)
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
)

_ARTIFACTS_FOLDER_NAME = "artifacts"
_WHEELS_FOLDER_NAME = "wheels"


def _get_list_from_file(path):
    with open(path, "r") as file:
        return file.read().splitlines()


def _get_pip_requirements_list(path):
    return _get_list_from_file(path)


def random_int(lo=1, hi=1000000000):
    return random.randint(lo, hi)


def random_file(ext="txt", dest_dir="."):
    return os.path.join(dest_dir, "temp_test_%d.%s" % (random_int(), ext))


def create_random_pip_requirements_file(model_path):
    pip_requirements_path = os.path.join(model_path, _REQUIREMENTS_FILE_NAME)
    with open(pip_requirements_path, "w") as f:
        f.write("mlflow\ncloudpickle==1.6.0\nxgboost==1.4.2\n")

    return pip_requirements_path


def create_wheels_directory(model_path):
    wheels_dir = os.path.join(model_path, _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME)
    os.makedirs(wheels_dir)
    return wheels_dir


def download_wheels(wheels_dir, pip_requirements_path):
    _download_wheels(wheels_dir, pip_requirements_path)


def create_random_conda_file(model_path):
    conda_yaml_path = os.path.join(model_path, _CONDA_ENV_FILE_NAME)

    with open(conda_yaml_path, "w") as f:
        f.write(
            """
channels:
- conda-forge
dependencies:
- python=3.8.10
- pip
- pip:
  - mlflow
  - xgboost==1.4.2
  - cloudpickle==1.6.0
name: xgb_env"""
        )

    return conda_yaml_path


def create_random_model_file(model_path):
    model_file_path = os.path.join(model_path, MLMODEL_FILE_NAME)

    with open(model_file_path, "w") as f:
        f.write(
            """
artifact_path: model
databricks_runtime: 10.0.x-snapshot-cpu-ml-scala2.12
flavors:
  python_function:
    artifacts:
      xgb_model:
        path: artifacts/xgb_model.pth
        uri: xgb_model.pth
    cloudpickle_version: 1.6.0
    env: conda.yaml
    loader_module: mlflow.pyfunc.model
    python_model: python_model.pkl
    python_version: 3.8.10
run_id: 8b1f2cbbf88a4fca9742e29b494e0418
utc_time_created: '2021-10-24 07:58:05.451529'"""
        )

    return model_file_path


def create_random_model_env(tmp_path, build_wheels=False):
    """
    Creates a demo model env with the following structure
      ├── MLmodel
      ├── conda.yaml
      ├── artifacts    # Optional
      │  ├── wheels
      └── requirements.txt
    """

    # Create a model directory of the format model_<random int>
    while True:
        model_dir = f"model_{random_int()}"
        model_path = os.path.join(tmp_path, model_dir)

        # If the directory doesn't exist create it and break out
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            break

    model_file_path = create_random_model_file(model_path)
    conda_yaml_path = create_random_conda_file(model_path)
    pip_requirements_path = create_random_pip_requirements_file(model_path)
    wheels_dir = None

    if build_wheels:
        wheels_dir = create_wheels_directory(model_path)
        _download_wheels(wheels_dir, pip_requirements_path)

    return model_path, model_file_path, conda_yaml_path, pip_requirements_path, wheels_dir


def test_overwrite_pip_requirements_with_wheels(tmp_path):
    def compare_wheels_dir_and_pip_requirements():
        # Get list of wheels from the directory
        wheels_list = []
        for wheel_file in os.listdir(wheels_dir):
            if wheel_file.endswith(".whl"):
                relative_wheel_path = os.path.join(
                    _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME, wheel_file
                )
                wheels_list.append(relative_wheel_path)

        pip_requirements_contents = _get_pip_requirements_list(pip_requirements_path)

        # Sort both lists alphabetically
        wheels_list.sort()
        pip_requirements_contents.sort()

        assert wheels_list == pip_requirements_contents

    # Test case: `wheels_dir` contains the required wheels (Default case)
    _, _, _, pip_requirements_path, wheels_dir = create_random_model_env(
        tmp_path, build_wheels=True
    )
    _overwrite_pip_requirements_with_wheels(wheels_dir, pip_requirements_path)
    compare_wheels_dir_and_pip_requirements()

    # Test case: `wheels_dir` contains files there are not wheels.
    _, _, _, pip_requirements_path, wheels_dir = create_random_model_env(
        tmp_path, build_wheels=True
    )

    with open(random_file(dest_dir=wheels_dir), "w") as f:
        f.write("This is a random file that is not a wheel.")

    _overwrite_pip_requirements_with_wheels(wheels_dir, pip_requirements_path)
    compare_wheels_dir_and_pip_requirements()


def test_update_model_file(tmp_path):
    # Test case: mlflow_model is not None (this occurs the model is logged)
    _, model_file_path, _, _, _ = create_random_model_env(tmp_path, build_wheels=True)
    run_id = random_int()
    mlflow_model = Model(artifact_path=None, run_id=run_id)

    with open(model_file_path) as f:
        original_model_file = yaml.safe_load(f)

    # `wheels` should not exist in the original model_file (sanity check)
    assert "wheels" not in original_model_file["flavors"]["python_function"]["artifacts"]
    # The `run_id` and `utc_time_created` in the mlflow_model should be different from the
    # `run_id` and  `utc_time_created` in the original model file (sanity check)
    assert original_model_file["run_id"] != mlflow_model.run_id
    assert original_model_file["utc_time_created"] != mlflow_model.utc_time_created

    _update_model_file(mlflow_model, model_file_path)

    with open(model_file_path) as f:
        model_file = yaml.safe_load(f)

    assert model_file["run_id"] == mlflow_model.run_id
    assert model_file["utc_time_created"] == mlflow_model.utc_time_created
    # TODO: Update this if we change this section of the model file
    # The key `wheels` should be in under artifacts in MLModel file and should be set to True
    assert (
        "wheels" in model_file["flavors"]["python_function"]["artifacts"]
        and model_file["flavors"]["python_function"]["artifacts"]["wheels"]
    )

    # Test case: mlflow_model is None (this occurs when the model is saved locally)
    _, model_file_path, _, _, _ = create_random_model_env(tmp_path, build_wheels=True)
    mlflow_model = None

    with open(model_file_path) as f:
        original_model_file = yaml.safe_load(f)

    # `wheels` should not exist in the original model_file (sanity check)
    assert "wheels" not in original_model_file["flavors"]["python_function"]["artifacts"]
    # `run_id` should exist in the original model_file (sanity check)
    assert "run_id" in original_model_file

    _update_model_file(mlflow_model, model_file_path)

    with open(model_file_path) as f:
        model_file = yaml.safe_load(f)

    # `run_id` should be deleted from the model_file
    assert "run_id" not in model_file
    # `utc_time_created` should be updated to the current time
    assert model_file["utc_time_created"] != original_model_file["utc_time_created"]
    # TODO: Update this if we change this section of the model file
    # The key `wheels` should be in under artifacts in MLModel file and should be set to True
    assert (
        "wheels" in model_file["flavors"]["python_function"]["artifacts"]
        and model_file["flavors"]["python_function"]["artifacts"]["wheels"]
    )


def test_update_conda_file_with_wheels(tmp_path):
    _, _, conda_yaml_path, pip_requirements_path, wheels_dir = create_random_model_env(
        tmp_path, build_wheels=True
    )

    def get_pip_requirements_from_conda_file():
        with open(conda_yaml_path) as f:
            conda_env = yaml.safe_load(f)

        conda_pip_requirements_list = []
        dependencies = conda_env.get("dependencies")

        for dependency in dependencies:
            if _is_pip_deps(dependency):
                conda_pip_requirements_list = dependency["pip"]

        return conda_pip_requirements_list

    original_pip_requirements_list = _get_pip_requirements_list(pip_requirements_path)
    original_conda_pip_requirements_list = get_pip_requirements_from_conda_file()

    original_conda_pip_requirements_list.sort()
    original_pip_requirements_list.sort()

    assert original_conda_pip_requirements_list == original_pip_requirements_list

    # Update pip requirements
    _overwrite_pip_requirements_with_wheels(wheels_dir, pip_requirements_path)

    # Update conda.yaml
    _update_conda_file_with_wheels(conda_yaml_path, pip_requirements_path)

    wheels_requirements_list = _get_pip_requirements_list(pip_requirements_path)
    conda_wheels_requirements_list = get_pip_requirements_from_conda_file()

    conda_wheels_requirements_list.sort()
    wheels_requirements_list.sort()

    assert conda_wheels_requirements_list == wheels_requirements_list
