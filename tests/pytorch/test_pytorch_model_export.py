import importlib
import json
import logging
import os
import pickle
import re
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn import datasets
from torch import nn
from torch.utils.data import DataLoader

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.pytorch
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import _read_example
from mlflow.pytorch import get_default_conda_env
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import Schema, TensorSpec
from mlflow.utils.environment import _mlflow_additional_pip_env, _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_conda_env_requirements,
    _compare_logged_code_paths,
    _is_available_on_pypi,
    _is_importable,
    _mlflow_major_version_string,
    assert_array_almost_equal,
    assert_register_model_called_with_local_model_path,
    get_serving_input_example,
)

_logger = logging.getLogger(__name__)

# This test suite is included as a code dependency when testing PyTorch model scoring in new
# processes and docker containers. In these environments, the `tests` module is not available.
# Therefore, we attempt to import from `tests` and gracefully emit a warning if it's unavailable.
try:
    from tests.helper_functions import pyfunc_serve_and_score_model
except ImportError:
    _logger.warning(
        "Failed to import test helper functions. Tests depending on these functions may fail!"
    )

EXTRA_PYFUNC_SERVING_TEST_ARGS = (
    [] if _is_available_on_pypi("torch") else ["--env-manager", "local"]
)


@pytest.fixture(scope="module")
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
    )
    y = data["target"]
    x = data.drop("target", axis=1)
    return x, y


@pytest.fixture(scope="module")
def iris_tensor_spec():
    return ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 4))]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 1))]),
    )


def get_dataset(data):
    x, y = data
    return [(xi.astype(np.float32), yi.astype(np.float32)) for xi, yi in zip(x.values, y.values)]


def train_model(model, data):
    dataset = get_dataset(data)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False
    )

    model.train()
    for _ in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            batch_size = batch[0].shape[0]
            y_pred = model(batch[0]).squeeze(dim=1)
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()


def get_sequential_model():
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))


@pytest.fixture
def sequential_model(data, scripted_model):
    model = get_sequential_model()
    if scripted_model:
        model = torch.jit.script(model)

    train_model(model=model, data=data)
    return model


def get_subclassed_model_definition():
    """
    Defines a PyTorch model class that inherits from ``torch.nn.Module``. This method can be invoked
    within a pytest fixture to define the model class in the ``__main__`` scope. Alternatively, it
    can be invoked within a module to define the class in the module's scope.
    """

    class SubclassedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)

        def forward(self, x):
            return self.linear(x)

    return SubclassedModel


@pytest.fixture(scope="module")
def main_scoped_subclassed_model(data):
    """
    A custom PyTorch model inheriting from ``torch.nn.Module`` whose class is defined in the
    "__main__" scope.
    """
    model_class = get_subclassed_model_definition()
    model = model_class()
    train_model(model=model, data=data)
    return model


class ModuleScopedSubclassedModel(get_subclassed_model_definition()):
    """
    A custom PyTorch model class defined in the test module scope. This is a subclass of
    ``torch.nn.Module``.
    """


@pytest.fixture(scope="module")
def module_scoped_subclassed_model(data):
    """
    A custom PyTorch model inheriting from ``torch.nn.Module`` whose class is defined in the test
    module scope.
    """
    model = ModuleScopedSubclassedModel()
    train_model(model=model, data=data)
    return model


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def pytorch_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["pytorch", "torchvision", "pytest"])
    return conda_env


def _predict(model, data):
    dataset = get_dataset(data)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    predictions = np.zeros((len(dataloader.sampler),))
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            y_preds = model(batch[0]).squeeze(dim=1).numpy()
            predictions[i * batch_size : (i + 1) * batch_size] = y_preds
    return predictions


@pytest.fixture
def sequential_predicted(sequential_model, data):
    return _predict(sequential_model, data)


@pytest.mark.parametrize("scripted_model", [True, False])
def test_signature_and_examples_are_saved_correctly(sequential_model, data, iris_tensor_spec):
    model = sequential_model
    example_ = data[0].head(3).values.astype(np.float32)
    for signature in (None, iris_tensor_spec):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.pytorch.save_model(
                    model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                if signature is None and example is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == iris_tensor_spec
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    np.testing.assert_allclose(_read_example(mlflow_model, path), example)


@pytest.mark.parametrize("scripted_model", [True, False])
def test_log_model(sequential_model, data, sequential_predicted):
    try:
        artifact_path = "pytorch"
        model_info = mlflow.pytorch.log_model(sequential_model, artifact_path=artifact_path)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri

        sequential_model_loaded = mlflow.pytorch.load_model(model_uri=model_uri)
        test_predictions = _predict(sequential_model_loaded, data)
        np.testing.assert_array_equal(test_predictions, sequential_predicted)
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(module_scoped_subclassed_model):
    custom_pickle_module = pickle
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.pytorch.log_model(
            artifact_path=artifact_path,
            pytorch_model=module_scoped_subclassed_model,
            pickle_module=custom_pickle_module,
            registered_model_name="AdsModel1",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_uri,
            registered_model_name="AdsModel1",
        )


def test_log_model_no_registered_model_name(module_scoped_subclassed_model):
    custom_pickle_module = pickle
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.pytorch.log_model(
            artifact_path=artifact_path,
            pytorch_model=module_scoped_subclassed_model,
            pickle_module=custom_pickle_module,
        )
        mlflow.tracking._model_registry.fluent._register_model.assert_not_called()


@pytest.mark.parametrize("scripted_model", [True, False])
def test_raise_exception(sequential_model):
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        path = tmp.path("model")
        with pytest.raises(IOError, match="No such file or directory"):
            mlflow.pytorch.load_model(path)

        with pytest.raises(TypeError, match="Argument 'pytorch_model' should be a torch.nn.Module"):
            mlflow.pytorch.save_model([1, 2, 3], path)

        mlflow.pytorch.save_model(sequential_model, path)
        with pytest.raises(MlflowException, match=f"Path '{os.path.abspath(path)}' already exists"):
            mlflow.pytorch.save_model(sequential_model, path)

        import sklearn.neighbors as knn

        from mlflow import sklearn

        path = tmp.path("knn.pkl")
        knn = knn.KNeighborsClassifier()
        with open(path, "wb") as f:
            pickle.dump(knn, f)
        path = tmp.path("knn")
        sklearn.save_model(knn, path=path)
        with pytest.raises(MlflowException, match='Model does not have the "pytorch" flavor'):
            mlflow.pytorch.load_model(path)


@pytest.mark.parametrize("scripted_model", [True, False])
def test_save_and_load_model(sequential_model, model_path, data, sequential_predicted):
    mlflow.pytorch.save_model(sequential_model, model_path)

    # Loading pytorch model
    sequential_model_loaded = mlflow.pytorch.load_model(model_path)
    np.testing.assert_array_equal(_predict(sequential_model_loaded, data), sequential_predicted)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_array_almost_equal(
        pyfunc_loaded.predict(data[0]).values[:, 0], sequential_predicted, decimal=4
    )


@pytest.mark.parametrize("scripted_model", [True, False])
def test_pyfunc_model_works_with_np_input_type(
    sequential_model, model_path, data, sequential_predicted
):
    mlflow.pytorch.save_model(sequential_model, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    # predict works with dataframes
    df_result = pyfunc_loaded.predict(data[0])
    assert type(df_result) == pd.DataFrame
    np.testing.assert_array_almost_equal(df_result.values[:, 0], sequential_predicted, decimal=4)

    # predict works with numpy ndarray
    np_result = pyfunc_loaded.predict(data[0].values.astype(np.float32))
    assert type(np_result) == np.ndarray
    np.testing.assert_array_almost_equal(np_result[:, 0], sequential_predicted, decimal=4)

    # predict does not work with lists
    with pytest.raises(
        TypeError, match="The PyTorch flavor does not support List or Dict input types"
    ):
        pyfunc_loaded.predict([1, 2, 3, 4])

    # predict does not work with scalars
    with pytest.raises(TypeError, match="Input data should be pandas.DataFrame or numpy.ndarray"):
        pyfunc_loaded.predict(4)


@pytest.mark.parametrize("scripted_model", [True, False])
def test_load_model_from_remote_uri_succeeds(
    sequential_model, model_path, mock_s3_bucket, data, sequential_predicted
):
    mlflow.pytorch.save_model(sequential_model, model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    sequential_model_loaded = mlflow.pytorch.load_model(model_uri=model_uri)
    np.testing.assert_array_equal(_predict(sequential_model_loaded, data), sequential_predicted)


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    sequential_model, model_path, pytorch_custom_env
):
    mlflow.pytorch.save_model(
        pytorch_model=sequential_model, path=model_path, conda_env=pytorch_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pytorch_custom_env

    with open(pytorch_custom_env) as f:
        pytorch_custom_env_text = f.read()
    with open(saved_conda_env_path) as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == pytorch_custom_env_text


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_save_persists_requirements_in_mlflow_model_directory(
    sequential_model, model_path, pytorch_custom_env
):
    mlflow.pytorch.save_model(
        pytorch_model=sequential_model, path=model_path, conda_env=pytorch_custom_env
    )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(pytorch_custom_env, saved_pip_req_path)


@pytest.mark.parametrize("scripted_model", [False])
def test_save_model_with_pip_requirements(sequential_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    tmpdir1 = tmp_path.joinpath("1")
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    mlflow.pytorch.save_model(sequential_model, tmpdir1, pip_requirements=str(req_file))
    _assert_pip_requirements(tmpdir1, [expected_mlflow_version, "a"], strict=True)

    # List of requirements
    tmpdir2 = tmp_path.joinpath("2")
    mlflow.pytorch.save_model(sequential_model, tmpdir2, pip_requirements=[f"-r {req_file}", "b"])
    _assert_pip_requirements(tmpdir2, [expected_mlflow_version, "a", "b"], strict=True)

    # Constraints file
    tmpdir3 = tmp_path.joinpath("3")
    mlflow.pytorch.save_model(sequential_model, tmpdir3, pip_requirements=[f"-c {req_file}", "b"])
    _assert_pip_requirements(
        tmpdir3, [expected_mlflow_version, "b", "-c constraints.txt"], ["a"], strict=True
    )


@pytest.mark.parametrize("scripted_model", [False])
def test_save_model_with_extra_pip_requirements(sequential_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.pytorch.get_default_pip_requirements()

    # Path to a requirements file
    tmpdir1 = tmp_path.joinpath("1")
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    mlflow.pytorch.save_model(sequential_model, tmpdir1, extra_pip_requirements=str(req_file))
    _assert_pip_requirements(tmpdir1, [expected_mlflow_version, *default_reqs, "a"])

    # List of requirements
    tmpdir2 = tmp_path.joinpath("2")
    mlflow.pytorch.save_model(
        sequential_model, tmpdir2, extra_pip_requirements=[f"-r {req_file}", "b"]
    )
    _assert_pip_requirements(tmpdir2, [expected_mlflow_version, *default_reqs, "a", "b"])

    # Constraints file
    tmpdir3 = tmp_path.joinpath("3")
    mlflow.pytorch.save_model(
        sequential_model, tmpdir3, extra_pip_requirements=[f"-c {req_file}", "b"]
    )
    _assert_pip_requirements(
        tmpdir3, [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"], ["a"]
    )


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_save_accepts_conda_env_as_dict(sequential_model, model_path):
    conda_env = dict(mlflow.pytorch.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.pytorch.save_model(pytorch_model=sequential_model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    sequential_model, pytorch_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model,
            artifact_path=artifact_path,
            conda_env=pytorch_custom_env,
        )
        model_path = _download_artifact_from_uri(
            f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pytorch_custom_env

    with open(pytorch_custom_env) as f:
        pytorch_custom_env_text = f.read()
    with open(saved_conda_env_path) as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == pytorch_custom_env_text


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_log_persists_requirements_in_mlflow_model_directory(
    sequential_model, pytorch_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model,
            artifact_path=artifact_path,
            conda_env=pytorch_custom_env,
        )
        model_path = _download_artifact_from_uri(
            f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(pytorch_custom_env, saved_pip_req_path)


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sequential_model, model_path
):
    mlflow.pytorch.save_model(pytorch_model=sequential_model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.pytorch.get_default_pip_requirements())


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sequential_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.pytorch.log_model(sequential_model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    _assert_pip_requirements(model_uri, mlflow.pytorch.get_default_pip_requirements())


@pytest.mark.parametrize("scripted_model", [True, False])
def test_load_model_with_differing_pytorch_version_logs_warning(sequential_model, model_path):
    mlflow.pytorch.save_model(pytorch_model=sequential_model, path=model_path)
    saver_pytorch_version = "1.0"
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    model_config.flavors[mlflow.pytorch.FLAVOR_NAME]["pytorch_version"] = saver_pytorch_version
    model_config.save(model_config_path)

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text % args % kwargs)

    loader_pytorch_version = "0.8.2"
    with mock.patch("mlflow.pytorch._logger.warning") as warn_mock, mock.patch(
        "torch.__version__", loader_pytorch_version
    ):
        warn_mock.side_effect = custom_warn
        mlflow.pytorch.load_model(model_uri=model_path)

    assert any(
        "does not match installed PyTorch version" in log_message
        and saver_pytorch_version in log_message
        and loader_pytorch_version in log_message
        for log_message in log_messages
    )


def test_pyfunc_model_serving_with_module_scoped_subclassed_model_and_default_conda_env(
    module_scoped_subclassed_model, data
):
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            pytorch_model=module_scoped_subclassed_model,
            artifact_path="pytorch_model",
            code_paths=[__file__],
            input_example=data[0],
        )

    inference_payload = get_serving_input_example(model_info.model_uri)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200

    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=module_scoped_subclassed_model, data=data),
        decimal=4,
    )


def test_save_model_with_wrong_codepaths_fails_correctly(
    module_scoped_subclassed_model, model_path, data
):
    with pytest.raises(TypeError, match="Argument code_paths should be a list, not <class 'str'>"):
        mlflow.pytorch.save_model(
            path=model_path, pytorch_model=module_scoped_subclassed_model, code_paths="some string"
        )


def test_pyfunc_model_serving_with_main_scoped_subclassed_model_and_custom_pickle_module(
    main_scoped_subclassed_model, data
):
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            pytorch_model=main_scoped_subclassed_model,
            artifact_path="pytorch_model",
            pickle_module=mlflow_pytorch_pickle_module,
            input_example=data[0],
        )

    inference_payload = get_serving_input_example(model_info.model_uri)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200

    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=main_scoped_subclassed_model, data=data),
        decimal=4,
    )


def test_load_model_succeeds_with_dependencies_specified_via_code_paths(
    module_scoped_subclassed_model, model_path, data
):
    # Save a PyTorch model whose class is defined in the current test suite. Because the
    # `tests` module is not available when the model is deployed for local scoring, we include
    # the test suite file as a code dependency
    mlflow.pytorch.save_model(
        path=model_path,
        pytorch_model=module_scoped_subclassed_model,
        code_paths=[__file__],
    )

    # Define a custom pyfunc model that loads a PyTorch model artifact using
    # `mlflow.pytorch.load_model`
    class TorchValidatorModel(pyfunc.PythonModel):
        def load_context(self, context):
            self.pytorch_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])

        def predict(self, _, model_input, params=None):
            with torch.no_grad():
                input_tensor = torch.from_numpy(model_input.values.astype(np.float32))
                output_tensor = self.pytorch_model(input_tensor)
                return pd.DataFrame(output_tensor.numpy())

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        model_info = pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            python_model=TorchValidatorModel(),
            artifacts={"pytorch_model": model_path},
            input_example=data[0],
            # save file into code_paths, otherwise after first model loading (happens when
            # validating input_example) then we can not load the model again
            code_paths=[__file__],
        )

    # Deploy the custom pyfunc model and ensure that it is able to successfully load its
    # constituent PyTorch model via `mlflow.pytorch.load_model`
    inference_payload = get_serving_input_example(model_info.model_uri)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200

    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=module_scoped_subclassed_model, data=data),
        decimal=4,
    )


def test_load_pyfunc_loads_torch_model_using_pickle_module_specified_at_save_time(
    module_scoped_subclassed_model, model_path
):
    custom_pickle_module = pickle

    mlflow.pytorch.save_model(
        path=model_path,
        pytorch_model=module_scoped_subclassed_model,
        pickle_module=custom_pickle_module,
    )

    import_module_fn = importlib.import_module
    imported_modules = []

    def track_module_imports(module_name):
        imported_modules.append(module_name)
        return import_module_fn(module_name)

    with mock.patch("importlib.import_module") as import_mock, mock.patch(
        "torch.load"
    ) as torch_load_mock:
        import_mock.side_effect = track_module_imports
        pyfunc.load_model(model_path)

    torch_load_mock.assert_called_with(mock.ANY, pickle_module=custom_pickle_module)
    assert custom_pickle_module.__name__ in imported_modules


def test_load_model_loads_torch_model_using_pickle_module_specified_at_save_time(
    module_scoped_subclassed_model,
):
    custom_pickle_module = pickle

    artifact_path = "pytorch_model"
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            artifact_path=artifact_path,
            pytorch_model=module_scoped_subclassed_model,
            pickle_module=custom_pickle_module,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

    import_module_fn = importlib.import_module
    imported_modules = []

    def track_module_imports(module_name):
        imported_modules.append(module_name)
        return import_module_fn(module_name)

    with mock.patch("importlib.import_module") as import_mock, mock.patch(
        "torch.load"
    ) as torch_load_mock:
        import_mock.side_effect = track_module_imports
        pyfunc.load_model(model_uri=model_uri)

    torch_load_mock.assert_called_with(mock.ANY, pickle_module=custom_pickle_module)
    assert custom_pickle_module.__name__ in imported_modules


def test_load_pyfunc_succeeds_when_data_is_model_file_instead_of_directory(
    module_scoped_subclassed_model, model_path, data
):
    """
    This test verifies that PyTorch models saved in older versions of MLflow are loaded successfully
    by ``mlflow.pytorch.load_model``. The ``data`` path associated with these older models is
    serialized PyTorch model file, as opposed to the current format: a directory containing a
    serialized model file and pickle module information.
    """
    mlflow.pytorch.save_model(path=model_path, pytorch_model=module_scoped_subclassed_model)

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    pyfunc_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME)
    assert pyfunc_conf is not None
    model_data_path = os.path.join(model_path, pyfunc_conf[pyfunc.DATA])
    assert os.path.exists(model_data_path)
    assert mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME in os.listdir(model_data_path)
    pyfunc_conf[pyfunc.DATA] = os.path.join(
        model_data_path, mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME
    )
    model_conf.save(model_conf_path)

    loaded_pyfunc = pyfunc.load_model(model_path)

    np.testing.assert_array_almost_equal(
        loaded_pyfunc.predict(data[0]),
        pd.DataFrame(_predict(model=module_scoped_subclassed_model, data=data)),
        decimal=4,
    )


def test_load_model_succeeds_when_data_is_model_file_instead_of_directory(
    module_scoped_subclassed_model, model_path, data
):
    """
    This test verifies that PyTorch models saved in older versions of MLflow are loaded successfully
    by ``mlflow.pytorch.load_model``. The ``data`` path associated with these older models is
    serialized PyTorch model file, as opposed to the current format: a directory containing a
    serialized model file and pickle module information.
    """
    artifact_path = "pytorch_model"
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            artifact_path=artifact_path, pytorch_model=module_scoped_subclassed_model
        )
        model_path = _download_artifact_from_uri(
            f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        )

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    pyfunc_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME)
    assert pyfunc_conf is not None
    model_data_path = os.path.join(model_path, pyfunc_conf[pyfunc.DATA])
    assert os.path.exists(model_data_path)
    assert mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME in os.listdir(model_data_path)
    pyfunc_conf[pyfunc.DATA] = os.path.join(
        model_data_path, mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME
    )
    model_conf.save(model_conf_path)

    loaded_pyfunc = pyfunc.load_model(model_path)

    np.testing.assert_array_almost_equal(
        loaded_pyfunc.predict(data[0]),
        pd.DataFrame(_predict(model=module_scoped_subclassed_model, data=data)),
        decimal=4,
    )


def test_load_model_allows_user_to_override_pickle_module_via_keyword_argument(
    module_scoped_subclassed_model, model_path
):
    mlflow.pytorch.save_model(
        path=model_path, pytorch_model=module_scoped_subclassed_model, pickle_module=pickle
    )

    with mock.patch("torch.load") as torch_load_mock, mock.patch(
        "mlflow.pytorch._logger.warning"
    ) as warn_mock:
        mlflow.pytorch.load_model(model_uri=model_path, pickle_module=mlflow_pytorch_pickle_module)
        torch_load_mock.assert_called_with(mock.ANY, pickle_module=mlflow_pytorch_pickle_module)
        warn_mock.assert_any_call(mock.ANY, mlflow_pytorch_pickle_module.__name__, pickle.__name__)


def test_load_model_raises_exception_when_pickle_module_cannot_be_imported(
    main_scoped_subclassed_model, model_path
):
    mlflow.pytorch.save_model(path=model_path, pytorch_model=main_scoped_subclassed_model)

    bad_pickle_module_name = "not.a.real.module"

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    model_data_path = os.path.join(model_path, pyfunc_conf[pyfunc.DATA])
    assert os.path.exists(model_data_path)
    assert mlflow.pytorch._PICKLE_MODULE_INFO_FILE_NAME in os.listdir(model_data_path)
    with open(
        os.path.join(model_data_path, mlflow.pytorch._PICKLE_MODULE_INFO_FILE_NAME), "w"
    ) as f:
        f.write(bad_pickle_module_name)

    with pytest.raises(
        MlflowException,
        match=r"Failed to import the pickle module.+" + re.escape(bad_pickle_module_name),
    ):
        mlflow.pytorch.load_model(model_uri=model_path)


def test_pyfunc_serve_and_score(data):
    model = torch.nn.Linear(4, 1)
    train_model(model=model, data=data)

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(model, artifact_path="model", input_example=data[0])

    inference_payload = get_serving_input_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    from mlflow.deployments import PredictionsResponse

    scores = PredictionsResponse.from_json(resp.content).get_predictions()
    np.testing.assert_array_almost_equal(scores.values[:, 0], _predict(model=model, data=data))


@pytest.mark.skipif(not _is_importable("transformers"), reason="This test requires transformers")
def test_pyfunc_serve_and_score_transformers():
    from transformers import BertConfig, BertModel

    from mlflow.deployments import PredictionsResponse

    class MyBertModel(BertModel):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs).last_hidden_state

    model = MyBertModel(
        BertConfig(
            vocab_size=16,
            hidden_size=2,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=2,
        )
    )
    model.eval()

    input_ids = model.dummy_inputs["input_ids"]

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            model, artifact_path="model", input_example=np.array(input_ids.tolist())
        )

    inference_payload = get_serving_input_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )

    scores = PredictionsResponse.from_json(resp.content.decode("utf-8")).get_predictions(
        predictions_format="ndarray"
    )
    assert_array_almost_equal(scores, model(input_ids).detach().numpy(), rtol=1e-6)


@pytest.fixture
def create_requirements_file(tmp_path):
    requirement_file_name = "requirements.txt"
    fp = tmp_path.joinpath(requirement_file_name)
    test_string = "mlflow"
    fp.write_text(test_string)
    return str(fp), test_string


@pytest.mark.parametrize("scripted_model", [True, False])
def test_requirements_file_log_model(create_requirements_file, sequential_model):
    requirements_file, content_expected = create_requirements_file
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model,
            artifact_path="models",
            requirements_file=requirements_file,
        )

        model_uri = "runs:/{run_id}/{model_path}".format(
            run_id=mlflow.active_run().info.run_id, model_path="models"
        )

        # Verify that explicitly specified requirements file overrides default requirements file
        conda_env = get_default_conda_env()
        pip_deps = conda_env["dependencies"][-1]["pip"]
        assert _mlflow_additional_pip_env(pip_deps) != content_expected

        with TempDir(remove_on_exit=True) as tmp:
            model_path = _download_artifact_from_uri(model_uri, tmp.path())
            model_config_path = os.path.join(model_path, "MLmodel")
            model_config = Model.load(model_config_path)
            flavor_config = model_config.flavors["pytorch"]

            assert "requirements_file" in flavor_config
            loaded_requirements_file = flavor_config["requirements_file"]

            assert "path" in loaded_requirements_file
            requirements_file_path = loaded_requirements_file["path"]
            requirements_file_path = os.path.join(model_path, requirements_file_path)
            with open(requirements_file_path) as fp:
                assert fp.read() == content_expected


@pytest.mark.parametrize("scripted_model", [True, False])
def test_requirements_file_save_model(create_requirements_file, sequential_model):
    requirements_file, content_expected = create_requirements_file
    with TempDir(remove_on_exit=True) as tmp:
        model_path = os.path.join(tmp.path(), "models")
        mlflow.pytorch.save_model(
            pytorch_model=sequential_model, path=model_path, requirements_file=requirements_file
        )

        # Verify that explicitly specified requirements file overrides default requirements file
        conda_env = get_default_conda_env()
        pip_deps = conda_env["dependencies"][-1]["pip"]
        assert _mlflow_additional_pip_env(pip_deps) != content_expected

        model_config_path = os.path.join(model_path, "MLmodel")
        model_config = Model.load(model_config_path)
        flavor_config = model_config.flavors["pytorch"]

        assert "requirements_file" in flavor_config
        loaded_requirements_file = flavor_config["requirements_file"]

        assert "path" in loaded_requirements_file
        requirements_file_path = loaded_requirements_file["path"]
        requirements_file_path = os.path.join(model_path, requirements_file_path)
        with open(requirements_file_path) as fp:
            assert fp.read() == content_expected


@pytest.mark.parametrize("scripted_model", [True, False])
def test_log_model_invalid_requirement_file_path(sequential_model):
    with mlflow.start_run(), pytest.raises(
        MlflowException, match="No such file or directory: 'non_existing_file.txt'"
    ):
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model,
            artifact_path="models",
            requirements_file="non_existing_file.txt",
        )


@pytest.mark.parametrize("scripted_model", [True, False])
def test_log_model_invalid_requirement_file_type(sequential_model):
    with mlflow.start_run(), pytest.raises(
        TypeError, match="Path to requirements file should be a string"
    ):
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model,
            artifact_path="models",
            requirements_file=["non_existing_file.txt"],
        )


def test_save_model_emits_deprecation_warning_for_requirements_file(tmp_path):
    reqs_file = tmp_path.joinpath("requirements.txt")
    reqs_file.write_text("torch")
    with pytest.warns(FutureWarning, match="`requirements_file` has been deprecated"):
        mlflow.pytorch.save_model(
            get_sequential_model(),
            tmp_path.joinpath("model"),
            requirements_file=str(reqs_file),
        )


@pytest.fixture
def create_extra_files(tmp_path):
    fp1 = tmp_path.joinpath("extra1.txt")
    fp2 = tmp_path.joinpath("extra2.txt")
    fp1.write_text("1")
    fp2.write_text("2")
    return [str(fp1), str(fp2)], ["1", "2"]


@pytest.mark.parametrize("scripted_model", [True, False])
def test_extra_files_log_model(create_extra_files, sequential_model):
    extra_files, contents_expected = create_extra_files
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model, artifact_path="models", extra_files=extra_files
        )

        model_uri = "runs:/{run_id}/{model_path}".format(
            run_id=mlflow.active_run().info.run_id, model_path="models"
        )
        with TempDir(remove_on_exit=True) as tmp:
            model_path = _download_artifact_from_uri(model_uri, tmp.path())
            model_config_path = os.path.join(model_path, "MLmodel")
            model_config = Model.load(model_config_path)
            flavor_config = model_config.flavors["pytorch"]

            assert "extra_files" in flavor_config
            loaded_extra_files = flavor_config["extra_files"]

            for loaded_extra_file, content_expected in zip(loaded_extra_files, contents_expected):
                assert "path" in loaded_extra_file
                extra_file_path = os.path.join(model_path, loaded_extra_file["path"])
                with open(extra_file_path) as fp:
                    assert fp.read() == content_expected


@pytest.mark.parametrize("scripted_model", [True, False])
def test_extra_files_save_model(create_extra_files, sequential_model):
    extra_files, contents_expected = create_extra_files
    with TempDir(remove_on_exit=True) as tmp:
        model_path = os.path.join(tmp.path(), "models")
        mlflow.pytorch.save_model(
            pytorch_model=sequential_model, path=model_path, extra_files=extra_files
        )
        model_config_path = os.path.join(model_path, "MLmodel")
        model_config = Model.load(model_config_path)
        flavor_config = model_config.flavors["pytorch"]

        assert "extra_files" in flavor_config
        loaded_extra_files = flavor_config["extra_files"]

        for loaded_extra_file, content_expected in zip(loaded_extra_files, contents_expected):
            assert "path" in loaded_extra_file
            extra_file_path = os.path.join(model_path, loaded_extra_file["path"])
            with open(extra_file_path) as fp:
                assert fp.read() == content_expected


@pytest.mark.parametrize("scripted_model", [True, False])
def test_log_model_invalid_extra_file_path(sequential_model):
    with mlflow.start_run(), pytest.raises(
        MlflowException, match="No such file or directory: 'non_existing_file.txt'"
    ):
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model,
            artifact_path="models",
            extra_files=["non_existing_file.txt"],
        )


@pytest.mark.parametrize("scripted_model", [True, False])
def test_log_model_invalid_extra_file_type(sequential_model):
    with mlflow.start_run(), pytest.raises(
        TypeError, match="Extra files argument should be a list"
    ):
        mlflow.pytorch.log_model(
            pytorch_model=sequential_model,
            artifact_path="models",
            extra_files="non_existing_file.txt",
        )


def state_dict_equal(state_dict1, state_dict2):
    for key1 in state_dict1:
        if key1 not in state_dict2:
            return False

        value1 = state_dict1[key1]
        value2 = state_dict2[key1]

        if type(value1) != type(value2):
            return False
        elif isinstance(value1, dict):
            if not state_dict_equal(value1, value2):
                return False
        elif isinstance(value1, torch.Tensor):
            if not torch.equal(value1, value2):
                return False
        elif value1 != value2:
            return False
        else:
            continue

    return True


@pytest.mark.parametrize("scripted_model", [True, False])
def test_save_state_dict(sequential_model, model_path, data):
    state_dict = sequential_model.state_dict()
    mlflow.pytorch.save_state_dict(state_dict, model_path)

    loaded_state_dict = mlflow.pytorch.load_state_dict(model_path)
    assert state_dict_equal(loaded_state_dict, state_dict)
    model = get_sequential_model()
    model.load_state_dict(loaded_state_dict)
    np.testing.assert_array_almost_equal(
        _predict(model, data),
        _predict(sequential_model, data),
        decimal=4,
    )


def test_save_state_dict_can_save_nested_state_dict(model_path):
    """
    This test ensures that `save_state_dict` supports a use case described in the page below
    where a user bundles multiple objects (e.g., model, optimizer, learning-rate scheduler)
    into a single nested state_dict and loads it back later for inference or re-training:
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """
    model = get_sequential_model()
    optim = torch.optim.Adam(model.parameters())
    state_dict = {"model": model.state_dict(), "optim": optim.state_dict()}
    mlflow.pytorch.save_state_dict(state_dict, model_path)

    loaded_state_dict = mlflow.pytorch.load_state_dict(model_path)
    assert state_dict_equal(loaded_state_dict, state_dict)
    model.load_state_dict(loaded_state_dict["model"])
    optim.load_state_dict(loaded_state_dict["optim"])


@pytest.mark.parametrize("not_state_dict", [0, "", get_sequential_model()])
def test_save_state_dict_throws_for_invalid_object_type(not_state_dict, model_path):
    with pytest.raises(TypeError, match="Invalid object type for `state_dict`"):
        mlflow.pytorch.save_state_dict(not_state_dict, model_path)


@pytest.mark.parametrize("scripted_model", [True, False])
def test_log_state_dict(sequential_model, data):
    artifact_path = "model"
    state_dict = sequential_model.state_dict()
    with mlflow.start_run():
        mlflow.pytorch.log_state_dict(state_dict, artifact_path)
        state_dict_uri = mlflow.get_artifact_uri(artifact_path)

    loaded_state_dict = mlflow.pytorch.load_state_dict(state_dict_uri)
    assert state_dict_equal(loaded_state_dict, state_dict)
    model = get_sequential_model()
    model.load_state_dict(loaded_state_dict)
    np.testing.assert_array_almost_equal(
        _predict(model, data),
        _predict(sequential_model, data),
        decimal=4,
    )


@pytest.mark.parametrize("scripted_model", [True, False])
def test_log_model_with_code_paths(sequential_model):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.pytorch._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.pytorch.log_model(
            sequential_model, artifact_path=artifact_path, code_paths=[__file__]
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.pytorch.FLAVOR_NAME)
        mlflow.pytorch.load_model(model_uri)
        add_mock.assert_called()


def test_virtualenv_subfield_points_to_correct_path(model_path):
    model = get_sequential_model()
    mlflow.pytorch.save_model(model, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_save_load_with_metadata(sequential_model, model_path):
    mlflow.pytorch.save_model(
        sequential_model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_log_with_metadata(sequential_model):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            sequential_model,
            artifact_path=artifact_path,
            metadata={"metadata_key": "metadata_value"},
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


@pytest.mark.parametrize("scripted_model", [True, False])
def test_model_log_with_signature_inference(sequential_model, data):
    artifact_path = "model"
    example_ = data[0].head(3).values.astype(np.float32)

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            sequential_model, artifact_path=artifact_path, input_example=example_
        )

    assert model_info.signature == ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 4))]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 1))]),
    )
    inference_payload = get_serving_input_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200
    deployed_model_preds = pd.DataFrame(json.loads(response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=sequential_model, data=(data[0].head(3), data[1].head(3))),
        decimal=4,
    )


@pytest.mark.parametrize("scripted_model", [False])
def test_load_model_to_device(sequential_model):
    with mock.patch("mlflow.pytorch._load_model") as load_model_mock:
        with mlflow.start_run():
            model_info = mlflow.pytorch.log_model(sequential_model, artifact_path="pytorch")
            mlflow.pyfunc.load_model(
                model_uri=model_info.model_uri, model_config={"device": "cuda"}
            )
            load_model_mock.assert_called_with(mock.ANY, device="cuda")
            mlflow.pytorch.load_model(model_uri=model_info.model_uri, device="cuda")
            load_model_mock.assert_called_with(path=mock.ANY, device="cuda")
