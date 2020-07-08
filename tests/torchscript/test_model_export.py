import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import logging
import mock
import pickle

import pytest
import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import yaml

import mlflow.pyfunc as pyfunc
import mlflow.torchscript
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import tracking
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.helper_functions import pyfunc_serve_and_score_model

_logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    y = data['target']
    x = data.drop('target', axis=1)
    return x, y


def get_dataset(data):
    x, y = data
    dataset = [(xi.astype(np.float32), yi.astype(np.float32))
               for xi, yi in zip(x.values, y.values)]
    return dataset


def train_model(model, data):
    dataset = get_dataset(data)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=True, drop_last=False)

    model.train()
    for _ in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            batch_size = batch[0].shape[0]
            y_pred = model(batch[0]).squeeze(dim=1)
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()


@pytest.fixture(scope='module')
def sequential_model(data):
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )

    model = torch.jit.script(model)
    train_model(model=model, data=data)
    return model


class PyTorchModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


@pytest.fixture(scope='module')
def subclassed_model(data):
    """
    A custom PyTorch model inheriting from ``torch.nn.Module`` whose class is defined in the
    "__main__" scope.
    """
    model = torch.jit.script(PyTorchModel())
    train_model(model=model, data=data)
    return model


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def pytorch_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
            conda_env,
            additional_conda_deps=["pytorch", "torchvision", "pytest"],
            additional_conda_channels=["pytorch"])
    return conda_env


def _predict(model, data):
    dataset = get_dataset(data)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False, drop_last=False)
    predictions = np.zeros((len(dataloader.sampler),))
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            y_preds = model(batch[0]).squeeze(dim=1).numpy()
            predictions[i * batch_size:(i + 1) * batch_size] = y_preds
    return predictions


@pytest.fixture(scope='module')
def sequential_predicted(sequential_model, data):
    return _predict(sequential_model, data)


@pytest.mark.large
def test_signature_and_examples_are_saved_correctly(sequential_model, data):
    model = sequential_model
    signature_ = infer_signature(*data)
    example_ = data[0].head(3)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.torchscript.save_model(model, path=path,
                                              signature=signature,
                                              input_example=example)
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


@pytest.mark.large
def test_log_model(sequential_model, data, sequential_predicted):
    old_uri = tracking.get_tracking_uri()
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            try:
                tracking.set_tracking_uri(tmp.path("test"))
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "torchscript"
                mlflow.torchscript.log_model(sequential_model, artifact_path=artifact_path)
                model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=mlflow.active_run().info.run_id,
                    artifact_path=artifact_path)

                # Load model
                sequential_model_loaded = mlflow.torchscript.load_model(model_uri=model_uri)

                test_predictions = _predict(sequential_model_loaded, data)
                np.testing.assert_array_equal(test_predictions, sequential_predicted)
            finally:
                mlflow.end_run()
                tracking.set_tracking_uri(old_uri)


def test_log_model_calls_register_model(subclassed_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.torchscript.log_model(
            artifact_path=artifact_path,
            model=subclassed_model,
            conda_env=None,
            registered_model_name="AdsModel1")
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=mlflow.active_run().info.run_id,
                                                            artifact_path=artifact_path)
        mlflow.register_model.assert_called_once_with(model_uri, "AdsModel1")


def test_log_model_no_registered_model_name(subclassed_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.torchscript.log_model(
            artifact_path=artifact_path,
            model=subclassed_model,
            conda_env=None)
        mlflow.register_model.assert_not_called()


@pytest.mark.large
def test_raise_exception(sequential_model):
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        path = tmp.path("model")
        with pytest.raises(IOError):
            mlflow.torchscript.load_model(path)

        with pytest.raises(TypeError):
            mlflow.torchscript.save_model([1, 2, 3], path)

        mlflow.torchscript.save_model(sequential_model, path)
        with pytest.raises(RuntimeError):
            mlflow.torchscript.save_model(sequential_model, path)

        from mlflow import sklearn
        import sklearn.neighbors as knn
        path = tmp.path("knn.pkl")
        knn = knn.KNeighborsClassifier()
        with open(path, "wb") as f:
            pickle.dump(knn, f)
        path = tmp.path("knn")
        sklearn.save_model(knn, path=path)
        with pytest.raises(MlflowException):
            mlflow.torchscript.load_model(path)


@pytest.mark.large
def test_save_and_load_model(sequential_model, model_path, data, sequential_predicted):
    mlflow.torchscript.save_model(sequential_model, model_path)

    # Loading torchscript model
    sequential_model_loaded = mlflow.torchscript.load_model(model_path)
    np.testing.assert_array_equal(_predict(sequential_model_loaded, data), sequential_predicted)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    np.testing.assert_array_almost_equal(
        pyfunc_loaded.predict(data[0]).values[:, 0], sequential_predicted, decimal=4)


@pytest.mark.large
def test_load_model_from_remote_uri_succeeds(
        sequential_model, model_path, mock_s3_bucket, data, sequential_predicted):
    mlflow.torchscript.save_model(sequential_model, model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    sequential_model_loaded = mlflow.torchscript.load_model(model_uri=model_uri)
    np.testing.assert_array_equal(_predict(sequential_model_loaded, data), sequential_predicted)


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
        sequential_model, model_path, pytorch_custom_env):
    mlflow.torchscript.save_model(
            model=sequential_model, path=model_path, conda_env=pytorch_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pytorch_custom_env

    with open(pytorch_custom_env, "r") as f:
        pytorch_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == pytorch_custom_env_text


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(sequential_model, model_path):
    conda_env = dict(mlflow.torchscript.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.torchscript.save_model(model=sequential_model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
        sequential_model, pytorch_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.torchscript.log_model(model=sequential_model,
                                     artifact_path=artifact_path,
                                     conda_env=pytorch_custom_env)
        model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path))

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pytorch_custom_env

    with open(pytorch_custom_env, "r") as f:
        pytorch_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == pytorch_custom_env_text


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        sequential_model, model_path):
    mlflow.torchscript.save_model(model=sequential_model, path=model_path, conda_env=None)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.torchscript.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        sequential_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.torchscript.log_model(model=sequential_model,
                                     artifact_path=artifact_path,
                                     conda_env=None)
        model_path = _download_artifact_from_uri("runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path))

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.torchscript.get_default_conda_env()


@pytest.mark.large
def test_load_model_with_differing_pytorch_version_logs_warning(sequential_model, model_path):
    mlflow.torchscript.save_model(model=sequential_model, path=model_path)
    saver_pytorch_version = "1.0"
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    model_config.flavors[mlflow.torchscript.FLAVOR_NAME]["pytorch_version"] = saver_pytorch_version
    model_config.save(model_config_path)

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text % args % kwargs)

    loader_pytorch_version = "0.8.2"
    with mock.patch("mlflow.torchscript._logger.warning") as warn_mock,\
            mock.patch("torch.__version__") as torch_version_mock:
        torch_version_mock.__str__ = lambda *args, **kwargs: loader_pytorch_version
        warn_mock.side_effect = custom_warn
        mlflow.torchscript.load_model(model_uri=model_path)

    assert any([
        "does not match installed PyTorch version" in log_message and
        saver_pytorch_version in log_message and
        loader_pytorch_version in log_message
        for log_message in log_messages
    ])


@pytest.mark.large
def test_pyfunc_model_serving_with_default_conda_env(
        subclassed_model, model_path, data):
    mlflow.torchscript.save_model(
        path=model_path,
        model=subclassed_model,
        conda_env=None)

    scoring_response = pyfunc_serve_and_score_model(
            model_uri=model_path,
            data=data[0],
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
            extra_args=["--no-conda"])
    assert scoring_response.status_code == 200

    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=subclassed_model, data=data),
        decimal=4)
