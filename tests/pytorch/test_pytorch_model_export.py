from __future__ import print_function

import os
import json

import pytest
import numpy as np
import pandas as pd
import pandas.testing
import sklearn.datasets as datasets
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import mlflow.pyfunc as pyfunc
import mlflow.pytorch
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import score_model_in_sagemaker_docker_container


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


@pytest.fixture(scope='module')
def model(data):
    dataset = get_dataset(data)
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            batch_size = batch[0].shape[0]
            y_pred = model(batch[0]).squeeze(dim=1)
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()

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
def predicted(model, data):
    return _predict(model, data)


def test_log_model(model, data, predicted):
    old_uri = tracking.get_tracking_uri()
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            try:
                tracking.set_tracking_uri(tmp.path("test"))
                if should_start_run:
                    mlflow.start_run()

                mlflow.pytorch.log_model(model, artifact_path="pytorch")

                # Load model
                run_id = mlflow.active_run().info.run_uuid
                model_loaded = mlflow.pytorch.load_model("pytorch", run_id=run_id)

                test_predictions = _predict(model_loaded, data)
                np.testing.assert_array_equal(test_predictions, predicted)
            finally:
                mlflow.end_run()
                tracking.set_tracking_uri(old_uri)


def test_raise_exception(model):
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        path = tmp.path("model")
        with pytest.raises(MlflowException):
            mlflow.pytorch.load_model(path)

        with pytest.raises(TypeError):
            mlflow.pytorch.save_model([1, 2, 3], path)

        mlflow.pytorch.save_model(model, path)
        with pytest.raises(RuntimeError):
            mlflow.pytorch.save_model(model, path)

        from mlflow import sklearn
        import sklearn.neighbors as knn
        import pickle
        path = tmp.path("knn.pkl")
        knn = knn.KNeighborsClassifier()
        with open(path, "wb") as f:
            pickle.dump(knn, f)
        path = tmp.path("knn")
        sklearn.save_model(knn, path=path)
        with pytest.raises(MlflowException):
            mlflow.pytorch.load_model(path)


def test_save_and_load_model(model, model_path, data, predicted):
    x, y = data
    mlflow.pytorch.save_model(model, model_path)

    # Loading pytorch model
    model_loaded = mlflow.pytorch.load_model(model_path)
    np.testing.assert_array_equal(_predict(model_loaded, data), predicted)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    np.testing.assert_array_almost_equal(
            pyfunc_loaded.predict(x).values[:, 0], predicted, decimal=4)


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
        model, model_path, pytorch_custom_env):
    mlflow.pytorch.save_model(
            pytorch_model=model, path=model_path, conda_env=pytorch_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pytorch_custom_env

    with open(pytorch_custom_env, "r") as f:
        pytorch_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == pytorch_custom_env_text


def test_model_save_accepts_conda_env_as_dict(model, model_path):
    conda_env = dict(mlflow.pytorch.DEFAULT_CONDA_ENV)
    conda_env["dependencies"].append("pytest")
    mlflow.pytorch.save_model(pytorch_model=model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
        model, pytorch_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.pytorch.log_model(pytorch_model=model,
                                 artifact_path=artifact_path,
                                 conda_env=pytorch_custom_env)
        run_id = mlflow.active_run().info.run_uuid
    model_path = tracking.utils._get_model_log_dir(artifact_path, run_id)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pytorch_custom_env

    with open(pytorch_custom_env, "r") as f:
        pytorch_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == pytorch_custom_env_text


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        model, model_path):
    mlflow.pytorch.save_model(pytorch_model=model, path=model_path, conda_env=None)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.pytorch.DEFAULT_CONDA_ENV


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.pytorch.log_model(pytorch_model=model,
                                 artifact_path=artifact_path,
                                 conda_env=None)
        run_id = mlflow.active_run().info.run_uuid
    model_path = tracking.utils._get_model_log_dir(artifact_path, run_id)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.pytorch.DEFAULT_CONDA_ENV


@pytest.mark.release
def test_sagemaker_docker_model_scoring_with_default_conda_env(model, model_path, data, predicted):
    mlflow.pytorch.save_model(pytorch_model=model, path=model_path, conda_env=None)

    scoring_response = score_model_in_sagemaker_docker_container(
            model_path=model_path,
            data=data[0],
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
            flavor=mlflow.pyfunc.FLAVOR_NAME,
            activity_polling_timeout_seconds=360)
    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))

    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        predicted,
        decimal=4)
