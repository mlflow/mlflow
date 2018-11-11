from __future__ import print_function

import pytest

import numpy as np
import pandas as pd

import sklearn.datasets as datasets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow.pytorch
from mlflow import tracking
from mlflow.utils.file_utils import TempDir


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
                assert np.all(test_predictions == predicted)
            finally:
                mlflow.end_run()
                tracking.set_tracking_uri(old_uri)


def test_raise_exception(model):
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        path = tmp.path("model")
        with pytest.raises(RuntimeError):
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
        with pytest.raises(ValueError):
            mlflow.pytorch.load_model(path)


def test_save_and_load_model(model, data, predicted):

    x, y = data
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        path = tmp.path("model")
        mlflow.pytorch.save_model(model, path)

        # Loading pytorch model
        model_loaded = mlflow.pytorch.load_model(path)
        assert np.all(_predict(model_loaded, data) == predicted)

        # Loading pyfunc model
        pyfunc_loaded = mlflow.pyfunc.load_pyfunc(path)
        assert np.all(pyfunc_loaded.predict(x).values[:, 0] == predicted)
