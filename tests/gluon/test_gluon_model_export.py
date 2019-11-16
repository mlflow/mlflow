import os
import warnings

import mxnet as mx
import mxnet.ndarray as nd
import pytest
from mxnet import context as ctx
from mxnet.gluon import Trainer
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data import DataLoader
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Dense
from mxnet.metric import Accuracy

import mlflow
import mlflow.gluon
from mlflow import pyfunc


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.mark.large
def test_model_save_load(model_path):
    mnist = mx.test_utils.get_mnist()
    train_data = nd.array(mnist["train_data"])
    train_label = nd.array(mnist["train_label"])
    test_data = nd.array(mnist["test_data"])
    train_data_loader = DataLoader(list(zip(train_data, train_label)),
                                   batch_size=128, last_batch="discard")

    model = HybridSequential()
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10))
    model.initialize()
    model.hybridize()
    trainer = Trainer(model.collect_params(), "adam",
                      optimizer_params={"learning_rate": .001, "epsilon": 1e-07})
    est = estimator.Estimator(net=model, loss=SoftmaxCrossEntropyLoss(),
                              metrics=Accuracy(), trainer=trainer)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(train_data_loader, epochs=3)

    expected = nd.argmax(model(test_data), axis=1)
    mlflow.gluon.save_model(model, model_path)
    # Loading Keras model
    model_loaded = mlflow.gluon.load_model(model_path, ctx.cpu())
    actual = nd.argmax(model_loaded(test_data), axis=1)
    assert all(expected == actual)
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    assert all(nd.argmax(pyfunc_loaded.predict(test_data), axis=1) == expected)
