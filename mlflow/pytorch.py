"""
The ``mlflow.pytorch`` module provides an API for logging and loading PyTorch models. This module
exports PyTorch models with the following flavors:

PyTorch (native) format
    This is the main flavor that can be loaded back into PyTorch.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os

import numpy as np
import pandas as pd
import torch

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


FLAVOR_NAME = "pytorch"


def log_model(pytorch_model, artifact_path, conda_env=None, **kwargs):
    """
    Log a PyTorch model as an MLflow artifact for the current run.

    :param pytorch_model: PyTorch model to be saved. Must accept a single ``torch.FloatTensor`` as
                          input and produce a single output tensor.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, this defines the environment
           for the model. At minimum, it should specify python, pytorch, and mlflow with appropriate
           versions.
    :param kwargs: kwargs to pass to ``torch.save`` method.

    >>> import torch
    >>> from torch.autograd import Variable
    >>> import mlflow
    >>> import mlflow.pytorch
    >>> # X data
    >>> x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
    >>> # Y data with its expected value: labels
    >>> y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
    >>> # Partial Model example modified from Sung Kim
    >>> # https://github.com/hunkim/PyTorchZeroToAll
    >>> class Model(torch.nn.Module):
    >>>    def __init__(self):
    >>>       super(Model, self).__init__()
    >>>       self.linear = torch.nn.Linear(1, 1)  # One in and one out
    >>>    def forward(self, x):
    >>>        y_pred = self.linear(x)
    >>>        return y_pred
    >>> # our model
    >>> model = Model()
    >>> criterion = torch.nn.MSELoss(size_average=False)
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    >>> # Training loop
    >>> for epoch in range(500):
    >>>    # Forward pass: Compute predicted y by passing x to the model
    >>>    y_pred = model(x_data)
    >>>    # Compute and print loss
    >>>    loss = criterion(y_pred, y_data)
    >>>    print(epoch, loss.data[0])
    >>>   #Zero gradients, perform a backward pass, and update the weights.
    >>>   optimizer.zero_grad()
    >>>   loss.backward()
    >>>   optimizer.step()
    >>> # After training
    >>> for hv in [4.0, 5.0, 6.0]:
    >>>     hour_var = Variable(torch.Tensor([[hv]]))
    >>>     y_pred = model(hour_var)
    >>>     print("predict (after training)",  hv, model(hour_var ).data[0][0])
    >>> # log the model
    >>> with mlflow.start_run() as run:
    >>>   mlflow.log_param("epochs", 500)
    >>>   mlflow.pytorch.log_model(pytorch_model, "models")
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.pytorch,
              pytorch_model=pytorch_model, conda_env=conda_env, **kwargs)


def save_model(pytorch_model, path, conda_env=None, mlflow_model=Model(), **kwargs):
    """
    Save a PyTorch model to a path on the local file system.

    :param pytorch_model: PyTorch model to be saved. Must accept a single ``torch.FloatTensor`` as
                          input and produce a single output tensor.
    :param path: Local path where the model is to be saved.
    :param conda_env: Path to a Conda environment file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify python, pytorch,
                      and mlflow with appropriate versions.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param kwargs: kwargs to pass to ``torch.save`` method.

    >>> import torch
    >>> import mlflow
    >>> import mlflow.pytorch
    >>> # create model and set values
    >>> pytorch_model = Model()
    >>> pytorch_model_path = ...
    >>> #train our model
    >>> for epoch in range(500):
    >>>     y_pred = model(x_data)
    >>>     ...
    >>> #save the model
    >>> with mlflow.start_run() as run:
    >>>   mlflow.log_param("epochs", 500)
    >>>   mlflow.pytorch.save_model(pytorch_model, pytorch_model_path)
    """
    if not isinstance(pytorch_model, torch.nn.Module):
        raise TypeError("Argument 'pytorch_model' should be a torch.nn.Module")

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise RuntimeError("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_path = os.path.join(path, "model.pth")

    # Save pytorch model
    torch.save(pytorch_model, model_path, **kwargs)
    model_file = os.path.basename(model_path)

    mlflow_model.add_flavor(FLAVOR_NAME, model_data=model_file, pytorch_version=torch.__version__)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.pytorch",
                        data=model_file, env=conda_env)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _load_model(path, **kwargs):
    mlflow_model_path = os.path.join(path, "MLmodel")
    if not os.path.exists(mlflow_model_path):
        raise RuntimeError("MLmodel is not found at '{}'".format(path))

    mlflow_model = Model.load(mlflow_model_path)

    if FLAVOR_NAME not in mlflow_model.flavors:
        raise ValueError("Could not find flavor '{}' amongst available flavors {}, "
                         "unable to load stored model"
                         .format(FLAVOR_NAME, list(mlflow_model.flavors.keys())))

    # This maybe replaced by a warning and then try/except torch.load
    flavor = mlflow_model.flavors[FLAVOR_NAME]
    if torch.__version__ != flavor["pytorch_version"]:
        raise ValueError("Stored model version '{}' does not match "
                         "installed PyTorch version '{}'"
                         .format(flavor["pytorch_version"], torch.__version__))

    path = os.path.abspath(path)
    path = os.path.join(path, mlflow_model.flavors[FLAVOR_NAME]['model_data'])
    return torch.load(path, **kwargs)


def load_model(path, run_id=None, **kwargs):
    """
    Load a PyTorch model from a local file (if ``run_id`` is ``None``) or a run.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.pytorch.log_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    :param kwargs: kwargs to pass to ``torch.load`` method.

    >>> import torch
    >>> import mlflow
    >>> import mlflow.pytorch
    >>> # set values
    >>> model_path_dir = ...
    >>> run_id="96771d893a5e46159d9f3b49bf9013e2"
    >>> pytorch_model = mlflow.pytorch.load_model(model_path_dir, run_id)
    >>> y_pred = pytorch_model(x_new_data)
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)

    return _load_model(path, **kwargs)


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _PyTorchWrapper(_load_model(os.path.dirname(path), **kwargs))


class _PyTorchWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model

    def predict(self, data, device='cpu'):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data should be pandas.DataFrame")
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(data.values.astype(np.float32)).to(device)
            preds = self.pytorch_model(input_tensor)
            if not isinstance(preds, torch.Tensor):
                raise TypeError("Expected PyTorch model to output a single output tensor, "
                                "but got output of type '{}'".format(type(preds)))
            predicted = pd.DataFrame(preds.numpy())
            predicted.index = data.index
            return predicted
