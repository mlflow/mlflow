from pathlib import Path
import logging
import yaml

import torch
import pandas as pd
import numpy as np
import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.exceptions import MlflowException
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "torchscript"
_SERIALIZED_TORCH_MODEL_FILE_NAME = "model.pt"
_logger = logging.getLogger(__name__)


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import torch

    return _mlflow_conda_env(
        additional_conda_deps=[
            "pytorch={}".format(torch.__version__),
        ],
        additional_conda_channels=[
            "pytorch",
        ])


def log_model(model, artifact_path, conda_env=None, registered_model_name=None,
              signature: ModelSignature = None, input_example: ModelInputExample = None,
              **kwargs):
    Model.log(artifact_path=artifact_path, flavor=mlflow.torchscript, model=model,
              conda_env=conda_env, registered_model_name=registered_model_name,
              signature=signature, input_example=input_example, **kwargs)


def save_model(model, path, conda_env=None, mlflow_model=None,
               signature: ModelSignature = None, input_example: ModelInputExample = None,
               **kwargs):
    if not hasattr(model, 'state_dict') or not hasattr(model, 'save'):
        # If it walks like a duck and it quacks like a duck, then it must be a duck
        raise TypeError("Argument 'model' should be a TorchScript model")
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RuntimeError("Path '{}' already exists".format(path))

    if mlflow_model is None:
        mlflow_model = Model()

    if signature is not None:
        mlflow_model.signature = signature

    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    filepath = str(path / _SERIALIZED_TORCH_MODEL_FILE_NAME)
    model.save(filepath, **kwargs)
    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(path / conda_env_subpath, "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    mlflow_model.add_flavor(FLAVOR_NAME, pytorch_version=torch.__version__)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.torchscript", env=conda_env_subpath)
    mlflow_model.save(path / "MLmodel")


def load_model(model_uri, **kwargs):
    local_model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
    torchscript_conf = _get_flavor_configuration(model_path=local_model_path,
                                             flavor_name=FLAVOR_NAME)
    if torch.__version__ != torchscript_conf["pytorch_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed PyTorch version '%s'",
            torchscript_conf["pytorch_version"], torch.__version__)
    return torch.jit.load(
        str(local_model_path / _SERIALIZED_TORCH_MODEL_FILE_NAME), **kwargs)


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``torchscript`` flavor.
    """
    loaded = load_model(path, **kwargs)
    return _PyTorchWrapper(loaded)


class _PyTorchWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """
    def __init__(self, torchscript_model):
        self.torchscript_model = torchscript_model

    def predict(self, data, device='cpu'):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data should be pandas.DataFrame")
        self.torchscript_model.to(device)
        self.torchscript_model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(data.values.astype(np.float32)).to(device)
            preds = self.torchscript_model(input_tensor)
            if not isinstance(preds, torch.Tensor):
                raise TypeError("Expected PyTorch model to output a single output tensor, "
                                "but got output of type '{}'".format(type(preds)))
            predicted = pd.DataFrame(preds.numpy())
            predicted.index = data.index
            return predicted
