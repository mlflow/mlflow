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
    import torchvision

    return _mlflow_conda_env(
        additional_conda_deps=[
            "pytorch={}".format(torch.__version__),
            "torchvision={}".format(torchvision.__version__),
        ],
        additional_conda_channels=[
            "pytorch",
        ])


def log_model(model, artifact_path, conda_env=None, registered_model_name=None,
              signature: ModelSignature = None, input_example: ModelInputExample = None,
              **kwargs):
    """Log a torchscript model as an MLflow artifact for the current run.

    :param model: Torchscript model to be saved. Must accept a single ``torch.FloatTensor`` as
                  input and produce a single output tensor.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model. The
                      following is an *example* dictionary representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults', 'pytorch'],
                            'dependencies': [
                                'python=3.7.0',
                                'pytorch=0.4.1',
                                'torchvision=0.2.1'
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.


    :param kwargs: kwargs to pass to ``torch.jit.save`` method.

    .. code-block:: python
        :caption: Example

        import torch
        import mlflow
        import mlflow.torchscript
        # X data
        x_data = torch.Tensor([[1.0], [2.0], [3.0]])
        # Y data with its expected value: labels
        y_data = torch.Tensor([[2.0], [4.0], [6.0]])
        # Partial Model example modified from Sung Kim
        # https://github.com/hunkim/PyTorchZeroToAll
        class Model(torch.nn.Module):
            def __init__(self):
               super(Model, self).__init__()
               self.linear = torch.nn.Linear(1, 1)  # One in and one out
            def forward(self, x):
                y_pred = self.linear(x)
            return y_pred
        # our model
        model = Model()
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Training loop
        for epoch in range(500):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x_data)
            # Compute and print loss
            loss = criterion(y_pred, y_data)
            print(epoch, loss.data.item())
            #Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # After training
        for hv in [4.0, 5.0, 6.0]:
            hour_var = torch.Tensor([[hv]])
            y_pred = model(hour_var)
            print("predict (after training)",  hv, model(hour_var).data[0][0])
        # log the model
        with mlflow.start_run() as run:
            mlflow.log_param("epochs", 500)
            scripted_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_model, "models")
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.torchscript, model=model,
              conda_env=conda_env, registered_model_name=registered_model_name,
              signature=signature, input_example=input_example, **kwargs)


def save_model(model, path, conda_env=None, mlflow_model=None,
               signature: ModelSignature = None, input_example: ModelInputExample = None,
               **kwargs):
    """
    Save a torchscript model to a path on the local file system.

    :param model: Torchscript model to be saved. Must accept a single ``torch.FloatTensor`` as
                  input and produce a single output tensor.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model. The
                      following is an *example* dictionary representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults', 'pytorch'],
                            'dependencies': [
                                'python=3.7.0',
                                'pytorch=1.5.0',
                                'torchvision=0.2.1'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    :param kwargs: kwargs to pass to ``torch.jit.save`` method.

    .. code-block:: python
        :caption: Example

        import torch
        import mlflow
        import mlflow.torchscript
        # Create model and set values
        model = Model()
        # train our model
        for epoch in range(500):
            y_pred = model(x_data)
            ...
        # Save the model
        with mlflow.start_run() as run:
            mlflow.log_param("epochs", 500)
            scripted_model = torch.jit.script(model)
            mlflow.torchscript.save_model(scripted_model, pytorch_model_path)
    """
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
    """
        Load a torchscript model from a local file or a run.

        :param model_uri: The location, in URI format, of the MLflow model, for example:

                          - ``/Users/me/path/to/local/model``
                          - ``relative/path/to/local/model``
                          - ``s3://my_bucket/path/to/model``
                          - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                          - ``models:/<model_name>/<model_version>``
                          - ``models:/<model_name>/<stage>``

                          For more information about supported URI schemes, see
                          `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                          artifact-locations>`_.

        :param kwargs: kwargs to pass to ``torch.jit.load`` method.
        :return: A torchscript model.

        .. code-block:: python
            :caption: Example

            import torch
            import mlflow
            import mlflow.torchscript
            # Set values
            model_path_dir = ...
            run_id = "96771d893a5e46159d9f3b49bf9013e2"
            model = mlflow.torchscript.load_model("runs:/" + run_id + "/" + model_path_dir)
            y_pred = model(x_new_data)
        """
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
