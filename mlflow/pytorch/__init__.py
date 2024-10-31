"""
The ``mlflow.pytorch`` module provides an API for logging and loading PyTorch models. This module
exports PyTorch models with the following flavors:

PyTorch (native) format
    This is the main flavor that can be loaded back into PyTorch.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

import atexit
import importlib
import logging
import os
import posixpath
import shutil
import warnings
from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.checkpoint_utils import download_checkpoint_artifact
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import (
    TempDir,
    get_total_file_size,
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "pytorch"

_SERIALIZED_TORCH_MODEL_FILE_NAME = "model.pth"
_TORCH_STATE_DICT_FILE_NAME = "state_dict.pth"
_PICKLE_MODULE_INFO_FILE_NAME = "pickle_module_info.txt"
_EXTRA_FILES_KEY = "extra_files"
_REQUIREMENTS_FILE_KEY = "requirements_file"
_TORCH_CPU_DEVICE_NAME = "cpu"
_TORCH_DEFAULT_GPU_DEVICE_NAME = "cuda"

_logger = logging.getLogger(__name__)

MIN_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["pytorch-lightning"]["autologging"]["minimum"])
MAX_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["pytorch-lightning"]["autologging"]["maximum"])


_MODEL_DATA_SUBPATH = "data"


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor. Calls to
        :func:`save_model()` and :func:`log_model()` produce a pip environment that, at minimum,
        contains these requirements.
    """
    return list(
        map(
            _get_pinned_requirement,
            [
                "torch",
                # We include CloudPickle in the default environment because
                # it's required by the default pickle module used by `save_model()`
                # and `log_model()`: `mlflow.pytorch.pickle_module`.
                "cloudpickle",
            ],
        )
    )


def get_default_conda_env():
    """
    Returns:
        The default Conda environment as a dictionary for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Log PyTorch model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model", signature=signature)

        # Fetch the associated conda environment
        env = mlflow.pytorch.get_default_conda_env()
        print(f"conda env: {env}")

    .. code-block:: text
        :caption: Output

        conda env {'name': 'mlflow-env',
                   'channels': ['conda-forge'],
                   'dependencies': ['python=3.8.15',
                                    {'pip': ['torch==1.5.1',
                                             'mlflow',
                                             'cloudpickle==1.6.0']}]}
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="torch"))
def log_model(
    pytorch_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    pickle_module=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Log a PyTorch model as an MLflow artifact for the current run.

    .. warning:: Log the model with a signature to avoid inference errors.
        If the model is logged without a signature, the MLflow Model Server relies on the
        default inferred data type from NumPy. However, PyTorch often expects different
        defaults, particularly when parsing floats. You must include the signature to ensure
        that the model is logged with the correct data type so that the MLflow model server
        can correctly provide valid input.

    Args:
        pytorch_model: PyTorch model to be saved. Can be either an eager model (subclass of
            ``torch.nn.Module``) or scripted model prepared via ``torch.jit.script`` or
            ``torch.jit.trace``.

            The model accept a single ``torch.FloatTensor`` as input and produce a single output
            tensor.

            If saving an eager model, any code dependencies of the model's class, including the
            class definition itself, should be included in one of the following locations:

                - The package(s) listed in the model's Conda environment, specified by the
                  ``conda_env`` parameter.
                - One or more of the files specified by the ``code_paths`` parameter.

        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        pickle_module: The module that PyTorch should use to serialize ("pickle") the specified
            ``pytorch_model``. This is passed as the ``pickle_module`` parameter to
            ``torch.save()``.  By default, this module is also used to deserialize ("unpickle") the
            PyTorch model at load time.
        registered_model_name: If given, create a model version under ``registered_model_name``,
            also create a registered model if one with the given name does not exist.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function waits for five
            minutes.  Specify 0 or None to skip waiting.

        requirements_file:

            .. warning::

                ``requirements_file`` has been deprecated. Please use ``pip_requirements`` instead.

            A string containing the path to requirements file. Remote URIs are resolved to absolute
            filesystem paths. For example, consider the following ``requirements_file`` string:

            .. code-block:: python

                requirements_file = "s3://my-bucket/path/to/my_file"

            In this case, the ``"my_file"`` requirements file is downloaded from S3. If ``None``,
            no requirements file is added to the model.

        extra_files: A list containing the paths to corresponding extra files, if ``None``, no
            extra files are added to the model. Remote URIs are resolved to absolute filesystem
            paths. For example, consider the following ``extra_files`` list:

            .. code-block:: python

                extra_files = ["s3://my-bucket/path/to/my_file1", "s3://my-bucket/path/to/my_file2"]

            In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        kwargs: kwargs to pass to ``torch.save`` method.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import numpy as np
        import torch
        import mlflow
        from mlflow import MlflowClient
        from mlflow.models import infer_signature

        # Define model, loss, and optimizer
        model = nn.Linear(1, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Create training data with relationship y = 2X
        X = torch.arange(1.0, 26.0).reshape(-1, 1)
        y = X * 2

        # Training loop
        epochs = 250
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing X to the model
            y_pred = model(X)

            # Compute the loss
            loss = criterion(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create model signature
        signature = infer_signature(X.numpy(), model(X).detach().numpy())

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

            # convert to scripted model and log the model
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

        # Fetch the logged model artifacts
        print(f"run_id: {run.info.run_id}")
        for artifact_path in ["model/data", "scripted_model/data"]:
            artifacts = [
                f.path for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)
            ]
            print(f"artifacts: {artifacts}")

    .. code-block:: text
        :caption: Output

        run_id: 1a1ec9e413ce48e9abf9aec20efd6f71
        artifacts: ['model/data/model.pth',
                    'model/data/pickle_module_info.txt']
        artifacts: ['scripted_model/data/model.pth',
                    'scripted_model/data/pickle_module_info.txt']

    .. figure:: ../_static/images/pytorch_logged_models.png

        PyTorch logged models
    """
    pickle_module = pickle_module or mlflow_pytorch_pickle_module
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.pytorch,
        pytorch_model=pytorch_model,
        conda_env=conda_env,
        code_paths=code_paths,
        pickle_module=pickle_module,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        requirements_file=requirements_file,
        extra_files=extra_files,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="torch"))
def save_model(
    pytorch_model,
    path,
    conda_env=None,
    mlflow_model=None,
    code_paths=None,
    pickle_module=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Save a PyTorch model to a path on the local file system.

    Args:
        pytorch_model: PyTorch model to be saved. Can be either an eager model (subclass of
            ``torch.nn.Module``) or a scripted model prepared via ``torch.jit.script`` or
            ``torch.jit.trace``.

            To save an eager model, any code dependencies of the model's class, including the class
            definition itself, should be included in one of the following locations:

                - The package(s) listed in the model's Conda environment, specified by the
                  ``conda_env`` parameter.
                - One or more of the files specified by the ``code_paths`` parameter.

        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        code_paths: {{ code_paths }}
        pickle_module: The module that PyTorch should use to serialize ("pickle") the specified
            ``pytorch_model``. This is passed as the ``pickle_module`` parameter to
            ``torch.save()``. By default, this module is also used to deserialize ("unpickle") the
            model at loading time.
        signature: {{ signature }}
        input_example: {{ input_example }}
        requirements_file:

            .. warning::

                ``requirements_file`` has been deprecated. Please use ``pip_requirements`` instead.

            A string containing the path to requirements file. Remote URIs are resolved to absolute
            filesystem paths. For example, consider the following ``requirements_file`` string:

            .. code-block:: python

                requirements_file = "s3://my-bucket/path/to/my_file"

            In this case, the ``"my_file"`` requirements file is downloaded from S3. If ``None``,
            no requirements file is added to the model.

        extra_files: A list containing the paths to corresponding extra files. Remote URIs
            are resolved to absolute filesystem paths.
            For example, consider the following ``extra_files`` list -

            extra_files = ["s3://my-bucket/path/to/my_file1", "s3://my-bucket/path/to/my_file2"]

            In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

            If ``None``, no extra files are added to the model.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata:{{ metadata }}
        kwargs: kwargs to pass to ``torch.save`` method.

    .. code-block:: python
        :caption: Example

        import os
        import mlflow
        import torch


        model = nn.Linear(1, 1)

        # Save PyTorch models to current working directory
        with mlflow.start_run() as run:
            mlflow.pytorch.save_model(model, "model")

            # Convert to a scripted model and save it
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.save_model(scripted_pytorch_model, "scripted_model")

        # Load each saved model for inference
        for model_path in ["model", "scripted_model"]:
            model_uri = f"{os.getcwd()}/{model_path}"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            print(f"Loaded {model_path}:")
            for x in [6.0, 8.0, 12.0, 30.0]:
                X = torch.Tensor([[x]])
                y_pred = loaded_model(X)
                print(f"predict X: {x}, y_pred: {y_pred.data.item():.2f}")
            print("--")

    .. code-block:: text
        :caption: Output

        Loaded model:
        predict X: 6.0, y_pred: 11.90
        predict X: 8.0, y_pred: 15.92
        predict X: 12.0, y_pred: 23.96
        predict X: 30.0, y_pred: 60.13
        --
        Loaded scripted_model:
        predict X: 6.0, y_pred: 11.90
        predict X: 8.0, y_pred: 15.92
        predict X: 12.0, y_pred: 23.96
        predict X: 30.0, y_pred: 60.13

    """
    import torch

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    pickle_module = pickle_module or mlflow_pytorch_pickle_module

    if not isinstance(pytorch_model, torch.nn.Module):
        raise TypeError("Argument 'pytorch_model' should be a torch.nn.Module")
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = _PyTorchWrapper(pytorch_model, device="cpu")
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    model_data_subpath = _MODEL_DATA_SUBPATH
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)

    # Persist the pickle module name as a file in the model's `data` directory. This is necessary
    # because the `data` directory is the only available parameter to `_load_pyfunc`, and it
    # does not contain the MLmodel configuration; therefore, it is not sufficient to place
    # the module name in the MLmodel
    #
    # TODO: Stop persisting this information to the filesystem once we have a mechanism for
    # supplying the MLmodel configuration to `mlflow.pytorch._load_pyfunc`
    pickle_module_path = os.path.join(model_data_path, _PICKLE_MODULE_INFO_FILE_NAME)
    with open(pickle_module_path, "w") as f:
        f.write(pickle_module.__name__)
    # Save pytorch model
    model_path = os.path.join(model_data_path, _SERIALIZED_TORCH_MODEL_FILE_NAME)
    if isinstance(pytorch_model, torch.jit.ScriptModule):
        torch.jit.ScriptModule.save(pytorch_model, model_path)
    else:
        torch.save(pytorch_model, model_path, pickle_module=pickle_module, **kwargs)

    torchserve_artifacts_config = {}

    if extra_files:
        torchserve_artifacts_config[_EXTRA_FILES_KEY] = []
        if not isinstance(extra_files, list):
            raise TypeError("Extra files argument should be a list")

        with TempDir() as tmp_extra_files_dir:
            for extra_file in extra_files:
                _download_artifact_from_uri(
                    artifact_uri=extra_file, output_path=tmp_extra_files_dir.path()
                )
                rel_path = posixpath.join(_EXTRA_FILES_KEY, os.path.basename(extra_file))
                torchserve_artifacts_config[_EXTRA_FILES_KEY].append({"path": rel_path})
            shutil.move(
                tmp_extra_files_dir.path(),
                posixpath.join(path, _EXTRA_FILES_KEY),
            )

    if requirements_file:
        warnings.warn(
            "`requirements_file` has been deprecated. Please use `pip_requirements` instead.",
            FutureWarning,
            stacklevel=2,
        )

        if not isinstance(requirements_file, str):
            raise TypeError("Path to requirements file should be a string")

        with TempDir() as tmp_requirements_dir:
            _download_artifact_from_uri(
                artifact_uri=requirements_file, output_path=tmp_requirements_dir.path()
            )
            rel_path = os.path.basename(requirements_file)
            torchserve_artifacts_config[_REQUIREMENTS_FILE_KEY] = {"path": rel_path}
            shutil.move(tmp_requirements_dir.path(rel_path), path)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_data=model_data_subpath,
        pytorch_version=str(torch.__version__),
        code=code_dir_subpath,
        **torchserve_artifacts_config,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.pytorch",
        data=model_data_subpath,
        pickle_module_name=pickle_module.__name__,
        code=code_dir_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        model_config={"device": None},
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    if not requirements_file:
        # Save `requirements.txt`
        write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _load_model(path, device=None, **kwargs):
    """
    Args:
        path: The path to a serialized PyTorch model.
        device: If specified, load the model on the specified device.
        kwargs: Additional kwargs to pass to the PyTorch ``torch.load`` function.
    """
    import torch

    if os.path.isdir(path):
        # `path` is a directory containing a serialized PyTorch model and a text file containing
        # information about the pickle module that should be used by PyTorch to load it
        model_path = os.path.join(path, "model.pth")
        pickle_module_path = os.path.join(path, _PICKLE_MODULE_INFO_FILE_NAME)
        with open(pickle_module_path) as f:
            pickle_module_name = f.read()
        if "pickle_module" in kwargs and kwargs["pickle_module"].__name__ != pickle_module_name:
            _logger.warning(
                "Attempting to load the PyTorch model with a pickle module, '%s', that does not"
                " match the pickle module that was used to save the model: '%s'.",
                kwargs["pickle_module"].__name__,
                pickle_module_name,
            )
        else:
            try:
                kwargs["pickle_module"] = importlib.import_module(pickle_module_name)
            except ImportError as exc:
                raise MlflowException(
                    message=(
                        "Failed to import the pickle module that was used to save the PyTorch"
                        f" model. Pickle module name: `{pickle_module_name}`"
                    ),
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from exc

    else:
        model_path = path

    if Version(torch.__version__) >= Version("1.5.0"):
        pytorch_model = torch.load(model_path, **kwargs)
    else:
        try:
            # load the model as an eager model.
            pytorch_model = torch.load(model_path, **kwargs)
        except Exception:
            # If fails, assume the model as a scripted model
            # `torch.jit.load` does not accept `pickle_module`.
            kwargs.pop("pickle_module", None)
            pytorch_model = torch.jit.load(model_path, **kwargs)

    pytorch_model.eval()
    if device:
        pytorch_model.to(device=device)
    return pytorch_model


def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load a PyTorch model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model, for example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see `Referencing Artifacts \
            <https://www.mlflow.org/docs/latest/concepts.html#artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output path will be created.
        kwargs: kwargs to pass to ``torch.load`` method.

    Returns:
        A PyTorch model.

    .. code-block:: python
        :caption: Example

        import torch
        import mlflow.pytorch


        model = nn.Linear(1, 1)

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

        # Inference after loading the logged model
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        for x in [4.0, 6.0, 30.0]:
            X = torch.Tensor([[x]])
            y_pred = loaded_model(X)
            print(f"predict X: {x}, y_pred: {y_pred.data.item():.2f}")

    .. code-block:: text
        :caption: Output

        predict X: 4.0, y_pred: 7.57
        predict X: 6.0, y_pred: 11.64
        predict X: 30.0, y_pred: 60.48
    """
    import torch

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    pytorch_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, pytorch_conf)

    if torch.__version__ != pytorch_conf["pytorch_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed PyTorch version '%s'",
            pytorch_conf["pytorch_version"],
            torch.__version__,
        )
    torch_model_artifacts_path = os.path.join(local_model_path, pytorch_conf["model_data"])
    return _load_model(path=torch_model_artifacts_path, **kwargs)


def _load_pyfunc(path, model_config=None):  # noqa: D417
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``pytorch`` flavor.
    """
    import torch

    device = model_config.get("device", None) if model_config else None
    # if CUDA is available, we use the default CUDA device.
    # To force inference to the CPU when the GPU is available, please set
    # MLFLOW_DEFAULT_PREDICTION_DEVICE to "cpu"
    # If a specific non-default device is passed in, we continue to respect that.
    if device is None:
        if MLFLOW_DEFAULT_PREDICTION_DEVICE.get():
            device = MLFLOW_DEFAULT_PREDICTION_DEVICE.get()
        elif torch.cuda.is_available():
            device = _TORCH_DEFAULT_GPU_DEVICE_NAME
        else:
            device = _TORCH_CPU_DEVICE_NAME

    return _PyTorchWrapper(_load_model(path, device=device), device=device)


class _PyTorchWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pytorch_model, device):
        self.pytorch_model = pytorch_model
        self.device = device

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.pytorch_model

    def predict(self, data, params: Optional[dict[str, Any]] = None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        import torch

        if params and "device" in params:
            raise ValueError(
                "device' can no longer be specified as an inference parameter. "
                "It must be specified at load time. "
                "Please specify the device at load time, for example: "
                "`mlflow.pyfunc.load_model(model_uri, model_config={'device': 'cuda'})`."
            )

        if isinstance(data, pd.DataFrame):
            inp_data = data.values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, (list, dict)):
            raise TypeError(
                "The PyTorch flavor does not support List or Dict input types. "
                "Please use a pandas.DataFrame or a numpy.ndarray"
            )
        else:
            raise TypeError("Input data should be pandas.DataFrame or numpy.ndarray")

        device = self.device
        with torch.no_grad():
            input_tensor = torch.from_numpy(inp_data).to(device)
            preds = self.pytorch_model(input_tensor)
            # if the predictions happened on a remote device, copy them back to
            # the host CPU for processing
            if device != _TORCH_CPU_DEVICE_NAME:
                preds = preds.to(_TORCH_CPU_DEVICE_NAME)
            if not isinstance(preds, torch.Tensor):
                raise TypeError(
                    "Expected PyTorch model to output a single output tensor, "
                    f"but got output of type '{type(preds)}'"
                )
            if isinstance(data, pd.DataFrame):
                predicted = pd.DataFrame(preds.numpy())
                predicted.index = data.index
            else:
                predicted = preds.numpy()
            return predicted


def log_state_dict(state_dict, artifact_path, **kwargs):
    """
    Log a state_dict as an MLflow artifact for the current run.

    .. warning::
        This function just logs a state_dict as an artifact and doesn't generate
        an :ref:`MLflow Model <models>`.

    Args:
        state_dict: state_dict to be saved.
        artifact_path: Run-relative artifact path.
        kwargs: kwargs to pass to ``torch.save``.

    .. code-block:: python
        :caption: Example

        # Log a model as a state_dict
        with mlflow.start_run():
            state_dict = model.state_dict()
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")

        # Log a checkpoint as a state_dict
        with mlflow.start_run():
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
            }
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")
    """

    with TempDir() as tmp:
        local_path = tmp.path()
        save_state_dict(state_dict=state_dict, path=local_path, **kwargs)
        mlflow.log_artifacts(local_path, artifact_path)


def save_state_dict(state_dict, path, **kwargs):
    """
    Save a state_dict to a path on the local file system

    Args:
        state_dict: state_dict to be saved.
        path: Local path where the state_dict is to be saved.
        kwargs: kwargs to pass to ``torch.save``.
    """
    import torch

    # The object type check here aims to prevent a scenario where a user accidentally passees
    # a model instead of a state_dict and `torch.save` (which accepts both model and state_dict)
    # successfully completes, leaving the user unaware of the mistake.
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Invalid object type for `state_dict`: {}. Must be an instance of `dict`".format(
                type(state_dict)
            )
        )

    os.makedirs(path, exist_ok=True)
    state_dict_path = os.path.join(path, _TORCH_STATE_DICT_FILE_NAME)
    torch.save(state_dict, state_dict_path, **kwargs)


def load_state_dict(state_dict_uri, **kwargs):
    """
    Load a state_dict from a local file or a run.

    Args:
        state_dict_uri: The location, in URI format, of the state_dict, for example:

            - ``/Users/me/path/to/local/state_dict``
            - ``relative/path/to/local/state_dict``
            - ``s3://my_bucket/path/to/state_dict``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/state_dict``

            For more information about supported URI schemes, see `Referencing Artifacts \
            <https://www.mlflow.org/docs/latest/concepts.html#artifact-locations>`_.

        kwargs: kwargs to pass to ``torch.load``.

    Returns:
        A state_dict

    .. code-block:: python
        :caption: Example

        with mlflow.start_run():
            artifact_path = "model"
            mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path)
            state_dict_uri = mlflow.get_artifact_uri(artifact_path)

        state_dict = mlflow.pytorch.load_state_dict(state_dict_uri)
    """
    import torch

    local_path = _download_artifact_from_uri(artifact_uri=state_dict_uri)
    state_dict_path = os.path.join(local_path, _TORCH_STATE_DICT_FILE_NAME)
    return torch.load(state_dict_path, **kwargs)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_every_n_epoch=1,
    log_every_n_step=None,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
    checkpoint=True,
    checkpoint_monitor="val_loss",
    checkpoint_mode="min",
    checkpoint_save_best_only=True,
    checkpoint_save_weights_only=False,
    checkpoint_save_freq="epoch",
):
    """
    Enables (or disables) and configures autologging from `PyTorch Lightning
    <https://pytorch-lightning.readthedocs.io/en/latest>`_ to MLflow.

    Autologging is performed when you call the `fit` method of
    `pytorch_lightning.Trainer() \
    <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#>`_.

    Explore the complete `PyTorch MNIST \
    <https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST>`_ for
    an expansive example with implementation of additional lightening steps.

    **Note**: Full autologging is only supported for PyTorch Lightning models,
    i.e., models that subclass
    `pytorch_lightning.LightningModule \
    <https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html>`_.
    Autologging support for vanilla PyTorch (ie models that only subclass
    `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_)
    only autologs calls to
    `torch.utils.tensorboard.SummaryWriter <https://pytorch.org/docs/stable/tensorboard.html>`_'s
    ``add_scalar`` and ``add_hparams`` methods to mlflow. In this case, there's also
    no notion of an "epoch".

    Args:
        log_every_n_epoch: If specified, logs metrics once every `n` epochs. By default, metrics
            are logged after every epoch.
        log_every_n_step: If specified, logs batch metrics once every `n` training step.
            By default, metrics are not logged for steps. Note that setting this to 1 can cause
            performance issues and is not recommended. Metrics are logged against Lightning's global
            step number, and when multiple optimizers are used it is assumed that all optimizers
            are stepped in each training step.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
        log_datasets: If ``True``, dataset information is logged to MLflow Tracking.
            If ``False``, dataset information is not logged.
        disable: If ``True``, disables the PyTorch Lightning autologging integration.
            If ``False``, enables the PyTorch Lightning autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run, which may be
            user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            pytorch and pytorch-lightning that have not been tested against this version
            of the MLflow client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during PyTorch
            Lightning autologging. If ``False``, show all events and warnings during PyTorch
            Lightning autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name. The registered model is
            created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
        checkpoint: Enable automatic model checkpointing, this feature only supports
            pytorch-lightning >= 1.6.0.
        checkpoint_monitor: In automatic model checkpointing, the metric name to monitor if
            you set `model_checkpoint_save_best_only` to True.
        checkpoint_mode: one of {"min", "max"}. In automatic model checkpointing,
            if save_best_only=True, the decision to overwrite the current save file is made based on
            either the maximization or the minimization of the monitored quantity.
        checkpoint_save_best_only: If True, automatic model checkpointing only saves when
            the model is considered the "best" model according to the quantity
            monitored and previous checkpoint model is overwritten.
        checkpoint_save_weights_only: In automatic model checkpointing, if True, then
            only the model’s weights will be saved. Otherwise, the optimizer states,
            lr-scheduler states, etc are added in the checkpoint too.
        checkpoint_save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. Note that if the saving isn't aligned to
            epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.

    .. code-block:: python
        :test:
        :caption: Example

        import os

        import lightning as L
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, Subset
        from torchmetrics import Accuracy
        from torchvision import transforms
        from torchvision.datasets import MNIST

        import mlflow.pytorch
        from mlflow import MlflowClient


        class MNISTModel(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)
                self.accuracy = Accuracy("multiclass", num_classes=10)

            def forward(self, x):
                return torch.relu(self.l1(x.view(x.size(0), -1)))

            def training_step(self, batch, batch_nb):
                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(dim=1)
                acc = self.accuracy(pred, y)

                # PyTorch `self.log` will be automatically captured by MLflow.
                self.log("train_loss", loss, on_epoch=True)
                self.log("acc", acc, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.02)


        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print(f"run_id: {r.info.run_id}")
            print(f"artifacts: {artifacts}")
            print(f"params: {r.data.params}")
            print(f"metrics: {r.data.metrics}")
            print(f"tags: {tags}")


        # Initialize our model.
        mnist_model = MNISTModel()

        # Load MNIST dataset.
        train_ds = MNIST(
            os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
        )
        # Only take a subset of the data for faster training.
        indices = torch.arange(32)
        train_ds = Subset(train_ds, indices)
        train_loader = DataLoader(train_ds, batch_size=8)

        # Initialize a trainer.
        trainer = L.Trainer(max_epochs=3)

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

        # Train the model.
        with mlflow.start_run() as run:
            trainer.fit(mnist_model, train_loader)

        # Fetch the auto logged parameters and metrics.
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    """
    try:
        import pytorch_lightning as pl
    except ImportError:
        pass
    else:
        from mlflow.pytorch._lightning_autolog import patched_fit

        safe_patch(
            FLAVOR_NAME, pl.Trainer, "fit", patched_fit, manage_run=True, extra_tags=extra_tags
        )

    try:
        import lightning as L
    except ImportError:
        pass
    else:
        from mlflow.pytorch._lightning_autolog import patched_fit

        safe_patch(
            FLAVOR_NAME, L.Trainer, "fit", patched_fit, manage_run=True, extra_tags=extra_tags
        )

    try:
        import torch.utils.tensorboard.writer
    except ImportError:
        pass
    else:
        from mlflow.pytorch._pytorch_autolog import (
            flush_metrics_queue,
            patched_add_event,
            patched_add_hparams,
            patched_add_summary,
        )

        safe_patch(
            FLAVOR_NAME,
            torch.utils.tensorboard.writer.FileWriter,
            "add_event",
            partial(patched_add_event, mlflow_log_every_n_step=log_every_n_step),
            manage_run=True,
            extra_tags=extra_tags,
        )
        safe_patch(
            FLAVOR_NAME,
            torch.utils.tensorboard.writer.FileWriter,
            "add_summary",
            patched_add_summary,
            manage_run=True,
            extra_tags=extra_tags,
        )
        safe_patch(
            FLAVOR_NAME,
            torch.utils.tensorboard.SummaryWriter,
            "add_hparams",
            patched_add_hparams,
            manage_run=True,
            extra_tags=extra_tags,
        )

        atexit.register(flush_metrics_queue)


if autolog.__doc__ is not None:
    autolog.__doc__ = autolog.__doc__.replace("MIN_REQ_VERSION", str(MIN_REQ_VERSION)).replace(
        "MAX_REQ_VERSION", str(MAX_REQ_VERSION)
    )


def load_checkpoint(model_class, run_id=None, epoch=None, global_step=None, kwargs=None):
    """
    If you enable "checkpoint" in autologging, during pytorch-lightning model
    training execution, checkpointed models are logged as MLflow artifacts.
    Using this API, you can load the checkpointed model.

    If you want to load the latest checkpoint, set both `epoch` and `global_step` to None.
    If "checkpoint_save_freq" is set to "epoch" in autologging,
    you can set `epoch` param to the epoch of the checkpoint to load specific epoch checkpoint.
    If "checkpoint_save_freq" is set to an integer in autologging,
    you can set `global_step` param to the global step of the checkpoint to load specific
    global step checkpoint.
    `epoch` param and `global_step` can't be set together.

    Args:
        model_class: The class of the training model, the class should inherit
            'pytorch_lightning.LightningModule'.
        run_id: The id of the run which model is logged to. If not provided,
            current active run is used.
        epoch: The epoch of the checkpoint to be loaded, if you set
            "checkpoint_save_freq" to "epoch".
        global_step: The global step of the checkpoint to be loaded, if
            you set "checkpoint_save_freq" to an integer.
        kwargs: Any extra kwargs needed to init the model.

    Returns:
        The instance of a pytorch-lightning model restored from the specified checkpoint.

    .. code-block:: python
        :caption: Example

        import mlflow

        mlflow.pytorch.autolog(checkpoint=True)

        model = MyLightningModuleNet()  # A custom-pytorch lightning model
        train_loader = create_train_dataset_loader()
        trainer = Trainer()

        with mlflow.start_run() as run:
            trainer.fit(model, train_loader)

        run_id = run.info.run_id

        # load latest checkpoint model
        latest_checkpoint_model = mlflow.pytorch.load_checkpoint(MyLightningModuleNet, run_id)

        # load history checkpoint model logged in second epoch
        checkpoint_model = mlflow.pytorch.load_checkpoint(MyLightningModuleNet, run_id, epoch=2)
    """
    with TempDir() as tmp_dir:
        downloaded_checkpoint_filepath = download_checkpoint_artifact(
            run_id=run_id, epoch=epoch, global_step=global_step, dst_path=tmp_dir.path()
        )
        return model_class.load_from_checkpoint(downloaded_checkpoint_filepath, **(kwargs or {}))


__all__ = [
    "autolog",
    "load_model",
    "save_model",
    "log_model",
    "get_default_pip_requirements",
    "get_default_conda_env",
    "load_checkpoint",
]

try:
    from mlflow.pytorch._lightning_autolog import MlflowModelCheckpointCallback  # noqa: F401

    __all__.append("MLflowModelCheckpointCallback")
except ImportError:
    # Swallow exception if pytorch-lightning is not installed.
    pass
