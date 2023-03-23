"""
The ``mlflow.pytorch`` module provides an API for logging and loading PyTorch models. This module
exports PyTorch models with the following flavors:

PyTorch (native) format
    This is the main flavor that can be loaded back into PyTorch.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import importlib
import logging
import os
import yaml
import warnings

import numpy as np
import pandas as pd
from functools import partial
from packaging.version import Version
import posixpath

import mlflow
import shutil
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.file_utils import (
    TempDir,
    write_to,
)
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

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


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
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
    :return: The default Conda environment as a dictionary for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.

    .. code-block:: python
        :caption: Example

        import mlflow.pytorch

        # Log PyTorch model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

        # Fetch the associated conda environment
        env = mlflow.pytorch.get_default_conda_env()
        print("conda env: {}".format(env))

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

        .. warning::

            Log the model with a signature to avoid inference errors.
            If the model is logged without a signature, the MLflow Model Server relies on the
            default inferred data type from NumPy. However, PyTorch often expects different
            defaults, particularly when parsing floats. You must include the signature to ensure
            that the model is logged with the correct data type so that the MLflow model server
            can correctly provide valid input.

    :param pytorch_model: PyTorch model to be saved. Can be either an eager model (subclass of
                          ``torch.nn.Module``) or scripted model prepared via ``torch.jit.script``
                          or ``torch.jit.trace``.

                          The model accept a single ``torch.FloatTensor`` as
                          input and produce a single output tensor.

                          If saving an eager model, any code dependencies of the
                          model's class, including the class definition itself, should be
                          included in one of the following locations:

                          - The package(s) listed in the model's Conda environment, specified
                            by the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_paths`` parameter.

    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param pickle_module: The module that PyTorch should use to serialize ("pickle") the specified
                          ``pytorch_model``. This is passed as the ``pickle_module`` parameter
                          to ``torch.save()``. By default, this module is also used to
                          deserialize ("unpickle") the PyTorch model at load time.
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.

    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    :param requirements_file:

        .. warning::

            ``requirements_file`` has been deprecated. Please use ``pip_requirements`` instead.

        A string containing the path to requirements file. Remote URIs are resolved to absolute
        filesystem paths. For example, consider the following ``requirements_file`` string:

        .. code-block:: python

            requirements_file = "s3://my-bucket/path/to/my_file"

        In this case, the ``"my_file"`` requirements file is downloaded from S3. If ``None``,
        no requirements file is added to the model.

    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -

                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]

                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

                      If ``None``, no extra files are added to the model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param kwargs: kwargs to pass to ``torch.save`` method.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import numpy as np
        import torch
        import mlflow.pytorch


        class LinearNNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)  # One in and one out

            def forward(self, x):
                y_pred = self.linear(x)
                return y_pred


        def gen_data():
            # Example linear model modified to use y = 2x
            # from https://github.com/hunkim/PyTorchZeroToAll
            # X training data, y labels
            X = torch.arange(1.0, 25.0).view(-1, 1)
            y = torch.from_numpy(np.array([x * 2 for x in X])).view(-1, 1)
            return X, y


        # Define model, loss, and optimizer
        model = LinearNNModel()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Training loop
        epochs = 250
        X, y = gen_data()
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing X to the model
            y_pred = model(X)

            # Compute the loss
            loss = criterion(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

            # convert to scripted model and log the model
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

        # Fetch the logged model artifacts
        print("run_id: {}".format(run.info.run_id))
        for artifact_path in ["model/data", "scripted_model/data"]:
            artifacts = [
                f.path for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)
            ]
            print("artifacts: {}".format(artifacts))

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

    :param pytorch_model: PyTorch model to be saved. Can be either an eager model (subclass of
                          ``torch.nn.Module``) or scripted model prepared via ``torch.jit.script``
                          or ``torch.jit.trace``.

                          The model accept a single ``torch.FloatTensor`` as
                          input and produce a single output tensor.

                          If saving an eager model, any code dependencies of the
                          model's class, including the class definition itself, should be
                          included in one of the following locations:

                          - The package(s) listed in the model's Conda environment, specified
                            by the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_paths`` parameter.

    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param pickle_module: The module that PyTorch should use to serialize ("pickle") the specified
                          ``pytorch_model``. This is passed as the ``pickle_module`` parameter
                          to ``torch.save()``. By default, this module is also used to
                          deserialize ("unpickle") the PyTorch model at load time.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.

    :param requirements_file:

        .. warning::

            ``requirements_file`` has been deprecated. Please use ``pip_requirements`` instead.

        A string containing the path to requirements file. Remote URIs are resolved to absolute
        filesystem paths. For example, consider the following ``requirements_file`` string:

        .. code-block:: python

            requirements_file = "s3://my-bucket/path/to/my_file"

        In this case, the ``"my_file"`` requirements file is downloaded from S3. If ``None``,
        no requirements file is added to the model.

    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -

                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]

                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

                      If ``None``, no extra files are added to the model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param kwargs: kwargs to pass to ``torch.save`` method.

    .. code-block:: python
        :caption: Example

        import os

        import torch
        import mlflow.pytorch


        # Class defined here
        class LinearNNModel(torch.nn.Module):
            ...


        # Initialize our model, criterion and optimizer
        ...

        # Training loop
        ...

        # Save PyTorch models to current working directory
        with mlflow.start_run() as run:
            mlflow.pytorch.save_model(model, "model")

            # Convert to a scripted model and save it
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.save_model(scripted_pytorch_model, "scripted_model")

        # Load each saved model for inference
        for model_path in ["model", "scripted_model"]:
            model_uri = "{}/{}".format(os.getcwd(), model_path)
            loaded_model = mlflow.pytorch.load_model(model_uri)
            print("Loaded {}:".format(model_path))
            for x in [6.0, 8.0, 12.0, 30.0]:
                X = torch.Tensor([[x]])
                y_pred = loaded_model(X)
                print("predict X: {}, y_pred: {:.2f}".format(x, y_pred.data.item()))
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

    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    model_data_subpath = "data"
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
    )
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


def _load_model(path, **kwargs):
    """
    :param path: The path to a serialized PyTorch model.
    :param kwargs: Additional kwargs to pass to the PyTorch ``torch.load`` function.
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
        return torch.load(model_path, **kwargs)
    else:
        try:
            # load the model as an eager model.
            return torch.load(model_path, **kwargs)
        except Exception:
            # If fails, assume the model as a scripted model
            # `torch.jit.load` does not accept `pickle_module`.
            kwargs.pop("pickle_module", None)
            return torch.jit.load(model_path, **kwargs)


def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load a PyTorch model from a local file or a run.

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
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :param kwargs: kwargs to pass to ``torch.load`` method.
    :return: A PyTorch model.

    .. code-block:: python
        :caption: Example

        import torch
        import mlflow.pytorch


        # Class defined here
        class LinearNNModel(torch.nn.Module):
            ...


        # Initialize our model, criterion and optimizer
        ...

        # Training loop
        ...

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

        # Inference after loading the logged model
        model_uri = "runs:/{}/model".format(run.info.run_id)
        loaded_model = mlflow.pytorch.load_model(model_uri)
        for x in [4.0, 6.0, 30.0]:
            X = torch.Tensor([[x]])
            y_pred = loaded_model(X)
            print("predict X: {}, y_pred: {:.2f}".format(x, y_pred.data.item()))

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


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``pytorch`` flavor.
    """
    return _PyTorchWrapper(_load_model(path, **kwargs))


class _PyTorchWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model

    def predict(self, data, device=None):
        import torch

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

        self.pytorch_model.to(device)
        self.pytorch_model.eval()
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
                    "but got output of type '{}'".format(type(preds))
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

    :param state_dict: state_dict to be saved.
    :param artifact_path: Run-relative artifact path.
    :param kwargs: kwargs to pass to ``torch.save``.

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

    :param state_dict: state_dict to be saved.
    :param path: Local path where the state_dict is to be saved.
    :param kwargs: kwargs to pass to ``torch.save``.
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

    :param state_dict_uri: The location, in URI format, of the state_dict, for example:

                    - ``/Users/me/path/to/local/state_dict``
                    - ``relative/path/to/local/state_dict``
                    - ``s3://my_bucket/path/to/state_dict``
                    - ``runs:/<mlflow_run_id>/run-relative/path/to/state_dict``

                    For more information about supported URI schemes, see
                    `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                    artifact-locations>`_.

    :param kwargs: kwargs to pass to ``torch.load``.
    :return: A state_dict

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
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
):  # pylint: disable=unused-argument
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

    .. Note:: Only pytorch-lightning modules between versions MIN_REQ_VERSION and
              MAX_REQ_VERSION are known to be compatible with mlflow's autologging.

    :param log_every_n_epoch: If specified, logs metrics once every `n` epochs. By default, metrics
                       are logged after every epoch.
    :param log_every_n_step: If specified, logs batch metrics once every `n` global step.
                       By default, metrics are not logged for steps. Note that setting this to 1 can
                       cause performance issues and is not recommended.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the PyTorch Lightning autologging integration.
                    If ``False``, enables the PyTorch Lightning autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      pytorch and pytorch-lightning that have not been tested against this version
                      of the MLflow client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during PyTorch
                   Lightning autologging. If ``False``, show all events and warnings during
                   PyTorch Lightning autologging.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.

    .. code-block:: python
        :caption: Example

        import os

        import pytorch_lightning as pl
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import MNIST

        try:
            from torchmetrics.functional import accuracy
        except ImportError:
            from pytorch_lightning.metrics.functional import accuracy

        import mlflow.pytorch
        from mlflow import MlflowClient

        # For brevity, here is the simplest most minimal example with just a training
        # loop step, (no validation, no testing). It illustrates how you can use MLflow
        # to auto log parameters, metrics, and models.


        class MNISTModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)

            def forward(self, x):
                return torch.relu(self.l1(x.view(x.size(0), -1)))

            def training_step(self, batch, batch_nb):
                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(dim=1)
                acc = accuracy(pred, y)

                # Use the current of PyTorch logger
                self.log("train_loss", loss, on_epoch=True)
                self.log("acc", acc, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.02)


        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print("run_id: {}".format(r.info.run_id))
            print("artifacts: {}".format(artifacts))
            print("params: {}".format(r.data.params))
            print("metrics: {}".format(r.data.metrics))
            print("tags: {}".format(tags))


        # Initialize our model
        mnist_model = MNISTModel()

        # Initialize DataLoader from MNIST Dataset
        train_ds = MNIST(
            os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
        )
        train_loader = DataLoader(train_ds, batch_size=32)

        # Initialize a trainer
        trainer = pl.Trainer(max_epochs=20, progress_bar_refresh_rate=20)

        # Auto log all MLflow entities
        mlflow.pytorch.autolog()

        # Train the model
        with mlflow.start_run() as run:
            trainer.fit(mnist_model, train_loader)

        # fetch the auto logged parameters and metrics
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    .. code-block:: text
        :caption: Output

        run_id: 42caa17b60cb489c8083900fb52506a7
        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/data']
        params: {'betas': '(0.9, 0.999)',
                 'weight_decay': '0',
                 'epochs': '20',
                 'eps': '1e-08',
                 'lr': '0.02',
                 'optimizer_name': 'Adam', '
                 amsgrad': 'False'}
        metrics: {'acc_step': 0.0,
                  'train_loss_epoch': 1.0917967557907104,
                  'train_loss_step': 1.0794280767440796,
                  'train_loss': 1.0794280767440796,
                  'acc_epoch': 0.0033333334140479565,
                  'acc': 0.0}
        tags: {'Mode': 'training'}

    .. figure:: ../_static/images/pytorch_lightening_autolog.png

        PyTorch autologged MLflow entities
    """
    import atexit

    try:
        import pytorch_lightning as pl
    except ImportError:
        pass
    else:
        from mlflow.pytorch._lightning_autolog import patched_fit

        safe_patch(FLAVOR_NAME, pl.Trainer, "fit", patched_fit, manage_run=True)

    try:
        import lightning as L
    except ImportError:
        pass
    else:
        from mlflow.pytorch._lightning_autolog import patched_fit

        safe_patch(FLAVOR_NAME, L.Trainer, "fit", patched_fit, manage_run=True)

    try:
        import torch.utils.tensorboard.writer
    except ImportError:
        pass
    else:
        from mlflow.pytorch._pytorch_autolog import (
            patched_add_event,
            patched_add_hparams,
            patched_add_summary,
            _flush_queue,
        )

        safe_patch(
            FLAVOR_NAME,
            torch.utils.tensorboard.writer.FileWriter,
            "add_event",
            partial(patched_add_event, mlflow_log_every_n_step=log_every_n_step),
            manage_run=True,
        )
        safe_patch(
            FLAVOR_NAME,
            torch.utils.tensorboard.writer.FileWriter,
            "add_summary",
            patched_add_summary,
            manage_run=True,
        )
        safe_patch(
            FLAVOR_NAME,
            torch.utils.tensorboard.SummaryWriter,
            "add_hparams",
            patched_add_hparams,
            manage_run=True,
        )

        atexit.register(_flush_queue)


if autolog.__doc__ is not None:
    autolog.__doc__ = autolog.__doc__.replace("MIN_REQ_VERSION", str(MIN_REQ_VERSION)).replace(
        "MAX_REQ_VERSION", str(MAX_REQ_VERSION)
    )
