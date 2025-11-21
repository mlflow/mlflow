"""TensorRT MLflow flavor implementation.

This module provides minimal save/log/load support for TensorRT engines (.plan).

Notes:
- PyFunc serving is not implemented. This flavor focuses on artifact logging and
  retrieving deserialized `tensorrt.ICudaEngine` via `load_model`.

"""

from __future__ import annotations

import os
from typing import Any

import yaml

from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
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
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "tensorrt"
_MODEL_DATA_SUBPATH = "data"
_ENGINE_FILE_NAME = "model.trt"


def get_default_pip_requirements() -> list[str]:
    """Returns a minimal default set of pip requirements for this flavor.

    TensorRT Python package name is commonly `tensorrt` (availability depends on platform).

    Returns:
        list[str]: A list of pinned pip requirements for the TensorRT flavor.

    """
    requirements = []
    for pkg in [
        "tensorrt",
        "tensorrt-cu11",
        "tensorrt-cu11-bindings",
        "tensorrt-cu11-libs",
        "tensorrt-cu12",
        "tensorrt-cu12-bindings",
        "tensorrt-cu12-libs",
        "tensorrt-cu13",
        "tensorrt-cu13-bindings",
        "tensorrt-cu13-libs",
    ]:
        try:
            requirements.append(_get_pinned_requirement(pkg))
        except ImportError:
            continue
    return requirements


def get_default_conda_env() -> dict[str, Any]:
    """Returns the default Conda environment for models produced by this flavor.

    Returns:
        dict[str, Any]: A dictionary representing the default Conda environment configuration.

    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    trt_engine: Any,
    path: str,
    conda_env: dict[str, Any] | None = None,
    code_paths: list[str] | None = None,
    mlflow_model: Model | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Save a TensorRT engine to a local directory in MLflow format.

    Args:
        trt_engine: A `tensorrt.ICudaEngine` instance to be saved.
        path: Destination directory for the MLflow model.
        conda_env: {conda_env}
        code_paths: {code_paths}
        mlflow_model: Optional MLflow `Model` instance.
        signature: Optional model signature.
        input_example: Optional input example for the model.
        pip_requirements: {pip_requirements}
        extra_pip_requirements: {extra_pip_requirements}
        metadata: {metadata}
        kwargs: Additional keywords for compatibility with other flavors.

    """
    import tensorrt as trt

    try:
        import pynvml

        pynvml.nvmlInit()
        cuda_driver_version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()

    except ImportError:
        cuda_driver_version = None

    try:
        import torch

        compute_capability: tuple[int, int] | None = torch.cuda.get_device_capability()
        cuda_version = torch.version.cuda
        gpu_model = torch.cuda.get_device_name(0)

    except ImportError:
        compute_capability = cuda_version = gpu_model = None

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if not isinstance(trt_engine, trt.ICudaEngine):
        raise TypeError("Argument 'trt_engine' should be a tensorrt.ICudaEngine")

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    if mlflow_model is None:
        mlflow_model = Model()

    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        from mlflow.models.utils import _save_example  # local import to match mlflow style

        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    model_data_subpath = _MODEL_DATA_SUBPATH
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)

    # Serialize engine to file
    engine_path = os.path.join(model_data_path, _ENGINE_FILE_NAME)
    serialized: bytes = trt_engine.serialize()
    with open(engine_path, "wb") as f:
        f.write(serialized)

    # Register flavor (note: no pyfunc flavor added for TensorRT)
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        compute_capability=compute_capability,
        gpu_model=gpu_model,
        cuda_driver_version=cuda_driver_version,
        cuda_version=cuda_version,
        trt_version=str(trt.__version__),
        model_data=model_data_subpath,
        code=code_dir_subpath,
        engine_file=_ENGINE_FILE_NAME,
    )

    # Include a pyfunc entry to enable model loading via `pyfunc.load_model`, but raise on use.
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.tensorrt",
        data=model_data_subpath,
        code=code_dir_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
    )

    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    # Environment files
    if conda_env is None:
        default_reqs = get_default_pip_requirements() if pip_requirements is None else None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w", encoding="utf-8") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _load_model(path: str):
    """Deserialize a TensorRT engine from the given directory path (data dir).

    Args:
        path: Local filesystem path to the directory containing the serialized engine.

    Returns:
        tensorrt.ICudaEngine: The deserialized TensorRT engine.

    """
    import tensorrt as trt

    engine_path = os.path.join(path, _ENGINE_FILE_NAME)
    trt_logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        serialized = f.read()
    return runtime.deserialize_cuda_engine(serialized)


def _load_pyfunc(path: str) -> _TensorRTWrapper:
    """Load PyFunc implementation for the TensorRT flavor.

    Args:
        path: Local filesystem path to the directory containing the model data.

    Returns:
        _TensorRTWrapper: A wrapper instance that implements the PyFunc interface.

    """
    engine = _load_model(path)
    return _TensorRTWrapper(engine)


class _TensorRTWrapper:
    """PyFunc wrapper for TensorRT engines.

    Expects inputs as a dictionary mapping input tensor names to torch.Tensors. Returns
    a dictionary mapping output tensor names to torch.Tensors.

    """

    def __init__(self, trt_engine) -> None:
        """Initialize the TensorRT wrapper.

        Args:
            trt_engine: A tensorrt.ICudaEngine instance to wrap.

        """
        import tensorrt as trt
        from packaging.version import Version

        self._engine = trt_engine
        self._trt = trt
        self._is_trt_below_10 = Version(trt.__version__) < Version("10")
        # Create an execution context
        self._context = self._engine.create_execution_context()

    def get_raw_model(self):
        """Returns the underlying TensorRT engine.

        Returns:
            tensorrt.ICudaEngine: The wrapped TensorRT engine instance.

        """
        return self._engine

    def predict(self, data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute inference using the TensorRT engine.

        Args:
            data (dict[str, Any]: A dictionary mapping input tensor names (str) to torch.Tensor
                objects. All tensors should be on the same CUDA device and have compatible dtypes.
            params (dict[str, Any]): Optional additional parameters (currently unused).

        Returns:
            dict[str, Any]: A dictionary mapping output tensor names to their
                corresponding torch.Tensor results.

        Raises:
            TypeError: If data is not a dictionary.

        """
        import torch

        if not isinstance(data, dict):
            raise TypeError(
                "TensorRT pyfunc expects a dict[str, torch.Tensor] with input tensor names."
            )

        # Determine batch size from the first tensor
        first_tensor = next(iter(data.values()))
        batch_size = int(getattr(first_tensor, "shape")[0])

        outputs = {}
        bindings = []

        # Iterate over bindings and set addresses
        num_items = (
            int(self._engine.num_bindings)
            if self._is_trt_below_10
            else int(self._engine.num_io_tensors)
        )

        for idx in range(num_items):
            name = self._engine.get_tensor_name(idx)
            mode = self._engine.get_tensor_mode(name)

            if mode == self._trt.TensorIOMode.INPUT:
                tensor = data[name].contiguous()
                if self._is_trt_below_10:
                    bindings.append(tensor.data_ptr())
                else:
                    self._context.set_tensor_address(name, tensor.data_ptr())
                # Provide the runtime input shape
                self._context.set_input_shape(name, tensor.shape)
            else:
                # Create an output tensor with the appropriate shape.
                # We take dtype/device from the first input tensor for simplicity.
                shape = tuple(self._engine.get_tensor_shape(name))
                # Replace dynamic batch dim with actual batch size if needed
                shape = (batch_size, *tuple(shape[1:]))
                out_tensor = torch.empty(
                    size=shape,
                    dtype=getattr(first_tensor, "dtype", None),
                    device=getattr(first_tensor, "device", None),
                )
                outputs[name] = out_tensor
                if self._is_trt_below_10:
                    bindings.append(out_tensor.data_ptr())
                else:
                    self._context.set_tensor_address(name, out_tensor.data_ptr())

        # Execute asynchronously on current CUDA stream
        cuda_stream = torch.cuda.current_stream().cuda_stream
        if self._is_trt_below_10:
            self._context.execute_async_v2(bindings, cuda_stream)
        else:
            self._context.execute_async_v3(cuda_stream)

        return outputs


def load_model(model_uri: str, dst_path: str | None = None):
    """Load a TensorRT engine from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:
            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``
        dst_path: The local filesystem path to which to download the model artifact.
                  This directory must already exist. If unspecified, a local output
                  path will be created.

    Returns:
        tensorrt.ICudaEngine: The loaded TensorRT engine.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    engine_dir = os.path.join(local_model_path, flavor_conf["model_data"])
    return _load_model(engine_dir)


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    trt_engine: Any,
    artifact_path: str | None = None,
    conda_env: dict[str, Any] | None = None,
    code_paths: list[str] | None = None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    **kwargs: Any,
):
    """Log a TensorRT engine as an MLflow artifact for the current run."""
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=__import__(__name__, fromlist=["*"]),
        trt_engine=trt_engine,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


__all__ = [
    "save_model",
    "log_model",
    "load_model",
    "get_default_pip_requirements",
    "get_default_conda_env",
]
