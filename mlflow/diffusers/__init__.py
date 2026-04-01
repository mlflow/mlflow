"""
The ``mlflow.diffusers`` module provides an API for logging and loading diffusion model
adapters (LoRA, LoKr, LoHa) as MLflow Models. This module exports adapter models with
the following flavors:

:py:mod:`mlflow.diffusers`
    Adapter weights in safetensors format, with a reference to the base model.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
    The pyfunc wrapper loads the base diffusion pipeline and applies the adapter
    at inference time.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema
from mlflow.types.schema import ColSpec
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

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "diffusers"

_ADAPTER_WEIGHTS_DIR = "adapter_weights"
_ADAPTER_CONFIG_FILE = "adapter_config.json"

SUPPORTED_ADAPTER_TYPES = ("lora", "lokr", "loha")


def _detect_device(device=None):
    import torch

    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def _get_default_signature():
    return ModelSignature(
        inputs=Schema([ColSpec(type=DataType.string, name="prompt")]),
        outputs=Schema([ColSpec(type=DataType.binary, name="image")]),
        params=ParamSchema([
            ParamSpec(name="num_inference_steps", dtype=DataType.integer, default=30),
            ParamSpec(name="guidance_scale", dtype=DataType.double, default=7.5),
            ParamSpec(name="height", dtype=DataType.integer, default=512),
            ParamSpec(name="width", dtype=DataType.integer, default=512),
        ]),
    )


def get_default_pip_requirements():
    import importlib.util

    # Core requirements — always needed
    packages = ["diffusers", "transformers", "torch"]

    # Optional but recommended packages — only pin if installed
    for optional in ["accelerate", "safetensors", "peft"]:
        if importlib.util.find_spec(optional):
            packages.append(optional)

    return [_get_pinned_requirement(pkg) for pkg in packages]


def get_default_conda_env():
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@dataclass
class DiffusersAdapterModel:
    adapter_path: str
    base_model_id: str
    adapter_type: Literal["lora", "lokr", "loha"]

    def load_pipeline(self, **kwargs):
        from diffusers import DiffusionPipeline

        device = _detect_device(kwargs.pop("device", None))
        kwargs.setdefault("torch_dtype", "auto")

        pipe = DiffusionPipeline.from_pretrained(self.base_model_id, **kwargs)

        if self.adapter_type == "lora":
            pipe.load_lora_weights(self.adapter_path)
        else:
            raise MlflowException(
                f"Loading adapter type '{self.adapter_type}' is not yet supported. "
                f"Currently only 'lora' adapters can be loaded at inference time."
            )

        return pipe.to(device)


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="diffusers"))
def save_model(
    adapter_path,
    path,
    base_model_id,
    adapter_type: Literal["lora", "lokr", "loha"] = "lora",
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """Save a diffusers adapter model to a path on the local file system.

    Args:
        adapter_path: Path to the adapter weights. Can be a single .safetensors file
            or a directory containing adapter files.
        path: Local path where the model is to be saved.
        base_model_id: HuggingFace model ID or local path of the base diffusion model
            that this adapter was trained on (e.g., "black-forest-labs/FLUX.1-dev").
        adapter_type: Type of adapter. One of "lora", "lokr", "loha".
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
    """
    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except ImportError:
        _logger.warning(
            "diffusers package is not installed. The saved model will not record a "
            "diffusers version, which may affect reproducibility at load time."
        )
        diffusers_version = "unknown"

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    adapter_type = adapter_type.lower()
    if adapter_type not in SUPPORTED_ADAPTER_TYPES:
        raise MlflowException.invalid_parameter_value(
            f"Unsupported adapter type: {adapter_type}. "
            f"Supported types: {SUPPORTED_ADAPTER_TYPES}"
        )

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise MlflowException.invalid_parameter_value(
            f"Adapter path does not exist: {adapter_path}"
        )

    path = Path(path)

    _validate_and_prepare_target_save_path(path)
    code_path_subdir = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()

    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        signature = _infer_signature_from_input_example(saved_example, None)
    elif signature is None:
        signature = _get_default_signature()
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    # Copy adapter weights
    weights_dst = path / _ADAPTER_WEIGHTS_DIR
    if adapter_path.is_file():
        weights_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_path, weights_dst / adapter_path.name)
    elif adapter_path.is_dir():
        shutil.copytree(adapter_path, weights_dst)
    else:
        raise MlflowException.invalid_parameter_value(
            f"Adapter path is neither a file nor a directory: {adapter_path}"
        )

    # Write adapter config
    adapter_config = {
        "base_model_id": base_model_id,
        "adapter_type": adapter_type,
    }
    with open(path / _ADAPTER_CONFIG_FILE, "w") as f:
        json.dump(adapter_config, f, indent=2)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        base_model_id=base_model_id,
        adapter_type=adapter_type,
        adapter_weights=_ADAPTER_WEIGHTS_DIR,
        diffusers_version=diffusers_version,
        code=code_path_subdir,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.diffusers",
        model_path=_ADAPTER_WEIGHTS_DIR,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_path_subdir,
    )

    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(str(path / MLMODEL_FILE_NAME))

    # Save environment files
    if conda_env is None:
        default_reqs = get_default_pip_requirements() if pip_requirements is None else None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(path / _CONDA_ENV_FILE_NAME, "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(str(path / _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(str(path / _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))
    _PythonEnv.current().to_yaml(str(path / _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="diffusers"))
def log_model(
    adapter_path,
    base_model_id,
    adapter_type: Literal["lora", "lokr", "loha"] = "lora",
    artifact_path: str | None = None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    name: str | None = None,
    **kwargs,
):
    """Log a diffusers adapter model as an MLflow artifact for the current run.

    Args:
        adapter_path: Path to the adapter weights. Can be a single .safetensors file
            or a directory containing adapter files.
        base_model_id: HuggingFace model ID or local path of the base diffusion model.
        adapter_type: Type of adapter. One of "lora", "lokr", "loha".
        artifact_path: Deprecated. Use ``name`` instead.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under this name.
        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for model version creation.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        name: {{ name }}
        kwargs: Extra arguments to pass to :py:func:`mlflow.models.Model.log`.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance.
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.diffusers,
        adapter_path=adapter_path,
        base_model_id=base_model_id,
        adapter_type=adapter_type,
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


def load_model(model_uri, dst_path=None):
    """Load a diffusers adapter model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. Examples:

            - ``/Users/me/path/to/local/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``

        dst_path: The local filesystem path to download the model artifact to.

    Returns:
        A :py:class:`DiffusersAdapterModel` with adapter_path, base_model_id,
        and adapter_type. Call ``.load_pipeline()`` to get a ready-to-use
        diffusers pipeline with the adapter applied.
    """
    local_model_path = Path(
        _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    )
    flavor_conf = _get_flavor_configuration(
        model_path=str(local_model_path), flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(str(local_model_path), flavor_conf)

    adapter_weights_path = local_model_path / flavor_conf["adapter_weights"]

    return DiffusersAdapterModel(
        adapter_path=str(adapter_weights_path),
        base_model_id=flavor_conf["base_model_id"],
        adapter_type=flavor_conf["adapter_type"],
    )


def _load_pyfunc(path, model_config=None):
    from mlflow.diffusers.wrapper import _DiffusersAdapterWrapper

    path = Path(path)
    flavor_conf = _get_flavor_configuration(model_path=str(path), flavor_name=FLAVOR_NAME)
    pyfunc_conf = _get_flavor_configuration(model_path=str(path), flavor_name=pyfunc.FLAVOR_NAME)
    adapter_path = str(path / pyfunc_conf["model_path"])

    return _DiffusersAdapterWrapper(
        adapter_path=adapter_path,
        flavor_conf=flavor_conf,
        model_config=model_config,
    )
