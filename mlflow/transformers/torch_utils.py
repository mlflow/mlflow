from __future__ import annotations

from typing import TYPE_CHECKING

from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    import torch

_TORCH_DTYPE_KEY = "torch_dtype"


def _get_dtype_argument_key() -> str:
    """
    Return the dtype argument name accepted by the installed transformers version.
    transformers 4.56.0 renamed `torch_dtype` to `dtype` and warns when the old name is used.
    """
    import transformers

    if Version(transformers.__version__) >= Version("4.56.0"):
        return "dtype"
    return _TORCH_DTYPE_KEY


def _extract_torch_dtype_if_set(pipeline) -> torch.dtype | None:
    """
    Extract the torch datatype argument if set and return as a string encoded value.
    """
    try:
        import torch
    except ImportError:
        # If torch is not installed, safe to assume the model doesn't have a custom torch_dtype
        return None

    # Check model dtype as pipeline's torch_dtype field doesn't always reflect the model's dtype
    model_dtype = pipeline.model.dtype if hasattr(pipeline.model, "dtype") else None

    # If the underlying model is PyTorch model, dtype must be a torch.dtype instance
    return model_dtype if isinstance(model_dtype, torch.dtype) else None


def _deserialize_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert the string-encoded `torch_dtype` pipeline argument back to the correct `torch.dtype`
    instance value for applying to a loaded pipeline instance.
    """
    try:
        import torch
    except ImportError as e:
        raise MlflowException(
            "Unable to determine if the value supplied by the argument "
            "torch_dtype is valid since torch is not installed.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e

    dtype_str = dtype_str.removeprefix("torch.")

    dtype = getattr(torch, dtype_str, None)
    if isinstance(dtype, torch.dtype):
        return dtype

    raise MlflowException(
        f"The value '{dtype_str}' is not a valid torch.dtype",
        error_code=INVALID_PARAMETER_VALUE,
    )
