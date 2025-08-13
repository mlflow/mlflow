from __future__ import annotations

from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    import torch

_TORCH_DTYPE_KEY = "torch_dtype"


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

    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]

    dtype = getattr(torch, dtype_str, None)
    if isinstance(dtype, torch.dtype):
        return dtype

    raise MlflowException(
        f"The value '{dtype_str}' is not a valid torch.dtype",
        error_code=INVALID_PARAMETER_VALUE,
    )
