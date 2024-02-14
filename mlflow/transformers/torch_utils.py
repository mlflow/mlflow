from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    import torch

_TORCH_DTYPE_KEY = "torch_dtype"


def _extract_torch_dtype_if_set(pipeline) -> Optional[torch.dtype]:
    """
    Extract the torch datatype argument if set and return as a string encoded value.
    """
    if torch_dtype := getattr(pipeline, _TORCH_DTYPE_KEY, None):
        # Torch dtype value may be a string or a torch.dtype instance
        if isinstance(torch_dtype, str):
            torch_dtype = _deserialize_torch_dtype(torch_dtype)
        return torch_dtype

    try:
        import torch
    except ImportError:
        # If torch is not installed, safe to assume the model doesn't have a custom torch_dtype
        return None

    # Transformers pipeline doesn't inherit underlying model's dtype, so we have to also check
    # the model's dtype.
    model = pipeline.model
    model_dtype = getattr(model.config, _TORCH_DTYPE_KEY, None) or getattr(model, "dtype", None)

    # However, we should not extract dtype from parameters if it's default one (float32),
    # to avoid setting torch_dtype for the model that doesn't support it.
    if isinstance(model_dtype, str):
        model_dtype = _deserialize_torch_dtype(model_dtype)
    return model_dtype if model_dtype != torch.float32 else None


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
