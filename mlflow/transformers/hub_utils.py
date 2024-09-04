import functools
import json
import logging
import os
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

_logger = logging.getLogger(__name__)


# NB: The maxsize=1 is added for encouraging the cache refresh so the user doesn't get stale
#    commit hash from the cache. This doesn't work perfectly because it only updates cache
#    when the user calls it with a different repo name, but it's better than nothing.
@functools.lru_cache(maxsize=1)
def get_latest_commit_for_repo(repo: str) -> str:
    """
    Fetches the latest commit hash for a repository from the HuggingFace model hub.
    """
    try:
        import huggingface_hub as hub
    except ImportError:
        raise MlflowException(
            "Unable to fetch model commit hash from the HuggingFace model hub. "
            "This is required for saving Transformer model without base model "
            "weights, while ensuring the version consistency of the model. "
            "Please install the `huggingface-hub` package and retry.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    return hub.HfApi().model_info(repo).sha


def is_valid_hf_repo_id(maybe_repo_id: Optional[str]) -> bool:
    """
    Check if the given string is a valid HuggingFace repo identifier e.g. "username/repo_id".
    """

    if not maybe_repo_id or os.path.isdir(maybe_repo_id):
        return False

    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id
    except ImportError:
        raise MlflowException(
            "Unable to validate the repository identifier for the HuggingFace model hub "
            "because the `huggingface-hub` package is not installed. Please install the "
            "package with `pip install huggingface-hub` command and retry."
        )

    try:
        validate_repo_id(maybe_repo_id)
        return True
    except HFValidationError as e:
        _logger.warning(f"The repository identified {maybe_repo_id} is invalid: {e}")
        return False


def get_hf_model_info_from_local_checkpoint(path):
    if not os.path.isdir(path):
        raise MlflowException(
            f"The provided path {path} is not a directory. Please provide a valid path to "
            "the directory containing the local Transformers model checkpoint.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        raise MlflowException(
            f"The provided directory {path} does not contain the config.json file. "
            "Please ensure that the directory contains a valid transformers model checkpoint.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    with open(config_path) as f:
        config = json.load(f)
    repo_id = config.get("_name_or_path")

    try:
        import huggingface_hub
    except ImportError:
        raise MlflowException(
            "Unable to fetch the model information from the HuggingFace model hub because "
            "the huggingface-hub package to be installed. Please install the package by "
            "running `pip install huggingface-hub`.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    return huggingface_hub.model_info(repo_id)


def infer_framework_from_repo(repo_id: str) -> str:
    """
    Infer framework mimicing Transformers implementation, but without loading
    the model into memory
    https://github.com/huggingface/transformers/blob/44f6fdd74f84744b159fa919474fd3108311a906/src/transformers/pipelines/base.py#L215C28-L215C37
    """
    import huggingface_hub
    from transformers.utils import is_torch_available

    if not is_torch_available():
        return "tf"

    # Check repo tag
    repo_tag = huggingface_hub.model_info(repo_id).tags
    is_torch_supported = "pytorch" in repo_tag
    if not is_torch_supported:
        return "tf"

    # Default to Pytorch if both are available, probably we should do a better check
    return "pt"
