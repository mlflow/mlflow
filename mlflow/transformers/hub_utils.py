import functools
import logging
import os
import shutil
from typing import Dict, Optional

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

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


def download_model_weights_from_hub(flavor_conf: Dict, dst_path: str):
    """
    Download the model weights from the HuggingFace model hub.

    The model repository can store multiple weight files for different frameworks. This function
    determines which files to download based on the framework specified in the flavor configuration
    and the availability of the weight files in the repository.
    """
    from huggingface_hub import hf_hub_download
    from transformers.utils import (
        CONFIG_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
        TF2_WEIGHTS_INDEX_NAME,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        WEIGHTS_INDEX_NAME,
        WEIGHTS_NAME,
    )
    from transformers.utils.hub import get_checkpoint_shard_files, has_file

    from mlflow.transformers.flavor_config import FlavorKey

    framework = flavor_conf.get(FlavorKey.FRAMEWORK)
    repo_id = flavor_conf[FlavorKey.MODEL_NAME]
    revision = flavor_conf[FlavorKey.MODEL_REVISION]

    def _try_download(index_filename: str, weight_filename: str) -> bool:
        """
        Try to download the weight files with the specific framework e.g. Pytorch.

        For each framework, the weight files can be stored in either a single file or multiple
        shards. For the single file case, the weight file is directly downloaded. For the multiple
        shards case, the index file is first downloaded to get the list of shard files, and then
        each shard file is downloaded.

        Returns:
            Whether the download is successful.
        """
        try:
            if has_file(repo_id, index_filename, revision=revision):
                index_local_file = hf_hub_download(
                    repo_id,
                    index_filename,
                    revision=revision,
                    local_dir=dst_path,
                )
                cached_files, _ = get_checkpoint_shard_files(
                    repo_id,
                    index_local_file,
                    revision=revision,
                )
                for file in cached_files:
                    # Copy files to the destination directory
                    shutil.copy(file, os.path.join(dst_path, os.path.basename(file)))
            else:
                hf_hub_download(repo_id, weight_filename, revision=revision, local_dir=dst_path)
            return True
        except Exception:
            return False

    # Load the config.json file
    hf_hub_download(repo_id, CONFIG_NAME, revision=revision, local_dir=dst_path)

    # Cascade through the possible weight files to download with the priority:
    #    Safetensor -> Pytorch -> TF 2 -> TF 1
    #
    # The index and file names are fixed for each framework, for example, for Pytorch
    # the index file is "pytorch_model.bin.index.json" and the weight file is
    # "pytorch_model.bin". Therefore, we can use the existence of those files to determine
    # which framework the weight file is for.
    if _try_download(SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME):
        pass
    elif framework == "pt":
        if _try_download(WEIGHTS_INDEX_NAME, WEIGHTS_NAME):
            pass
        else:
            raise MlflowException(
                "Framework is set to Pytorch, but the weight file is not found "
                f"in the HuggingFace Hub repository {repo_id}.",
            )
    else:
        if _try_download(TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME) or _try_download(
            None, TF_WEIGHTS_NAME
        ):
            pass
        else:
            raise MlflowException(
                "Framework is set to Tensorflow, but the weight file is not found "
                f"in the HuggingFace Hub repository {repo_id}.",
            )


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
