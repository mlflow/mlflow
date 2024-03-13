import importlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import transformers
from transformers.utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME

from mlflow.environment_variables import (
    MLFLOW_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES,
    MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model
from mlflow.utils.file_utils import create_tmp_dir
from mlflow.utils.model_utils import DEFAULT_CODE_SUBPATH, FLAVOR_CONFIG_CODE

_logger = logging.getLogger(__name__)

# File/directory names for saved artifacts
_MODEL_BINARY_FILE_NAME = "model"
_COMPONENTS_BINARY_DIR_NAME = "components"
_PROCESSOR_BINARY_DIR_NAME = "processor"
_MLFLOW_PYFUNC_CUSTOM_MODULES_NAME = "mlflow_pyfunc_custom_modules"

# we copy custom module code into this directory
# so custom class defs can imported properly
MLFLOW_CUSTOM_MODULES_DIR_ENV_VAR = "MLFLOW_CUSTOM_MODULES_DIR"


def save_pipeline_pretrained_weights(
    path, pipeline, flavor_conf, processor=None, code_dir_subpath=None
):
    """
    Save the binary artifacts of the pipeline to the specified local path.

    Args:
        path: The local path to save the pipeline
        pipeline: Transformers pipeline instance
        flavor_config: The flavor configuration constructed for the pipeline
        processor: Optional processor instance to save alongside the pipeline
    """
    model = get_peft_base_model(pipeline.model) if is_peft_model(pipeline.model) else pipeline.model

    # if the user didn't provide any code_paths, we
    # create a directory to store the remote code
    if code_dir_subpath is None:
        code_dir_subpath = DEFAULT_CODE_SUBPATH

    # this function does nothing if there is no remote code
    save_remote_code_to_code_path(pipeline, path / code_dir_subpath)

    model.save_pretrained(
        save_directory=path.joinpath(_MODEL_BINARY_FILE_NAME),
        max_shard_size=MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE.get(),
    )

    component_dir = path.joinpath(_COMPONENTS_BINARY_DIR_NAME)
    for name in flavor_conf.get(FlavorKey.COMPONENTS, []):
        getattr(pipeline, name).save_pretrained(component_dir.joinpath(name))

    if processor:
        processor.save_pretrained(component_dir.joinpath(_PROCESSOR_BINARY_DIR_NAME))


def load_model_and_components_from_local(path, flavor_conf, accelerate_conf, device=None):
    """
    Load the model and components of a Transformer pipeline from the specified local path.

    Args:
        path: The local path contains MLflow model artifacts
        flavor_conf: The flavor configuration
        accelerate_conf: The configuration for the accelerate library
        device: The device to load the model onto
    """
    loaded = {}

    # NB: Path resolution for models that were saved prior to 2.4.1 release when the patching for
    #     the saved pipeline or component artifacts was handled by duplicate entries for components
    #     (artifacts/pipeline/* and artifacts/components/*) and pipelines were saved via the
    #     "artifacts/pipeline/*" path. In order to load the older formats after the change, the
    #     presence of the new path key is checked.
    model_path = path.joinpath(flavor_conf.get(FlavorKey.MODEL_BINARY, "pipeline"))
    hf_config = json.loads(path.joinpath(_MODEL_BINARY_FILE_NAME, "config.json").read_text())
    loaded[FlavorKey.MODEL] = _load_model(
        model_path, flavor_conf, hf_config, accelerate_conf, device
    )

    components = flavor_conf.get(FlavorKey.COMPONENTS, [])
    if FlavorKey.PROCESSOR_TYPE in flavor_conf:
        components.append("processor")

    for component_key in components:
        loaded[component_key] = _load_component(flavor_conf, component_key, local_path=path)

    return loaded


def load_model_and_components_from_huggingface_hub(flavor_conf, accelerate_conf, device=None):
    """
    Load the model and components of a Transformer pipeline from HuggingFace Hub.

    Args:
        flavor_conf: The flavor configuration
        accelerate_conf: The configuration for the accelerate library
        device: The device to load the model onto
    """
    loaded = {}

    model_repo = flavor_conf[FlavorKey.MODEL_NAME]
    model_revision = flavor_conf.get(FlavorKey.MODEL_REVISION)

    if not model_revision:
        raise MlflowException(
            "The model was saved with 'save_pretrained' set to False, but the commit hash is not "
            "found in the saved metadata. Loading the model with the different version may cause "
            "inconsistency issue and security risk.",
            error_code=INVALID_STATE,
        )

    loaded[FlavorKey.MODEL] = _load_model(
        model_repo, flavor_conf, accelerate_conf, device, revision=model_revision
    )

    components = flavor_conf.get(FlavorKey.COMPONENTS, [])
    if FlavorKey.PROCESSOR_TYPE in flavor_conf:
        components.append("processor")

    for name in components:
        loaded[name] = _load_component(flavor_conf, name)

    return loaded


def _load_component(flavor_conf, name, local_path=None):
    import transformers

    cls = getattr(transformers, flavor_conf[FlavorKey.COMPONENT_TYPE.format(name)])

    if local_path is not None:
        # Load component from local file
        path = local_path.joinpath(_COMPONENTS_BINARY_DIR_NAME, name)
        return cls.from_pretrained(str(path))
    else:
        # Load component from HuggingFace Hub
        repo = flavor_conf[FlavorKey.COMPONENT_NAME.format(name)]
        revision = flavor_conf.get(FlavorKey.COMPONENT_REVISION.format(name))
        return cls.from_pretrained(repo, revision=revision)


def save_remote_code_to_code_path(built_pipeline: transformers.Pipeline, code_path: Path):
    model = built_pipeline.model
    config = model.config.to_dict()
    if "auto_map" not in config:
        # no remote code to save
        return

    auto_map = config["auto_map"]
    for definition in auto_map.values():
        repo_and_local_path = definition.split("--")

        if len(repo_and_local_path) == 2:
            repo_path, _ = repo_and_local_path
        else:
            # HF saves these types of definitions locally in the
            # save_pretrained() method. see `custom_object_save()`
            # in `transformers/dynamic_module_utils.py`
            return

        user, repo = repo_path.split("/")
        hf_cache_path = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME

        # TODO: figure out how to get the revision hash of the model.
        #       this is necessary because the remote code is saved in
        #       a directory named with the revision hash. for now, we
        #       just grab the first directory in the user/repo path that
        #       isn't something like __init__.py or __pycache__, but of
        #       course this is not suitable long-term.
        user_repo_path = hf_cache_path / user / repo
        commit_path = [
            path for path in list(user_repo_path.iterdir()) if not path.name.startswith("_")
        ][0]

        local_module_path = code_path / user / repo

        if local_module_path.exists():
            return

        local_module_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(commit_path, local_module_path, dirs_exist_ok=True)


def init_custom_module_directory(model_path: Path, flavor_conf: dict):
    if MLFLOW_CUSTOM_MODULES_DIR_ENV_VAR not in os.environ:
        tmp_dir = create_tmp_dir()
        os.environ[MLFLOW_CUSTOM_MODULES_DIR_ENV_VAR] = str(tmp_dir)
    custom_module_dir = Path(os.getenv(MLFLOW_CUSTOM_MODULES_DIR_ENV_VAR))

    custom_module_path = custom_module_dir / _MLFLOW_PYFUNC_CUSTOM_MODULES_NAME
    custom_module_path.mkdir(parents=True, exist_ok=True)
    init_py = custom_module_path / "__init__.py"
    if not init_py.exists():
        init_py.touch()
        sys.path.append(str(custom_module_dir))
        importlib.invalidate_caches()

    # TODO: only do this if model config specifies
    #       a local (non-HF Hub) path (though should
    #       be a no-op if it isn't, since the `model`
    #       directory should not contain any .py files)
    for file in model_path.rglob("*.py"):
        rel_path = file.relative_to(model_path)
        dest_path = custom_module_path / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest_path)

    # if modules exist in model/code, also copy them to the custom module directory
    config_code_subpath = flavor_conf.get(FLAVOR_CONFIG_CODE, DEFAULT_CODE_SUBPATH)
    code_subpath = DEFAULT_CODE_SUBPATH if config_code_subpath is None else config_code_subpath
    code_path = model_path.parent / code_subpath
    if code_path.exists():
        for item in code_path.iterdir():
            if item.is_file():
                shutil.copy(item, custom_module_path / item.name)
            else:
                shutil.copytree(item, custom_module_path / item.name, dirs_exist_ok=True)


def _load_model_from_code_paths(cls_type, hf_config):
    auto_map = hf_config.get("auto_map")
    for definition in auto_map.values():
        repo_and_local_path = definition.split("--")

        # class was defined from remote HF code
        if len(repo_and_local_path) == 2:
            repo_path, local_path = repo_and_local_path
            user, repo = repo_path.split(os.path.sep)
            user_repo_path = f".{user}.{repo}"
        else:
            # class was defined locally from model/ directory
            repo_path = ""
            local_path = repo_and_local_path[0]
            user_repo_path = ""

        submodule_components = local_path.split(".")
        cls = submodule_components[-1]
        submodule_path = ".".join(submodule_components[:-1])
        full_submodule_path = (
            f"{_MLFLOW_PYFUNC_CUSTOM_MODULES_NAME}{user_repo_path}.{submodule_path}"
        )

        if cls_type == cls:
            mod = importlib.import_module(full_submodule_path)
            return getattr(mod, cls_type)

    raise MlflowException(f"couldn't find definition for {cls_type}")


def _load_model(model_name_or_path, flavor_conf, hf_config, accelerate_conf, device, revision=None):
    """
    Try to load a model with various loading strategies.
      1. Try to load the model with accelerate
      2. Try to load the model with the specified device
      3. Load the model without the device
    """
    import transformers

    model_classname = flavor_conf[FlavorKey.MODEL_TYPE]
    if hasattr(transformers, model_classname):
        cls = getattr(transformers, model_classname)
    else:
        try:
            init_custom_module_directory(model_name_or_path, flavor_conf)
            cls = _load_model_from_code_paths(flavor_conf[FlavorKey.MODEL_TYPE], hf_config)
        except Exception as e:
            raise MlflowException(
                "Encountered unexpected error while loading model from code paths. "
                f"Full exception trace: {e}"
            )

    load_kwargs = {"revision": revision} if revision else {}

    if model := _try_load_model_with_accelerate(
        cls, model_name_or_path, {**accelerate_conf, **load_kwargs}
    ):
        return model

    load_kwargs["device"] = device
    if torch_dtype := flavor_conf.get(FlavorKey.TORCH_DTYPE):
        load_kwargs[FlavorKey.TORCH_DTYPE] = torch_dtype

    if model := _try_load_model_with_device(cls, model_name_or_path, load_kwargs):
        return model
    _logger.warning(
        "Could not specify device parameter for this pipeline type."
        "Falling back to loading the model with the default device."
    )

    load_kwargs.pop("device", None)
    return cls.from_pretrained(model_name_or_path, **load_kwargs)


def _try_load_model_with_accelerate(model_class, model_name_or_path, load_kwargs):
    if MLFLOW_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES.get():
        return None

    try:
        return model_class.from_pretrained(model_name_or_path, **load_kwargs)
    except (ValueError, TypeError, NotImplementedError, ImportError):
        # NB: ImportError is caught here in the event that `accelerate` is not installed
        # on the system, which will raise if `low_cpu_mem_usage` is set or the argument
        # `device_map` is set and accelerate is not installed.
        pass


def _try_load_model_with_device(model_class, model_name_or_path, load_kwargs):
    try:
        return model_class.from_pretrained(model_name_or_path, **load_kwargs)
    except OSError as e:
        revision = load_kwargs.get("revision")
        if f"{revision} is not a valid git identifier" in str(e):
            raise MlflowException(
                f"The model was saved with a HuggingFace Hub repository name '{model_name_or_path}'"
                f"and a commit hash '{revision}', but the commit is not found in the repository. "
            )
        else:
            raise e
    except (ValueError, TypeError, NotImplementedError):
        pass
