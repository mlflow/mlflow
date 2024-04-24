import logging

from mlflow.environment_variables import (
    MLFLOW_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES,
    MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.transformers.flavor_config import FlavorKey, get_peft_base_model, is_peft_model

_logger = logging.getLogger(__name__)

# File/directory names for saved artifacts
_MODEL_BINARY_FILE_NAME = "model"
_COMPONENTS_BINARY_DIR_NAME = "components"
_PROCESSOR_BINARY_DIR_NAME = "processor"


def save_pipeline_pretrained_weights(path, pipeline, flavor_conf, processor=None):
    """
    Save the binary artifacts of the pipeline to the specified local path.

    Args:
        path: The local path to save the pipeline
        pipeline: Transformers pipeline instance
        flavor_config: The flavor configuration constructed for the pipeline
        processor: Optional processor instance to save alongside the pipeline
    """
    model = get_peft_base_model(pipeline.model) if is_peft_model(pipeline.model) else pipeline.model

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
    loaded[FlavorKey.MODEL] = _load_model(model_path, flavor_conf, accelerate_conf, device)

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

    _COMPONENT_TO_AUTOCLASS_MAP = {
        FlavorKey.TOKENIZER: transformers.AutoTokenizer,
        FlavorKey.FEATURE_EXTRACTOR: transformers.AutoFeatureExtractor,
        FlavorKey.PROCESSOR: transformers.AutoProcessor,
        FlavorKey.IMAGE_PROCESSOR: transformers.AutoImageProcessor,
    }

    component_name = flavor_conf[FlavorKey.COMPONENT_TYPE.format(name)]
    if hasattr(transformers, component_name):
        cls = getattr(transformers, component_name)
        trust_remote = False
    else:
        if local_path is None:
            raise MlflowException(
                f"A custom component `{component_name}` was specified, "
                "but no local config file was found to retrieve the "
                "definition. Make sure your model was saved with "
                "save_pretrained=True."
            )
        cls = _COMPONENT_TO_AUTOCLASS_MAP[name]
        trust_remote = True

    if local_path is not None:
        # Load component from local file
        path = local_path.joinpath(_COMPONENTS_BINARY_DIR_NAME, name)
        return cls.from_pretrained(str(path), trust_remote_code=trust_remote)
    else:
        # Load component from HuggingFace Hub
        repo = flavor_conf[FlavorKey.COMPONENT_NAME.format(name)]
        revision = flavor_conf.get(FlavorKey.COMPONENT_REVISION.format(name))
        return cls.from_pretrained(repo, revision=revision, trust_remote_code=trust_remote)


def _load_class_from_transformers_config(model_name_or_path, revision=None):
    """
    This method retrieves the Transformers AutoClass from the transformers config.
    Using the correct AutoClass allows us to leverage Transformers' model loading
    machinery, which is necessary for supporting models using custom code.
    """
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        revision=revision,
        # trust_remote_code is set to True in order to
        # make sure the config gets loaded as the correct
        # class. if this is not set for custom models, the
        # base class will be loaded instead of the custom one.
        trust_remote_code=True,
    )

    # the model's class name (e.g. "MPTForCausalLM")
    # is stored in the `architectures` field. it
    # seems to usually just have one element.
    class_name = config.architectures[0]

    # if the class is available in transformers natively,
    # then we don't need to execute any custom code.
    if hasattr(transformers, class_name):
        cls = getattr(transformers, class_name)
        return cls, False
    else:
        # else, we need to fetch the correct AutoClass.
        # this is defined in the `auto_map` field. there
        # should only be one AutoClass that maps to the
        # model's class name.
        auto_classes = [
            auto_class
            for auto_class, module in config.auto_map.items()
            if module.split(".")[-1] == class_name
        ]

        if len(auto_classes) == 0:
            raise MlflowException(f"Couldn't find a loader class for {class_name}")

        auto_class = auto_classes[0]
        cls = getattr(transformers, auto_class)

        # we will need to trust remote code when loading the model
        return cls, True


def _load_model(model_name_or_path, flavor_conf, accelerate_conf, device, revision=None):
    """
    Try to load a model with various loading strategies.
      1. Try to load the model with accelerate
      2. Try to load the model with the specified device
      3. Load the model without the device
    """
    import transformers

    if hasattr(transformers, flavor_conf[FlavorKey.MODEL_TYPE]):
        cls = getattr(transformers, flavor_conf[FlavorKey.MODEL_TYPE])
        trust_remote = False
    else:
        cls, trust_remote = _load_class_from_transformers_config(
            model_name_or_path, revision=revision
        )

    load_kwargs = {"revision": revision} if revision else {}
    if trust_remote:
        load_kwargs.update({"trust_remote_code": True})

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
