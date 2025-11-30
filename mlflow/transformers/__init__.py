"""MLflow module for HuggingFace/transformer support."""

from __future__ import annotations

import ast
import base64
import binascii
import contextlib
import copy
import functools
import importlib
import json
import logging
import os
import pathlib
import re
import shutil
import string
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
from packaging.version import Version

from mlflow import pyfunc
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.environment_variables import (
    MLFLOW_DEFAULT_PREDICTION_DEVICE,
    MLFLOW_HUGGINGFACE_DEVICE_MAP_STRATEGY,
    MLFLOW_HUGGINGFACE_USE_DEVICE_MAP,
    MLFLOW_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE,
    MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT,
)
from mlflow.exceptions import MlflowException
from mlflow.models import (
    Model,
    ModelInputExample,
    ModelSignature,
)
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path
from mlflow.transformers.flavor_config import (
    FlavorKey,
    build_flavor_config,
    build_flavor_config_from_local_checkpoint,
    update_flavor_conf_to_persist_pretrained_model,
)
from mlflow.transformers.hub_utils import (
    is_valid_hf_repo_id,
)
from mlflow.transformers.llm_inference_utils import (
    _LLM_INFERENCE_TASK_CHAT,
    _LLM_INFERENCE_TASK_COMPLETIONS,
    _LLM_INFERENCE_TASK_EMBEDDING,
    _LLM_INFERENCE_TASK_KEY,
    _LLM_INFERENCE_TASK_PREFIX,
    _METADATA_LLM_INFERENCE_TASK_KEY,
    _SUPPORTED_LLM_INFERENCE_TASK_TYPES_BY_PIPELINE_TASK,
    _get_default_task_for_llm_inference_task,
    convert_messages_to_prompt,
    infer_signature_from_llm_inference_task,
    postprocess_output_for_llm_inference_task,
    postprocess_output_for_llm_v1_embedding_task,
    preprocess_llm_embedding_params,
    preprocess_llm_inference_input,
)
from mlflow.transformers.model_io import (
    _COMPONENTS_BINARY_DIR_NAME,
    _MODEL_BINARY_FILE_NAME,
    load_model_and_components_from_huggingface_hub,
    load_model_and_components_from_local,
    save_local_checkpoint,
    save_pipeline_pretrained_weights,
)
from mlflow.transformers.peft import (
    _PEFT_ADAPTOR_DIR_NAME,
    get_model_with_peft_adapter,
    get_peft_base_model,
    is_peft_model,
)
from mlflow.transformers.signature import (
    format_input_example_for_special_cases,
    infer_or_get_default_signature,
)
from mlflow.transformers.torch_utils import _TORCH_DTYPE_KEY, _deserialize_torch_dtype
from mlflow.types.utils import _validate_input_dictionary_contains_only_strings_and_lists_of_strings
from mlflow.utils import _truncate_and_ellipsize
from mlflow.utils.autologging_utils import (
    autologging_integration,
    disable_discrete_autologging,
    safe_patch,
)
from mlflow.utils.docstring_utils import (
    LOG_MODEL_PARAM_DOCS,
    docstring_version_compatibility_warning,
    format_docstring,
)
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
    infer_pip_requirements,
)
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.logging_utils import suppress_logs
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _download_artifact_from_uri,
    _get_flavor_configuration,
    _get_flavor_configuration_from_uri,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

# The following import is only used for type hinting
if TYPE_CHECKING:
    import torch
    from transformers import Pipeline

# Transformers pipeline complains that PeftModel is not supported for any task type, even
# when the wrapped model is supported. As MLflow require users to use pipeline for logging,
# we should suppress that confusing error message.
_PEFT_PIPELINE_ERROR_MSG = re.compile(r"The model 'PeftModel[^']*' is not supported for")

FLAVOR_NAME = "transformers"

_CARD_TEXT_FILE_NAME = "model_card.md"
_CARD_DATA_FILE_NAME = "model_card_data.yaml"
_INFERENCE_CONFIG_BINARY_KEY = "inference_config.txt"
_LICENSE_FILE_NAME = "LICENSE.txt"
_LICENSE_FILE_PATTERN = re.compile(r"license(\.[a-z]+|$)", re.IGNORECASE)

_SUPPORTED_RETURN_TYPES = {"pipeline", "components"}
# The default device id for CPU is -1 and GPU IDs are ordinal starting at 0, as documented here:
# https://huggingface.co/transformers/v4.7.0/main_classes/pipelines.html
_TRANSFORMERS_DEFAULT_CPU_DEVICE_ID = -1
_TRANSFORMERS_DEFAULT_GPU_DEVICE_ID = 0
_SUPPORTED_SAVE_KEYS = {
    FlavorKey.MODEL,
    FlavorKey.TOKENIZER,
    FlavorKey.FEATURE_EXTRACTOR,
    FlavorKey.IMAGE_PROCESSOR,
    FlavorKey.TORCH_DTYPE,
}

_SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES = {
    "feature-extraction",
    "fill-mask",
    "summarization",
    "text2text-generation",
    "text-generation",
}

_PROMPT_TEMPLATE_RETURN_FULL_TEXT_INFO = (
    "text-generation pipelines saved with prompt templates have the `return_full_text` "
    "pipeline kwarg set to False by default. To override this behavior, provide a "
    "`model_config` dict with `return_full_text` set to `True` when saving the model."
)


# Alias for the audio data types that Transformers pipeline (e.g. Whisper) expects.
# It can be one of:
#  1. A string representing the path or URL to an audio file.
#  2. A bytes object representing the raw audio data.
#  3. A float numpy array representing the audio time series.
AudioInput = str | bytes | np.ndarray

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(model) -> list[str]:
    """
    Args:
        model: The model instance to be saved in order to provide the required underlying
            deep learning execution framework dependency requirements. Note that this must
            be the actual model instance and not a Pipeline.

    Returns:
        A list of default pip requirements for MLflow Models that have been produced with the
        ``transformers`` flavor. Calls to :py:func:`save_model()` and :py:func:`log_model()`
        produce a pip environment that contain these requirements at a minimum.
    """
    packages = ["transformers"]

    try:
        engine = _get_engine_type(model)
        packages.append(engine)
    except Exception as e:
        packages += ["torch", "tensorflow"]
        _logger.warning(
            "Could not infer model execution engine type due to huggingface_hub not "
            "being installed or unable to connect in online mode. Adding both Pytorch"
            f"and Tensorflow to requirements.\nFailure cause: {e}"
        )

    if "torch" in packages:
        packages.append("torchvision")
        if importlib.util.find_spec("accelerate"):
            packages.append("accelerate")

    if is_peft_model(model):
        packages.append("peft")

    return [_get_pinned_requirement(module) for module in packages]


def _validate_transformers_model_dict(transformers_model):
    """
    Validator for a submitted save dictionary for the transformers model. If any additional keys
    are provided, raise to indicate which invalid keys were submitted.
    """
    if isinstance(transformers_model, dict):
        invalid_keys = [key for key in transformers_model.keys() if key not in _SUPPORTED_SAVE_KEYS]
        if invalid_keys:
            raise MlflowException(
                "Invalid dictionary submitted for 'transformers_model'. The "
                f"key(s) {invalid_keys} are not permitted. Must be one of: "
                f"{_SUPPORTED_SAVE_KEYS}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if FlavorKey.MODEL not in transformers_model:
            raise MlflowException(
                f"The 'transformers_model' dictionary must have an entry for {FlavorKey.MODEL}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        model = transformers_model[FlavorKey.MODEL]
    else:
        model = transformers_model.model
    if not hasattr(model, "name_or_path"):
        raise MlflowException(
            f"The submitted model type {type(model).__name__} does not inherit "
            "from a transformers pre-trained model. It is missing the attribute "
            "'name_or_path'. Please verify that the model is a supported "
            "transformers model.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def get_default_conda_env(model):
    """
    Returns:
        The default Conda environment for MLflow Models produced with the ``transformers``
        flavor, based on the model instance framework type of the model to be logged.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(model))


class _DummyModel(NamedTuple):
    name_or_path: str


class _DummyPipeline(NamedTuple):
    task: str
    model: _DummyModel


@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    transformers_model,
    path: str,
    processor=None,
    task: str | None = None,
    torch_dtype: torch.dtype | None = None,
    model_card=None,
    code_paths: list[str] | None = None,
    mlflow_model: Model | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    pip_requirements: list[str] | str | None = None,
    extra_pip_requirements: list[str] | str | None = None,
    conda_env=None,
    metadata: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    save_pretrained: bool = True,
    **kwargs,  # pylint: disable=unused-argument
) -> None:
    """
    Save a trained transformers model to a path on the local file system. Note that
    saving transformers models with custom code (i.e. models that require
    ``trust_remote_code=True``) requires ``transformers >= 4.26.0``.

    Args:
        transformers_model:
            The transformers model to save. This can be one of the following format:

                1. A transformers `Pipeline` instance.
                2. A dictionary that maps required components of a pipeline to the named keys
                    of ["model", "image_processor", "tokenizer", "feature_extractor"].
                    The `model` key in the dictionary must map to a value that inherits from
                    `PreTrainedModel`, `TFPreTrainedModel`, or `FlaxPreTrainedModel`.
                    All other component entries in the dictionary must support the defined task
                    type that is associated with the base model type configuration.
                3. A string that represents a path to a local/DBFS directory containing a model
                    checkpoint. The directory must contain a `config.json` file that is required
                    for loading the transformers model. This is particularly useful when logging
                    a model that cannot be loaded into memory for serialization.

            An example of specifying a `Pipeline` from a default pipeline instantiation:

            .. code-block:: python

                from transformers import pipeline

                qa_pipe = pipeline("question-answering", "csarron/mobilebert-uncased-squad-v2")

                with mlflow.start_run():
                    mlflow.transformers.save_model(
                        transformers_model=qa_pipe,
                        path="path/to/save/model",
                    )

            An example of specifying component-level parts of a transformers model is shown below:

            .. code-block:: python

                from transformers import MobileBertForQuestionAnswering, AutoTokenizer

                architecture = "csarron/mobilebert-uncased-squad-v2"
                tokenizer = AutoTokenizer.from_pretrained(architecture)
                model = MobileBertForQuestionAnswering.from_pretrained(architecture)

                with mlflow.start_run():
                    components = {
                        "model": model,
                        "tokenizer": tokenizer,
                    }
                    mlflow.transformers.save_model(
                        transformers_model=components,
                        path="path/to/save/model",
                    )

            An example of specifying a local checkpoint path is shown below:

            .. code-block:: python

                with mlflow.start_run():
                    mlflow.transformers.save_model(
                        transformers_model="path/to/local/checkpoint",
                        path="path/to/save/model",
                    )

        path: Local path destination for the serialized model to be saved.
        processor: An optional ``Processor`` subclass object. Some model architectures,
            particularly multi-modal types, utilize Processors to combine text
            encoding and image or audio encoding in a single entrypoint.

            .. Note:: If a processor is supplied when saving a model, the
                        model will be unavailable for loading as a ``Pipeline`` or for
                        usage with pyfunc inference.
        task: The transformers-specific task type of the model, or MLflow inference task type.
            If provided a transformers-specific task type, these strings are utilized so
            that a pipeline can be created with the appropriate internal call architecture
            to meet the needs of a given model.
            If this argument is provided as a inference task type or not specified, the
            pipeline utilities within the transformers library will be used to infer the
            correct task type. If the value specified is not a supported type,
            an Exception will be thrown.
        torch_dtype: The Pytorch dtype applied to the model when loading back. This is useful
            when you want to save the model with a specific dtype that is different from the
            dtype of the model when it was trained. If not specified, the current dtype of the
            model instance will be used.
        model_card: An Optional `ModelCard` instance from `huggingface-hub`. If provided, the
            contents of the model card will be saved along with the provided
            `transformers_model`. If not provided, an attempt will be made to fetch
            the card from the base pretrained model that is provided (or the one that is
            included within a provided `Pipeline`).

            .. Note:: In order for a ModelCard to be fetched (if not provided),
                        the huggingface_hub package must be installed and the version
                        must be >=0.10.0

        code_paths: {{ code_paths }}
        mlflow_model: An MLflow model object that specifies the flavor that this model is being
            added to.
        signature: A Model Signature object that describes the input and output Schema of the
            model. The model signature can be inferred using `infer_signature` function
            of `mlflow.models.signature`.

            .. code-block:: python
                :caption: Example

                from mlflow.models import infer_signature
                from mlflow.transformers import generate_signature_output
                from transformers import pipeline

                en_to_de = pipeline("translation_en_to_de")

                data = "MLflow is great!"
                output = generate_signature_output(en_to_de, data)
                signature = infer_signature(data, output)

                mlflow.transformers.save_model(
                    transformers_model=en_to_de,
                    path="/path/to/save/model",
                    signature=signature,
                    input_example=data,
                )

                loaded = mlflow.pyfunc.load_model("/path/to/save/model")
                print(loaded.predict(data))
                # MLflow ist großartig!

            If an input_example is provided and the signature is not, a signature will
            be inferred automatically and applied to the MLmodel file iff the
            pipeline type is a text-based model (NLP). If the pipeline type is not
            a supported type, this inference functionality will not function correctly
            and a warning will be issued. In order to ensure that a precise signature
            is logged, it is recommended to explicitly provide one.
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        conda_env: {{ conda_env }}
        metadata: {{ metadata }}
        model_config:
            A dict of valid overrides that can be applied to a pipeline instance during inference.
            These arguments are used exclusively for the case of loading the model as a ``pyfunc``
            Model or for use in Spark.
            These values are not applied to a returned Pipeline from a call to
            ``mlflow.transformers.load_model()``

            .. Warning:: If the key provided is not compatible with either the
                    Pipeline instance for the task provided or is not a valid
                    override to any arguments available in the Model, an
                    Exception will be raised at runtime. It is very important
                    to validate the entries in this dictionary to ensure
                    that they are valid prior to saving or logging.

            An example of providing overrides for a question generation model:

            .. code-block:: python

                from transformers import pipeline, AutoTokenizer

                task = "text-generation"
                architecture = "gpt2"

                sentence_pipeline = pipeline(
                    task=task,
                    tokenizer=AutoTokenizer.from_pretrained(architecture),
                    model=architecture,
                )

                # Validate that the overrides function
                prompts = ["Generative models are", "I'd like a coconut so that I can"]

                # validation of config prior to save or log
                model_config = {
                    "top_k": 2,
                    "num_beams": 5,
                    "max_length": 30,
                    "temperature": 0.62,
                    "top_p": 0.85,
                    "repetition_penalty": 1.15,
                }

                # Verify that no exceptions are thrown
                sentence_pipeline(prompts, **model_config)

                mlflow.transformers.save_model(
                    transformers_model=sentence_pipeline,
                    path="/path/for/model",
                    task=task,
                    model_config=model_config,
                )
        prompt_template: {{ prompt_template }}
        save_pretrained: {{ save_pretrained }}
        kwargs: Optional additional configurations for transformers serialization.

    """
    import transformers

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = pathlib.Path(path).absolute()

    _validate_and_prepare_target_save_path(str(path))

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, str(path))

    if isinstance(transformers_model, transformers.Pipeline):
        _validate_transformers_model_dict(transformers_model)
        built_pipeline = transformers_model
    elif isinstance(transformers_model, dict):
        _validate_transformers_model_dict(transformers_model)
        built_pipeline = _build_pipeline_from_model_input(transformers_model, task=task)
    elif isinstance(transformers_model, str):
        # When a string is passed, it should be a path to model checkpoint in local storage or DBFS
        if transformers_model.startswith("dbfs:"):
            # Replace the DBFS URI to the actual mount point
            transformers_model = transformers_model.replace("dbfs:", "/dbfs", 1)

        if task is None:
            raise MlflowException(
                "The `task` argument must be specified when logging a model from a local "
                "checkpoint. Please provide the task type of the pipeline.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if not save_pretrained:
            raise MlflowException(
                "The `save_pretrained` argument must be set to True when logging a model from a "
                "local checkpoint. Please set `save_pretrained=True`.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Create a dummy pipeline object to be used for saving the model
        built_pipeline = _DummyPipeline(
            task=task, model=_DummyModel(name_or_path=transformers_model)
        )
    else:
        raise MlflowException(
            "The `transformers_model` must be one of the following types: \n"
            " (1) a transformers Pipeline\n"
            " (2) a dictionary of components for a transformers Pipeline\n"
            " (3) a path to a local/DBFS directory containing a transformers model checkpoint.\n"
            f"received: {type(transformers_model)}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # Verify that the model has not been loaded to distributed memory
    # NB: transformers does not correctly save a model whose weights have been loaded
    # using accelerate iff the model weights have been loaded using a device_map that is
    # heterogeneous. There is a distinct possibility for a partial write to occur, causing an
    # invalid state of the model's weights in this scenario. Hence, we raise.
    # We might be able to remove this check once this PR is merged to transformers:
    # https://github.com/huggingface/transformers/issues/20072
    if _is_model_distributed_in_memory(built_pipeline.model):
        raise MlflowException(
            "The model that is attempting to be saved has been loaded into memory "
            "with an incompatible configuration. If you are using the accelerate "
            "library to load your model, please ensure that it is saved only after "
            "loading with the default device mapping. Do not specify `device_map` "
            "and please try again."
        )

    if mlflow_model is None:
        mlflow_model = Model()

    if task and task.startswith(_LLM_INFERENCE_TASK_PREFIX):
        llm_inference_task = task

        # For local checkpoint saving, we set built_pipeline.task to the original `task`
        # argument value earlier, which is LLM v1 task. Thereby here we update it to the
        # corresponding Transformers task type.
        if isinstance(transformers_model, str):
            default_task = _get_default_task_for_llm_inference_task(llm_inference_task)
            built_pipeline = built_pipeline._replace(task=default_task)

        _validate_llm_inference_task_type(llm_inference_task, built_pipeline.task)
    else:
        llm_inference_task = None

    if llm_inference_task:
        mlflow_model.signature = infer_signature_from_llm_inference_task(
            llm_inference_task, signature
        )
    elif signature is not None:
        mlflow_model.signature = signature

    if input_example is not None:
        input_example = format_input_example_for_special_cases(input_example, built_pipeline)
        _save_example(mlflow_model, input_example, str(path))

    if metadata is not None:
        mlflow_model.metadata = metadata

    # Check task consistency between model metadata and task argument
    #  NB: Using mlflow_model.metadata instead of passed metadata argument directly, because
    #  metadata argument is not directly propagated from log_model() to save_model(), instead
    #  via the mlflow_model object attribute.
    if (
        mlflow_model.metadata is not None
        and (metadata_task := mlflow_model.metadata.get(_METADATA_LLM_INFERENCE_TASK_KEY))
        and metadata_task != task
    ):
        raise MlflowException(
            f"LLM v1 task type '{metadata_task}' is specified in "
            "metadata, but it doesn't match the task type provided in the `task` argument: "
            f"'{task}'. The mismatched task type may cause incorrect model inference behavior. "
            "Please provide the correct LLM v1 task type in the `task` argument. E.g. "
            f'`mlflow.transformers.save_model(task="{metadata_task}", ...)`',
            error_code=INVALID_PARAMETER_VALUE,
        )

    if prompt_template is not None:
        # prevent saving prompt templates for unsupported pipeline types
        if built_pipeline.task not in _SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES:
            raise MlflowException(
                f"Prompt templating is not supported for the `{built_pipeline.task}` task type. "
                f"Supported task types are: {_SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES}."
            )

        _validate_prompt_template(prompt_template)
        if mlflow_model.metadata:
            mlflow_model.metadata[FlavorKey.PROMPT_TEMPLATE] = prompt_template
        else:
            mlflow_model.metadata = {FlavorKey.PROMPT_TEMPLATE: prompt_template}

    if is_peft_model(built_pipeline.model):
        _logger.info(
            "Overriding save_pretrained to False for PEFT models, following the Transformers "
            "behavior. The PEFT adaptor and config will be saved, but the base model weights "
            "will not and reference to the HuggingFace Hub repository will be logged instead."
        )
        # This will only save PEFT adaptor weights and config, not the base model weights
        built_pipeline.model.save_pretrained(path.joinpath(_PEFT_ADAPTOR_DIR_NAME))
        save_pretrained = False

    if not save_pretrained and not is_valid_hf_repo_id(built_pipeline.model.name_or_path):
        _logger.warning(
            "The save_pretrained parameter is set to False, but the specified model does not "
            "have a valid HuggingFace Hub repository identifier. Therefore, the weights will "
            "be saved to disk anyway."
        )
        save_pretrained = True

    # Create the flavor configuration
    if isinstance(transformers_model, str):
        flavor_conf = build_flavor_config_from_local_checkpoint(
            transformers_model, built_pipeline.task, processor, torch_dtype
        )
    else:
        flavor_conf = build_flavor_config(built_pipeline, processor, torch_dtype, save_pretrained)

    if llm_inference_task:
        flavor_conf.update({_LLM_INFERENCE_TASK_KEY: llm_inference_task})
        if mlflow_model.metadata:
            mlflow_model.metadata[_METADATA_LLM_INFERENCE_TASK_KEY] = llm_inference_task
        else:
            mlflow_model.metadata = {_METADATA_LLM_INFERENCE_TASK_KEY: llm_inference_task}

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        transformers_version=transformers.__version__,
        code=code_dir_subpath,
        **flavor_conf,
    )

    # Flavor config should not be mutated after being added to MLModel
    flavor_conf = MappingProxyType(flavor_conf)

    # Save pipeline model and components weights
    if save_pretrained:
        if isinstance(transformers_model, str):
            save_local_checkpoint(path, transformers_model, flavor_conf, processor)
        else:
            save_pipeline_pretrained_weights(path, built_pipeline, flavor_conf, processor)
    else:
        repo = built_pipeline.model.name_or_path
        _logger.info(
            "Skipping saving pretrained model weights to disk as the save_pretrained argument"
            f"is set to False. The reference to the HuggingFace Hub repository {repo} "
            "will be logged instead."
        )

    model_name = built_pipeline.model.name_or_path

    # Get the model card from either the argument or the HuggingFace marketplace
    card_data = model_card or _fetch_model_card(model_name)

    # If the card data can be acquired, save the text and the data separately
    _write_card_data(card_data, path)

    # Write the license information (or guidance) along with the model
    _write_license_information(model_name, card_data, path)

    # Only allow a subset of task types to have a pyfunc definition.
    # Currently supported types are NLP-based language tasks which have a pipeline definition
    # consisting exclusively of a Model and a Tokenizer.
    if (
        # TODO: when a local checkpoint path is provided as a model, we assume it is eligible
        # for pyfunc prediction. This may not be true for all cases, so we should revisit this.
        isinstance(transformers_model, str) or _should_add_pyfunc_to_model(built_pipeline)
    ):
        if mlflow_model.signature is None:
            mlflow_model.signature = infer_or_get_default_signature(
                pipeline=built_pipeline,
                example=input_example,
                model_config=model_config,
                flavor_config=flavor_conf,
            )

        # if pipeline is text-generation and a prompt template is specified,
        # provide the return_full_text=False config by default to avoid confusing
        # extra text for end-users
        if prompt_template is not None and built_pipeline.task == "text-generation":
            return_full_text_key = "return_full_text"
            model_config = model_config or {}
            if return_full_text_key not in model_config:
                model_config[return_full_text_key] = False
                _logger.info(_PROMPT_TEMPLATE_RETURN_FULL_TEXT_INFO)

        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.transformers",
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_dir_subpath,
            model_config=model_config,
        )
    else:
        if processor:
            reason = "the model has been saved with a 'processor' argument supplied."
        else:
            reason = (
                "the model is not a language-based model and requires a complex input type "
                "that is currently not supported."
            )
        _logger.warning(
            f"This model is unable to be used for pyfunc prediction because {reason} "
            f"The pyfunc flavor will not be added to the Model."
        )

    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size

    mlflow_model.save(str(path.joinpath(MLMODEL_FILE_NAME)))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(built_pipeline.model)
            if isinstance(transformers_model, str) or is_peft_model(built_pipeline.model):
                _logger.info(
                    "A local checkpoint path or PEFT model is given as the `transformers_model`. "
                    "To avoid loading the full model into memory, we don't infer the pip "
                    "requirement for the model. Instead, we will use the default requirements, "
                    "but it may not capture all required pip libraries for the model. Consider "
                    "providing the pip requirements explicitly."
                )
            else:
                # Infer the pip requirements with a timeout to avoid hanging at prediction
                inferred_reqs = infer_pip_requirements(
                    model_uri=str(path),
                    flavor=FLAVOR_NAME,
                    fallback=default_reqs,
                    timeout=MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT.get(),
                )
                default_reqs = set(inferred_reqs).union(default_reqs)
            default_reqs = sorted(default_reqs)
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with path.joinpath(_CONDA_ENV_FILE_NAME).open("w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(str(path.joinpath(_CONSTRAINTS_FILE_NAME)), "\n".join(pip_constraints))

    write_to(str(path.joinpath(_REQUIREMENTS_FILE_NAME)), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(str(path.joinpath(_PYTHON_ENV_FILE_NAME)))


@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    transformers_model,
    artifact_path: str | None = None,
    processor=None,
    task: str | None = None,
    torch_dtype: torch.dtype | None = None,
    model_card=None,
    code_paths: list[str] | None = None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: list[str] | str | None = None,
    extra_pip_requirements: list[str] | str | None = None,
    conda_env=None,
    metadata: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    save_pretrained: bool = True,
    prompts: list[str | Prompt] | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    **kwargs,
):
    """
    Log a ``transformers`` object as an MLflow artifact for the current run. Note that
    logging transformers models with custom code (i.e. models that require
    ``trust_remote_code=True``) requires ``transformers >= 4.26.0``.

    Args:
        transformers_model:
            The transformers model to save. This can be one of the following format:

                1. A transformers `Pipeline` instance.
                2. A dictionary that maps required components of a pipeline to the named keys
                    of ["model", "image_processor", "tokenizer", "feature_extractor"].
                    The `model` key in the dictionary must map to a value that inherits from
                    `PreTrainedModel`, `TFPreTrainedModel`, or `FlaxPreTrainedModel`.
                    All other component entries in the dictionary must support the defined task
                    type that is associated with the base model type configuration.
                3. A string that represents a path to a local/DBFS directory containing a model
                    checkpoint. The directory must contain a `config.json` file that is required
                    for loading the transformers model. This is particularly useful when logging
                    a model that cannot be loaded into memory for serialization.

            An example of specifying a `Pipeline` from a default pipeline instantiation:

            .. code-block:: python

                from transformers import pipeline

                qa_pipe = pipeline("question-answering", "csarron/mobilebert-uncased-squad-v2")

                with mlflow.start_run():
                    mlflow.transformers.log_model(
                        transformers_model=qa_pipe,
                        name="model",
                    )

            An example of specifying component-level parts of a transformers model is shown below:

            .. code-block:: python

                from transformers import MobileBertForQuestionAnswering, AutoTokenizer

                architecture = "csarron/mobilebert-uncased-squad-v2"
                tokenizer = AutoTokenizer.from_pretrained(architecture)
                model = MobileBertForQuestionAnswering.from_pretrained(architecture)

                with mlflow.start_run():
                    components = {
                        "model": model,
                        "tokenizer": tokenizer,
                    }
                    mlflow.transformers.log_model(
                        transformers_model=components,
                        name="model",
                    )

            An example of specifying a local checkpoint path is shown below:

            .. code-block:: python

                with mlflow.start_run():
                    mlflow.transformers.log_model(
                        transformers_model="path/to/local/checkpoint",
                        name="model",
                    )

        artifact_path: Deprecated. Use `name` instead.
        processor: An optional ``Processor`` subclass object. Some model architectures,
            particularly multi-modal types, utilize Processors to combine text
            encoding and image or audio encoding in a single entrypoint.

                .. Note:: If a processor is supplied when logging a model, the
                    model will be unavailable for loading as a ``Pipeline`` or for usage
                    with pyfunc inference.
        task: The transformers-specific task type of the model. These strings are utilized so
            that a pipeline can be created with the appropriate internal call architecture
            to meet the needs of a given model. If this argument is not specified, the
            pipeline utilities within the transformers library will be used to infer the
            correct task type. If the value specified is not a supported type within the
            version of transformers that is currently installed, an Exception will be thrown.
        torch_dtype: The Pytorch dtype applied to the model when loading back. This is useful
            when you want to save the model with a specific dtype that is different from the
            dtype of the model when it was trained. If not specified, the current dtype of the
            model instance will be used.
        model_card: An Optional `ModelCard` instance from `huggingface-hub`. If provided, the
            contents of the model card will be saved along with the provided
            `transformers_model`. If not provided, an attempt will be made to fetch
            the card from the base pretrained model that is provided (or the one that is
            included within a provided `Pipeline`).

                .. Note:: In order for a ModelCard to be fetched (if not provided),
                    the huggingface_hub package must be installed and the version
                    must be >=0.10.0

        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: A Model Signature object that describes the input and output Schema of the
            model. The model signature can be inferred using `infer_signature` function
            of `mlflow.models.signature`.

            .. code-block:: python
                :caption: Example

                from mlflow.models import infer_signature
                from mlflow.transformers import generate_signature_output
                from transformers import pipeline

                en_to_de = pipeline("translation_en_to_de")

                data = "MLflow is great!"
                output = generate_signature_output(en_to_de, data)
                signature = infer_signature(data, output)

                with mlflow.start_run() as run:
                    mlflow.transformers.log_model(
                        transformers_model=en_to_de,
                        name="english_to_german_translator",
                        signature=signature,
                        input_example=data,
                    )

                model_uri = f"runs:/{run.info.run_id}/english_to_german_translator"
                loaded = mlflow.pyfunc.load_model(model_uri)

                print(loaded.predict(data))
                # MLflow ist großartig!

            If an input_example is provided and the signature is not, a signature will
            be inferred automatically and applied to the MLmodel file iff the
            pipeline type is a text-based model (NLP). If the pipeline type is not
            a supported type, this inference functionality will not function correctly
            and a warning will be issued. In order to ensure that a precise signature
            is logged, it is recommended to explicitly provide one.
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version
            to finish being created and is in ``READY`` status.
            By default, the function waits for five minutes.
            Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        conda_env: {{ conda_env }}
        metadata: {{ metadata }}
        model_config:
            A dict of valid overrides that can be applied to a pipeline instance during inference.
            These arguments are used exclusively for the case of loading the model as a ``pyfunc``
            Model or for use in Spark. These values are not applied to a returned Pipeline from a
            call to ``mlflow.transformers.load_model()``

            .. Warning:: If the key provided is not compatible with either the
                         Pipeline instance for the task provided or is not a valid
                         override to any arguments available in the Model, an
                         Exception will be raised at runtime. It is very important
                         to validate the entries in this dictionary to ensure
                         that they are valid prior to saving or logging.

            An example of providing overrides for a question generation model:

            .. code-block:: python

                from transformers import pipeline, AutoTokenizer

                task = "text-generation"
                architecture = "gpt2"

                sentence_pipeline = pipeline(
                    task=task,
                    tokenizer=AutoTokenizer.from_pretrained(architecture),
                    model=architecture,
                )

                # Validate that the overrides function
                prompts = ["Generative models are", "I'd like a coconut so that I can"]

                # validation of config prior to save or log
                model_config = {
                    "top_k": 2,
                    "num_beams": 5,
                    "max_length": 30,
                    "temperature": 0.62,
                    "top_p": 0.85,
                    "repetition_penalty": 1.15,
                }

                # Verify that no exceptions are thrown
                sentence_pipeline(prompts, **model_config)

                with mlflow.start_run():
                    mlflow.transformers.log_model(
                        transformers_model=sentence_pipeline,
                        name="my_sentence_generator",
                        task=task,
                        model_config=model_config,
                    )
        prompt_template: {{ prompt_template }}
        save_pretrained: {{ save_pretrained }}
        prompts: {{ prompts }}
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        kwargs: Additional arguments for :py:class:`mlflow.models.model.Model`
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=sys.modules[__name__],  # Get the current module.
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        metadata=metadata,
        transformers_model=transformers_model,
        processor=processor,
        task=task,
        torch_dtype=torch_dtype,
        model_card=model_card,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        # NB: We don't validate the serving input if the provided model is a path
        # to a local checkpoint. This is because the purpose of supporting that
        # input format is to avoid loading large model into memory. Serving input
        # validation loads the model into memory and make prediction, which is
        # expensive and can cause OOM errors.
        validate_serving_input=not isinstance(transformers_model, str),
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        model_config=model_config,
        prompt_template=prompt_template,
        save_pretrained=save_pretrained,
        prompts=prompts,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
def load_model(
    model_uri: str, dst_path: str | None = None, return_type="pipeline", device=None, **kwargs
):
    """
    Load a ``transformers`` object from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to utilize for downloading the model artifact.
            This directory must already exist if provided. If unspecified, a local output
            path will be created.
        return_type: A return type modifier for the stored ``transformers`` object.
            If set as "components", the return type will be a dictionary of the saved
            individual components of either the ``Pipeline`` or the pre-trained model.
            The components for NLP-focused models will typically consist of a
            return representation as shown below with a text-classification example:

            .. code-block:: python

                {"model": BertForSequenceClassification, "tokenizer": BertTokenizerFast}

            Vision models will return an ``ImageProcessor`` instance of the appropriate
            type, while multi-modal models will return both a ``FeatureExtractor`` and
            a ``Tokenizer`` along with the model.
            Returning "components" can be useful for certain model types that do not
            have the desired pipeline return types for certain use cases.
            If set as "pipeline", the model, along with any and all required
            ``Tokenizer``, ``FeatureExtractor``, ``Processor``, or ``ImageProcessor``
            objects will be returned within a ``Pipeline`` object of the appropriate
            type defined by the ``task`` set by the model instance type. To override
            this behavior, supply a valid ``task`` argument during model logging or
            saving. Default is "pipeline".
        device: The device on which to load the model. Default is None. Use 0 to
            load to the default GPU.
        kwargs: Optional configuration options for loading of a ``transformers`` object.
            For information on parameters and their usage, see
            `transformers documentation <https://huggingface.co/docs/transformers/index>`_.

    Returns:
        A ``transformers`` model instance or a dictionary of components
    """

    if return_type not in _SUPPORTED_RETURN_TYPES:
        raise MlflowException(
            f"The specified return_type mode '{return_type}' is unsupported. "
            "Please select one of: 'pipeline' or 'components'.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    model_uri = str(model_uri)

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)

    flavor_config = _get_flavor_configuration_from_uri(model_uri, FLAVOR_NAME, _logger)

    if return_type == "pipeline" and FlavorKey.PROCESSOR_TYPE in flavor_config:
        raise MlflowException(
            "This model has been saved with a processor. Processor objects are "
            "not compatible with Pipelines. Please load this model by specifying "
            "the 'return_type'='components'.",
            error_code=BAD_REQUEST,
        )

    _add_code_from_conf_to_system_path(local_model_path, flavor_config)

    return _load_model(local_model_path, flavor_config, return_type, device, **kwargs)


def persist_pretrained_model(model_uri: str) -> None:
    """
    Persist Transformers pretrained model weights to the artifacts directory of the specified
    model_uri. This API is primary used for updating an MLflow Model that was logged or saved
    with setting save_pretrained=False. Such models cannot be registered to Databricks Workspace
    Model Registry, due to the full pretrained model weights being absent in the artifacts.
    Transformers models saved in this mode store only the reference to the HuggingFace Hub
    repository. This API will download the model weights from the HuggingFace Hub repository
    and save them in the artifacts of the given model_uri so that the model can be registered
    to Databricks Workspace Model Registry.

    Args:
        model_uri: The URI of the existing MLflow Model of the Transformers flavor.
            It must be logged/saved with save_pretrained=False.

    Examples:

    .. code-block:: python

        import mlflow

        # Saving a model with save_pretrained=False
        with mlflow.start_run() as run:
            model = pipeline("question-answering", "csarron/mobilebert-uncased-squad-v2")
            mlflow.transformers.log_model(
                transformers_model=model, name="pipeline", save_pretrained=False
            )

        # The model cannot be registered to the Model Registry as it is
        try:
            mlflow.register_model(f"runs:/{run.info.run_id}/pipeline", "qa_pipeline")
        except MlflowException as e:
            print(e.message)

        # Use this API to persist the pretrained model weights
        mlflow.transformers.persist_pretrained_model(f"runs:/{run.info.run_id}/pipeline")

        # Now the model can be registered to the Model Registry
        mlflow.register_model(f"runs:/{run.info.run_id}/pipeline", "qa_pipeline")
    """
    # Check if the model weight already exists in the model artifact before downloading
    root_uri, artifact_path = _get_root_uri_and_artifact_path(model_uri)
    artifact_repo = get_artifact_repository(root_uri)

    file_names = [os.path.basename(f.path) for f in artifact_repo.list_artifacts(artifact_path)]
    if MLMODEL_FILE_NAME in file_names and _MODEL_BINARY_FILE_NAME in file_names:
        _logger.info(
            "The full pretrained model weight already exists in the artifact directory of the "
            f"specified model_uri: {model_uri}. No action is needed."
        )
        return

    with TempDir() as tmp_dir:
        local_model_path = artifact_repo.download_artifacts(artifact_path, dst_path=tmp_dir.path())
        pipeline = load_model(local_model_path, return_type="pipeline")

        # Update MLModel flavor config
        mlmodel_path = os.path.join(local_model_path, MLMODEL_FILE_NAME)
        model_conf = Model.load(mlmodel_path)
        updated_flavor_conf = update_flavor_conf_to_persist_pretrained_model(
            model_conf.flavors[FLAVOR_NAME]
        )
        model_conf.add_flavor(FLAVOR_NAME, **updated_flavor_conf)
        model_conf.save(mlmodel_path)

        # Save pretrained weights
        save_pipeline_pretrained_weights(
            pathlib.Path(local_model_path), pipeline, updated_flavor_conf
        )

        # Upload updated local artifacts to MLflow
        for dir_to_upload in (_MODEL_BINARY_FILE_NAME, _COMPONENTS_BINARY_DIR_NAME):
            local_dir = os.path.join(local_model_path, dir_to_upload)
            if not os.path.isdir(local_dir):
                continue

            try:
                artifact_repo.log_artifacts(local_dir, os.path.join(artifact_path, dir_to_upload))
            except Exception as e:
                # NB: log_artifacts method doesn't support rollback for partial uploads,
                raise MlflowException(
                    f"Failed to upload {local_dir} to the existing model_uri due to {e}."
                    "Some other files may have been uploaded."
                ) from e

        # Upload MLModel file
        artifact_repo.log_artifact(mlmodel_path, artifact_path)

    _logger.info(f"The pretrained model has been successfully persisted in {model_uri}.")


def _is_model_distributed_in_memory(transformers_model):
    """Check if the model is distributed across multiple devices in memory."""

    # Check if the model attribute exists. If not, accelerate was not used and the model can
    # be safely saved
    if not hasattr(transformers_model, "hf_device_map"):
        return False
    # If the device map has more than one unique value entry, then the weights are not within
    # a contiguous memory system (VRAM, SYS, or DISK) and thus cannot be safely saved.
    return len(set(transformers_model.hf_device_map.values())) > 1


# This function attempts to determine if a GPU is available for the PyTorch and TensorFlow libraries
def is_gpu_available():
    # try pytorch and if it fails, try tf
    is_gpu = None
    try:
        import torch

        is_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    if is_gpu is None:
        try:
            import tensorflow as tf

            is_gpu = tf.test.is_gpu_available()
        except ImportError:
            pass
    if is_gpu is None:
        is_gpu = False
    return is_gpu


def _load_model(path: str, flavor_config, return_type: str, device=None, **kwargs):
    """
    Loads components from a locally serialized ``Pipeline`` object.
    """
    import transformers

    conf = {
        "task": flavor_config[FlavorKey.TASK],
    }
    if framework := flavor_config.get(FlavorKey.FRAMEWORK):
        conf["framework"] = framework

    # Note that we don't set the device in the conf yet because device is
    # incompatible with device_map.
    accelerate_model_conf = {}
    if MLFLOW_HUGGINGFACE_USE_DEVICE_MAP.get():
        device_map_strategy = MLFLOW_HUGGINGFACE_DEVICE_MAP_STRATEGY.get()
        conf["device_map"] = device_map_strategy
        accelerate_model_conf["device_map"] = device_map_strategy
        # Cannot use device with device_map
        if device is not None:
            raise MlflowException.invalid_parameter_value(
                "The environment variable MLFLOW_HUGGINGFACE_USE_DEVICE_MAP is set to True, but "
                f"the `device` argument is provided with value {device}. The device_map and "
                "`device` argument cannot be used together. Set MLFLOW_HUGGINGFACE_USE_DEVICE_MAP "
                "to False to specify a particular device ID, or pass None for the `device` "
                "argument to use device_map."
            )
        device = None
    elif device is None:
        if device_value := MLFLOW_DEFAULT_PREDICTION_DEVICE.get():
            try:
                device = int(device_value)
            except ValueError:
                _logger.warning(
                    f"Invalid value for {MLFLOW_DEFAULT_PREDICTION_DEVICE}: {device_value}. "
                    f"{MLFLOW_DEFAULT_PREDICTION_DEVICE} value must be an integer. "
                    f"Setting to: {_TRANSFORMERS_DEFAULT_CPU_DEVICE_ID}."
                )
                device = _TRANSFORMERS_DEFAULT_CPU_DEVICE_ID
        elif is_gpu_available():
            device = _TRANSFORMERS_DEFAULT_GPU_DEVICE_ID

    if device is not None:
        conf["device"] = device
        accelerate_model_conf["device"] = device

    if dtype_val := kwargs.get(_TORCH_DTYPE_KEY) or flavor_config.get(FlavorKey.TORCH_DTYPE):
        if isinstance(dtype_val, str):
            dtype_val = _deserialize_torch_dtype(dtype_val)
        conf[_TORCH_DTYPE_KEY] = dtype_val
        flavor_config[_TORCH_DTYPE_KEY] = dtype_val
        accelerate_model_conf[_TORCH_DTYPE_KEY] = dtype_val

    accelerate_model_conf["low_cpu_mem_usage"] = MLFLOW_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE.get()

    # Load model and components either from local or from HuggingFace Hub. We check for the
    # presence of the model revision (a commit hash of the hub repository) that is only present
    # in the model logged with `save_pretrained=False
    if FlavorKey.MODEL_REVISION not in flavor_config:
        model_and_components = load_model_and_components_from_local(
            path=pathlib.Path(path),
            flavor_conf=flavor_config,
            accelerate_conf=accelerate_model_conf,
            device=device,
        )
    else:
        model_and_components = load_model_and_components_from_huggingface_hub(
            flavor_conf=flavor_config, accelerate_conf=accelerate_model_conf, device=device
        )

    # Load and apply PEFT adaptor if saved
    if peft_adapter_dir := flavor_config.get(FlavorKey.PEFT, None):
        model_and_components[FlavorKey.MODEL] = get_model_with_peft_adapter(
            base_model=model_and_components[FlavorKey.MODEL],
            peft_adapter_path=os.path.join(path, peft_adapter_dir),
        )

    conf = {**conf, **model_and_components}

    if return_type == "pipeline":
        conf.update(**kwargs)
        with suppress_logs("transformers.pipelines.base", filter_regex=_PEFT_PIPELINE_ERROR_MSG):
            return transformers.pipeline(**conf)
    elif return_type == "components":
        return conf


def _fetch_model_card(model_name):
    """
    Attempts to retrieve the model card for the specified model architecture iff the
    `huggingface_hub` library is installed. If a card cannot be found in the registry or
    the library is not installed, returns None.
    """
    try:
        import huggingface_hub as hub
    except ImportError:
        _logger.warning(
            "Unable to store ModelCard data with the saved artifact. In order to "
            "preserve this information, please install the huggingface_hub package "
            "by running 'pip install huggingingface_hub>0.10.0'"
        )
        return

    if hasattr(hub, "ModelCard"):
        try:
            return hub.ModelCard.load(model_name)
        except Exception as e:
            _logger.warning(f"The model card could not be retrieved from the hub due to {e}")
    else:
        _logger.warning(
            "The version of huggingface_hub that is installed does not provide "
            f"ModelCard functionality. You have version {hub.__version__} installed. "
            "Update huggingface_hub to >= '0.10.0' to retrieve the ModelCard data."
        )


def _write_card_data(card_data, path):
    """
    Writes the card data, if specified or available, to the provided path in two separate files
    """
    if card_data:
        try:
            path.joinpath(_CARD_TEXT_FILE_NAME).write_text(card_data.text, encoding="utf-8")
        except UnicodeError as e:
            _logger.warning(f"Unable to save the model card text due to: {e}")

        with path.joinpath(_CARD_DATA_FILE_NAME).open("w") as file:
            yaml.safe_dump(
                card_data.data.to_dict(), stream=file, default_flow_style=False, encoding="utf-8"
            )


def _extract_license_file_from_repository(model_name):
    """Returns the top-level file inventory of `RepoFile` objects from the huggingface hub"""
    try:
        import huggingface_hub as hub
    except ImportError:
        _logger.debug(
            f"Unable to list repository contents for the model repo {model_name}. In order "
            "to enable repository listing functionality, please install the huggingface_hub "
            "package by running `pip install huggingface_hub>0.10.0"
        )
        return
    try:
        files = hub.list_repo_files(model_name)
        return next(file for file in files if _LICENSE_FILE_PATTERN.search(file))
    except Exception as e:
        _logger.debug(
            f"Failed to retrieve repository file listing data for {model_name} due to {e}"
        )


def _write_license_information(model_name, card_data, path):
    """Writes the license file or instructions to retrieve license information."""

    fallback = (
        f"A license file could not be found for the '{model_name}' repository. \n"
        "To ensure that you are in compliance with the license requirements for this "
        f"model, please visit the model repository here: https://huggingface.co/{model_name}"
    )

    if license_file := _extract_license_file_from_repository(model_name):
        try:
            import huggingface_hub as hub

            license_location = hub.hf_hub_download(repo_id=model_name, filename=license_file)
        except Exception as e:
            _logger.warning(f"Failed to download the license file due to: {e}")
        else:
            local_license_path = pathlib.Path(license_location)
            target_path = path.joinpath(local_license_path.name)
            try:
                shutil.copy(local_license_path, target_path)
                return
            except Exception as e:
                _logger.warning(f"The license file could not be copied due to: {e}")

    # Fallback or card data license info
    if card_data and card_data.data.license != "other":
        fallback = f"{fallback}\nThe declared license type is: '{card_data.data.license}'"
    else:
        _logger.warning(
            "Unable to find license information for this model. Please verify "
            "permissible usage for the model you are storing prior to use."
        )
    path.joinpath(_LICENSE_FILE_NAME).write_text(fallback, encoding="utf-8")


def _get_supported_pretrained_model_types():
    """
    Users might not have all the necessary libraries installed to determine the supported model
    """

    supported_model_types = ()

    try:
        from transformers import FlaxPreTrainedModel

        supported_model_types += (FlaxPreTrainedModel,)
    except Exception:
        pass

    try:
        from transformers import PreTrainedModel

        supported_model_types += (PreTrainedModel,)
    except Exception:
        pass

    try:
        from transformers import TFPreTrainedModel

        supported_model_types += (TFPreTrainedModel,)
    except Exception:
        pass

    return supported_model_types


def _build_pipeline_from_model_input(model_dict: dict[str, Any], task: str | None) -> Pipeline:
    """
    Utility for generating a pipeline from component parts. If required components are not
    specified, use the transformers library pipeline component validation to force raising an
    exception. The underlying Exception thrown in transformers is verbose enough for diagnosis.
    """

    from transformers import pipeline

    model = model_dict[FlavorKey.MODEL]

    if not (isinstance(model, _get_supported_pretrained_model_types()) or is_peft_model(model)):
        raise MlflowException(
            "The supplied model type is unsupported. The model must be one of: "
            "PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel, or PeftModel",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if task is None or task.startswith(_LLM_INFERENCE_TASK_PREFIX):
        default_task = _get_default_task_for_llm_inference_task(task)
        task = _get_task_for_model(model.name_or_path, default_task=default_task)

    try:
        with suppress_logs("transformers.pipelines.base", filter_regex=_PEFT_PIPELINE_ERROR_MSG):
            return pipeline(task=task, **model_dict)
    except Exception as e:
        raise MlflowException(
            "The provided model configuration cannot be created as a Pipeline. "
            "Please verify that all required and compatible components are "
            "specified with the correct keys.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


def _get_task_for_model(model_name_or_path: str, default_task=None) -> str:
    """
    Get the Transformers pipeline task type fro the model instance.

    NB: The get_task() function only works for remote models available in the Hugging
    Face hub, so the default task should be supplied when using a custom local model.
    """
    from transformers.pipelines import get_supported_tasks, get_task

    try:
        model_task = get_task(model_name_or_path)
        if model_task in get_supported_tasks():
            return model_task
        elif default_task is not None:
            _logger.warning(
                f"The task '{model_task}' inferred from the model is not"
                "supported by the transformers pipeline. MLflow will "
                f"construct the pipeline with the fallback task {default_task} "
                "inferred from the specified 'llm/v1/xxx' task."
            )
            return default_task
        else:
            raise MlflowException(
                f"Cannot construct transformers pipeline because the task '{model_task}' "
                "inferred from the model is not supported by the transformers pipeline. "
                "Please construct the pipeline instance manually and pass it to the "
                "`log_model` or `save_model` function."
            )

    except RuntimeError as e:
        if default_task:
            return default_task
        raise MlflowException(
            "The task could not be inferred from the model. If you are saving a custom "
            "local model that is not available in the Hugging Face hub, please provide "
            "the `task` argument to the `log_model` or `save_model` function.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


def _validate_llm_inference_task_type(llm_inference_task: str, pipeline_task: str) -> None:
    """
    Validates that an ``inference_task`` type is supported by ``transformers`` pipeline type.
    """
    supported_llm_inference_tasks = _SUPPORTED_LLM_INFERENCE_TASK_TYPES_BY_PIPELINE_TASK.get(
        pipeline_task, []
    )

    if llm_inference_task not in supported_llm_inference_tasks:
        raise MlflowException(
            f"The task provided is invalid. '{llm_inference_task}' is not a supported task for "
            f"the {pipeline_task} pipeline. Must be one of {supported_llm_inference_tasks}",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _get_engine_type(model):
    """
    Determines the underlying execution engine for the model based on the 3 currently supported
    deep learning framework backends: ``tensorflow``, ``torch``, or ``flax``.
    """
    from transformers import FlaxPreTrainedModel, PreTrainedModel, TFPreTrainedModel
    from transformers.utils import is_torch_available

    if is_peft_model(model):
        model = get_peft_base_model(model)

    for cls in model.__class__.__mro__:
        if issubclass(cls, TFPreTrainedModel):
            return "tensorflow"
        elif issubclass(cls, PreTrainedModel):
            return "torch"
        elif issubclass(cls, FlaxPreTrainedModel):
            return "flax"

    # As a fallback, we check current environment to determine the engine type
    return "torch" if is_torch_available() else "tensorflow"


def _should_add_pyfunc_to_model(pipeline) -> bool:
    """
    Discriminator for determining whether a particular task type and model instance from within
    a ``Pipeline`` is currently supported for the pyfunc flavor.

    Image and Video pipelines can still be logged and used, but are not available for
    loading as pyfunc.
    Similarly, esoteric model types (Graph Models, Timeseries Models, and Reinforcement Learning
    Models) are not permitted for loading as pyfunc due to the complex input types that, in
    order to support, will require significant modifications (breaking changes) to the pyfunc
    contract.
    """
    import transformers

    exclusion_model_types = {
        "GraphormerPreTrainedModel",
        "InformerPreTrainedModel",
        "TimeSeriesTransformerPreTrainedModel",
        "DecisionTransformerPreTrainedModel",
    }

    # NB: When pyfunc functionality is added for these pipeline types over time, remove the
    # entries from the following list.
    exclusion_pipeline_types = [
        "DocumentQuestionAnsweringPipeline",
        "ImageToTextPipeline",
        "VisualQuestionAnsweringPipeline",
        "ImageSegmentationPipeline",
        "DepthEstimationPipeline",
        "ObjectDetectionPipeline",
        "VideoClassificationPipeline",
        "ZeroShotImageClassificationPipeline",
        "ZeroShotObjectDetectionPipeline",
        "ZeroShotAudioClassificationPipeline",
    ]

    for model_type in exclusion_model_types:
        if hasattr(transformers, model_type):
            if isinstance(pipeline.model, getattr(transformers, model_type)):
                return False
    if type(pipeline).__name__ in exclusion_pipeline_types:
        return False
    return True


def _get_model_config(local_path, pyfunc_config):
    """
    Load the model configuration if it was provided for use in the `_TransformersWrapper` pyfunc
    Model wrapper.
    """
    config_path = local_path.joinpath("inference_config.txt")
    if config_path.exists():
        _logger.warning(
            "Inference config stored in file ``inference_config.txt`` is deprecated. New logged "
            "models will store the model configuration in the ``pyfunc`` flavor configuration."
        )
        return json.loads(config_path.read_text())
    else:
        return pyfunc_config or {}


def _load_pyfunc(path, model_config: dict[str, Any] | None = None):
    """
    Loads the model as pyfunc model
    """
    local_path = pathlib.Path(path)
    flavor_configuration = _get_flavor_configuration(local_path, FLAVOR_NAME)
    model_config = _get_model_config(local_path.joinpath(_COMPONENTS_BINARY_DIR_NAME), model_config)
    prompt_template = _get_prompt_template(local_path)

    return _TransformersWrapper(
        _load_model(str(local_path), flavor_configuration, "pipeline"),
        flavor_configuration,
        model_config,
        prompt_template,
    )


def _is_conversational_pipeline(pipeline):
    """
    Checks if the pipeline is a ConversationalPipeline.
    """
    if cp := _try_import_conversational_pipeline():
        return isinstance(pipeline, cp)
    return False


def _try_import_conversational_pipeline():
    """
    Try importing ConversationalPipeline because for version > 4.41.2
    it is removed from the transformers package.
    """
    try:
        from transformers import ConversationalPipeline

        return ConversationalPipeline
    except ImportError:
        return


def generate_signature_output(pipeline, data, model_config=None, params=None, flavor_config=None):
    """
    Utility for generating the response output for the purposes of extracting an output signature
    for model saving and logging. This function simulates loading of a saved model or pipeline
    as a ``pyfunc`` model without having to incur a write to disk.

    Args:
        pipeline: A ``transformers`` pipeline object. Note that component-level or model-level
            inputs are not permitted for extracting an output example.
        data: An example input that is compatible with the given pipeline
        model_config: Any additional model configuration, provided as kwargs, to inform
            the format of the output type from a pipeline inference call.
        params: A dictionary of additional parameters to pass to the pipeline for inference.
        flavor_config: The flavor configuration for the model.

    Returns:
        The output from the ``pyfunc`` pipeline wrapper's ``predict`` method
    """
    import transformers

    from mlflow.transformers import signature

    if not isinstance(pipeline, transformers.Pipeline):
        raise MlflowException(
            f"The pipeline type submitted is not a valid transformers Pipeline. "
            f"The type {type(pipeline).__name__} is not supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return signature.generate_signature_output(pipeline, data, model_config, params)


class _TransformersWrapper:
    def __init__(self, pipeline, flavor_config=None, model_config=None, prompt_template=None):
        self.pipeline = pipeline
        self.flavor_config = flavor_config
        # The predict method updates the model_config several times. This should be done over a
        # deep copy of the original model_config that was specified by the user, otherwise the
        # prediction won't be idempotent. Hence we creates an immutable dictionary of the original
        # model config here and enforce creating a deep copy at every predict call.
        self.model_config = MappingProxyType(model_config or {})

        self.prompt_template = prompt_template
        self._conversation = None
        # NB: Current special-case custom pipeline types that have not been added to
        # the native-supported transformers package but require custom parsing:
        # InstructionTextGenerationPipeline [Dolly] https://huggingface.co/databricks/dolly-v2-12b
        #   (and all variants)
        self._supported_custom_generator_types = {"InstructionTextGenerationPipeline"}
        self.llm_inference_task = (
            self.flavor_config.get(_LLM_INFERENCE_TASK_KEY) if self.flavor_config else None
        )

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.pipeline

    def _convert_pandas_to_dict(self, data):
        import transformers

        if not isinstance(self.pipeline, transformers.ZeroShotClassificationPipeline):
            return data.to_dict(orient="records")
        else:
            # NB: The ZeroShotClassificationPipeline requires an input in the form of
            # Dict[str, Union[str, List[str]]] and will throw if an additional nested
            # List is present within the List value (which is what the duplicated values
            # within the orient="list" conversion in Pandas will do. This parser will
            # deduplicate label lists to a single list.
            unpacked = data.to_dict(orient="list")
            parsed = {}
            for key, value in unpacked.items():
                if isinstance(value, list):
                    contents = []
                    for item in value:
                        # Deduplication logic
                        if item not in contents:
                            contents.append(item)
                    # Collapse nested lists to return the correct data structure for the
                    # ZeroShotClassificationPipeline input structure
                    parsed[key] = (
                        contents
                        if all(isinstance(item, str) for item in contents) and len(contents) > 1
                        else contents[0]
                    )
            return parsed

    def _merge_model_config_with_params(self, model_config, params):
        if params:
            _logger.warning(
                "params provided to the `predict` method will override the inference "
                "configuration saved with the model. If the params provided are not "
                "valid for the pipeline, MlflowException will be raised."
            )
            # Override the inference configuration with any additional kwargs provided by the user.
            return {**model_config, **params}
        else:
            return model_config

    def _validate_model_config_and_return_output(self, data, model_config, return_tensors=False):
        import transformers

        if return_tensors:
            model_config["return_tensors"] = True
            if model_config.get("return_full_text", None) is not None:
                _logger.warning(
                    "The `return_full_text` parameter is mutually exclusive with the "
                    "`return_tensors` parameter set when a MLflow inference task is provided. "
                    "The `return_full_text` parameter will be ignored."
                )
                # `return_full_text` is mutually exclusive with `return_tensors`
                model_config["return_full_text"] = None

        try:
            if isinstance(data, dict):
                return self.pipeline(**data, **model_config)
            return self.pipeline(data, **model_config)
        except ValueError as e:
            if "The following `model_kwargs` are not used by the model" in str(e):
                raise MlflowException.invalid_parameter_value(
                    "The params provided to the `predict` method are not valid "
                    f"for pipeline {type(self.pipeline).__name__}.",
                ) from e
            if isinstance(
                self.pipeline,
                (
                    transformers.AutomaticSpeechRecognitionPipeline,
                    transformers.AudioClassificationPipeline,
                ),
            ) and (
                # transformers <= 4.33.3
                "Malformed soundfile" in str(e)
                # transformers > 4.33.3
                or "Soundfile is either not in the correct format or is malformed" in str(e)
            ):
                raise MlflowException.invalid_parameter_value(
                    "Failed to process the input audio data. Either the audio file is "
                    "corrupted or a uri was passed in without overriding the default model "
                    "signature. If submitting a string uri, please ensure that the model has "
                    "been saved with a signature that defines a string input type.",
                ) from e
            raise

    def predict(self, data, params: dict[str, Any] | None = None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        # NB: This `predict` method updates the model_config several times. To make the predict
        # call idempotent, we keep the original self.model_config immutable and creates a deep
        # copy of it at every predict call.
        model_config = copy.deepcopy(dict(self.model_config))
        params = self._merge_model_config_with_params(model_config, params)

        if self.llm_inference_task == _LLM_INFERENCE_TASK_CHAT:
            data, params = preprocess_llm_inference_input(data, params, self.flavor_config)
            data = [convert_messages_to_prompt(msgs, self.pipeline.tokenizer) for msgs in data]
        elif self.llm_inference_task == _LLM_INFERENCE_TASK_COMPLETIONS:
            data, params = preprocess_llm_inference_input(data, params, self.flavor_config)
        elif self.llm_inference_task == _LLM_INFERENCE_TASK_EMBEDDING:
            data, params = preprocess_llm_embedding_params(data)

        if isinstance(data, pd.DataFrame):
            input_data = self._convert_pandas_to_dict(data)
        elif isinstance(data, (dict, str, bytes, np.ndarray)):
            input_data = data
        elif isinstance(data, list):
            if not all(isinstance(entry, (str, dict)) for entry in data):
                raise MlflowException(
                    "Invalid data submission. Ensure all elements in the list are strings "
                    "or dictionaries. If dictionaries are supplied, all keys in the "
                    "dictionaries must be strings and values must be either str or List[str].",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            input_data = data
        else:
            raise MlflowException(
                "Input data must be either a pandas.DataFrame, a string, bytes, List[str], "
                "List[Dict[str, str]], List[Dict[str, Union[str, List[str]]]], "
                "or Dict[str, Union[str, List[str]]].",
                error_code=INVALID_PARAMETER_VALUE,
            )
        input_data = self._parse_raw_pipeline_input(input_data)
        # Validate resolved or input dict types
        if isinstance(input_data, dict):
            _validate_input_dictionary_contains_only_strings_and_lists_of_strings(input_data)
        elif isinstance(input_data, list) and all(isinstance(entry, dict) for entry in input_data):
            # Validate each dict inside an input List[Dict]
            all(
                _validate_input_dictionary_contains_only_strings_and_lists_of_strings(x)
                for x in input_data
            )
        return self._predict(input_data, params)

    def _predict(self, data, model_config):
        import transformers

        # NB: the ordering of these conditional statements matters. TranslationPipeline and
        # SummarizationPipeline both inherit from TextGenerationPipeline (they are subclasses)
        # in which the return data structure from their __call__ implementation is modified.
        if isinstance(self.pipeline, transformers.TranslationPipeline):
            self._validate_str_or_list_str(data)
            output_key = "translation_text"
        elif isinstance(self.pipeline, transformers.SummarizationPipeline):
            self._validate_str_or_list_str(data)
            data = self._format_prompt_template(data)
            output_key = "summary_text"
        elif isinstance(self.pipeline, transformers.Text2TextGenerationPipeline):
            data = self._parse_text2text_input(data)
            data = self._format_prompt_template(data)
            output_key = "generated_text"
        elif isinstance(self.pipeline, transformers.TextGenerationPipeline):
            self._validate_str_or_list_str(data)
            data = self._format_prompt_template(data)
            output_key = "generated_text"
        elif isinstance(self.pipeline, transformers.QuestionAnsweringPipeline):
            data = self._parse_question_answer_input(data)
            output_key = "answer"
        elif isinstance(self.pipeline, transformers.FillMaskPipeline):
            self._validate_str_or_list_str(data)
            data = self._format_prompt_template(data)
            output_key = "token_str"
        elif isinstance(self.pipeline, transformers.TextClassificationPipeline):
            output_key = "label"
        elif isinstance(self.pipeline, transformers.ImageClassificationPipeline):
            data = self._convert_image_input(data)
            output_key = "label"
        elif isinstance(self.pipeline, transformers.ZeroShotClassificationPipeline):
            output_key = "labels"
            data = self._parse_json_encoded_list(data, "candidate_labels")
        elif isinstance(self.pipeline, transformers.TableQuestionAnsweringPipeline):
            output_key = "answer"
            data = self._parse_json_encoded_dict_payload_to_dict(data, "table")
        elif isinstance(self.pipeline, transformers.TokenClassificationPipeline):
            output_key = {"entity_group", "entity"}
        elif isinstance(self.pipeline, transformers.FeatureExtractionPipeline):
            output_key = None
            data = self._parse_feature_extraction_input(data)
            data = self._format_prompt_template(data)
        elif _is_conversational_pipeline(self.pipeline):
            output_key = None
            if not self._conversation:
                # this import is valid if conversational_pipeline is not None
                self._conversation = transformers.Conversation()
            self._conversation.add_user_input(data)
        elif type(self.pipeline).__name__ in self._supported_custom_generator_types:
            self._validate_str_or_list_str(data)
            output_key = "generated_text"
        elif isinstance(self.pipeline, transformers.AutomaticSpeechRecognitionPipeline):
            if model_config.get("return_timestamps", None) in ["word", "char"]:
                output_key = None
            else:
                output_key = "text"
            data = self._convert_audio_input(data)
        elif isinstance(self.pipeline, transformers.AudioClassificationPipeline):
            data = self._convert_audio_input(data)
            output_key = None
        else:
            raise MlflowException(
                f"The loaded pipeline type {type(self.pipeline).__name__} is "
                "not enabled for pyfunc predict functionality.",
                error_code=BAD_REQUEST,
            )

        # Optional input preservation for specific pipeline types. This is True (include raw
        # formatting output), but if `include_prompt` is set to False in the `model_config`
        # option during model saving, excess newline characters and the fed-in prompt will be
        # trimmed out from the start of the response.
        include_prompt = model_config.pop("include_prompt", True)
        # Optional stripping out of `\n` for specific generator pipelines.
        collapse_whitespace = model_config.pop("collapse_whitespace", False)

        data = self._convert_cast_lists_from_np_back_to_list(data)

        # Generate inference data with the pipeline object
        if _is_conversational_pipeline(self.pipeline):
            conversation_output = self.pipeline(self._conversation)
            return conversation_output.generated_responses[-1]
        else:
            # If inference task is defined, return tensors internally to get usage information
            return_tensors = False
            if self.llm_inference_task:
                return_tensors = True
                output_key = "generated_token_ids"

            raw_output = self._validate_model_config_and_return_output(
                data, model_config=model_config, return_tensors=return_tensors
            )

        # Handle the pipeline outputs
        if type(self.pipeline).__name__ in self._supported_custom_generator_types or isinstance(
            self.pipeline, transformers.TextGenerationPipeline
        ):
            output = self._strip_input_from_response_in_instruction_pipelines(
                data,
                raw_output,
                output_key,
                self.flavor_config,
                include_prompt,
                collapse_whitespace,
            )

            if self.llm_inference_task:
                output = postprocess_output_for_llm_inference_task(
                    data,
                    output,
                    self.pipeline,
                    self.flavor_config,
                    model_config,
                    self.llm_inference_task,
                )

        elif isinstance(self.pipeline, transformers.FeatureExtractionPipeline):
            if self.llm_inference_task:
                output = [np.array(tensor[0][0]) for tensor in raw_output]
                output = postprocess_output_for_llm_v1_embedding_task(
                    data, output, self.pipeline.tokenizer
                )
            else:
                return self._parse_feature_extraction_output(raw_output)
        elif isinstance(self.pipeline, transformers.FillMaskPipeline):
            output = self._parse_list_of_multiple_dicts(raw_output, output_key)
        elif isinstance(self.pipeline, transformers.ZeroShotClassificationPipeline):
            return self._flatten_zero_shot_text_classifier_output_to_df(raw_output)
        elif isinstance(self.pipeline, transformers.TokenClassificationPipeline):
            output = self._parse_tokenizer_output(raw_output, output_key)
        elif isinstance(
            self.pipeline, transformers.AutomaticSpeechRecognitionPipeline
        ) and model_config.get("return_timestamps", None) in ["word", "char"]:
            output = json.dumps(raw_output)
        elif isinstance(
            self.pipeline,
            (
                transformers.AudioClassificationPipeline,
                transformers.TextClassificationPipeline,
                transformers.ImageClassificationPipeline,
            ),
        ):
            return pd.DataFrame(raw_output)
        else:
            output = self._parse_lists_of_dict_to_list_of_str(raw_output, output_key)

        sanitized = self._sanitize_output(output, data)
        return self._wrap_strings_as_list_if_scalar(sanitized)

    def _parse_raw_pipeline_input(self, data):
        """
        Converts inputs to the expected types for specific Pipeline types.
        Specific logic for individual pipeline types are called via their respective methods if
        the input isn't a basic str or List[str] input type of Pipeline.
        These parsers are required due to the conversion that occurs within schema validation to
        a Pandas DataFrame encapsulation, a format which is unsupported for the `transformers`
        library.
        """
        import transformers

        if isinstance(self.pipeline, transformers.TableQuestionAnsweringPipeline):
            data = self._coerce_exploded_dict_to_single_dict(data)
            return self._parse_input_for_table_question_answering(data)
        elif _is_conversational_pipeline(self.pipeline):
            return self._parse_conversation_input(data)
        elif (  # noqa: SIM114
            isinstance(
                self.pipeline,
                (
                    transformers.FillMaskPipeline,
                    transformers.TextGenerationPipeline,
                    transformers.TranslationPipeline,
                    transformers.SummarizationPipeline,
                    transformers.TokenClassificationPipeline,
                ),
            )
            and isinstance(data, list)
            and all(isinstance(entry, dict) for entry in data)
        ):
            return [list(entry.values())[0] for entry in data]
        # NB: For Text2TextGenerationPipeline, we need more complex handling for dictionary,
        # as we allow both single string input and dictionary input (or list of them). Both
        # are once wrapped to Pandas DataFrame during schema enforcement and convert back to
        # dictionary. The difference between two is columns of the DataFrame, where the first
        # case (string) will have auto-generated columns like 0, 1, ... while the latter (dict)
        # will have the original keys to be the columns. When converting back to dictionary,
        # those columns will becomes the key of dictionary.
        #
        # E.g.
        #  1. If user's input is string like model.predict("foo")
        #    -> Raw input: "foo"
        #    -> Pandas dataframe has column 0, with single row "foo"
        #    -> Derived dictionary will be {0: "foo"}
        #  2. If user's input is dictionary like model.predict({"text": "foo"})
        #    -> Raw input: {"text": "foo"}
        #    -> Pandas dataframe has column "text", with single row "foo"
        #    -> Derived dictionary will be {"text": "foo"}
        #
        # Then for the first case, we want to extract values only, similar to other pipelines.
        # However, for the second case, we want to keep the key-value pair as it is.
        # In long-term, we should definitely change the upstream handling to avoid this
        # complexity, but here we just try to make it work by checking if the key is auto-generated.
        elif (
            isinstance(self.pipeline, transformers.Text2TextGenerationPipeline)
            and isinstance(data, list)
            and all(isinstance(entry, dict) for entry in data)
            # Pandas Dataframe derived dictionary will have integer key (row index)
            and 0 in data[0].keys()
        ):
            return [list(entry.values())[0] for entry in data]
        elif isinstance(self.pipeline, transformers.TextClassificationPipeline):
            return self._validate_text_classification_input(data)
        else:
            return data

    @staticmethod
    def _validate_text_classification_input(data):
        """
        Perform input type validation for TextClassification pipelines and casting of data
        that is manipulated internally by the MLflow model server back to a structure that
        can be used for pipeline inference.

        To illustrate the input and outputs of this function, for the following inputs to
        the pyfunc.predict() call for this pipeline type:

        "text to classify"
        ["text to classify", "other text to classify"]
        {"text": "text to classify", "text_pair": "pair text"}
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        Pyfunc processing will convert these to the following structures:

        [{0: "text to classify"}]
        [{0: "text to classify"}, {0: "other text to classify"}]
        [{"text": "text to classify", "text_pair": "pair text"}]
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        The purpose of this function is to convert them into the correct format for input
        to the pipeline (wrapping as a list has no bearing on the correctness of the
        inferred classifications):

        ["text to classify"]
        ["text to classify", "other text to classify"]
        [{"text": "text to classify", "text_pair": "pair text"}]
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        Additionally, for dict input types (the 'text' & 'text_pair' input example), the dict
        input will be JSON stringified within MLflow model serving. In order to reconvert this
        structure back into the appropriate type, we use ast.literal_eval() to convert back
        to a dict. We avoid using JSON.loads() due to pandas DataFrame conversions that invert
        single and double quotes with escape sequences that are not consistent if the string
        contains escaped quotes.
        """

        def _check_keys(payload):
            """Check if a dictionary contains only allowable keys."""
            allowable_str_keys = {"text", "text_pair"}
            if set(payload) - allowable_str_keys and not all(
                isinstance(key, int) for key in payload.keys()
            ):
                raise MlflowException(
                    "Text Classification pipelines may only define dictionary inputs with keys "
                    f"defined as {allowable_str_keys}"
                )

        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            _check_keys(data)
            return data
        elif isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return data
            elif all(isinstance(item, dict) for item in data):
                for payload in data:
                    _check_keys(payload)
                if list(data[0].keys())[0] == 0:
                    data = [item[0] for item in data]
                try:
                    # NB: To support MLflow serving signature validation, the value within dict
                    # inputs is JSON encoded. In order for the proper data structure input support
                    # for a {"text": "a", "text_pair": "b"} (or the list of such a structure) as
                    # an input, we have to convert the string encoded dict back to a dict.
                    # Due to how unescaped characters (such as "'") are encoded, using an explicit
                    # json.loads() attempted cast can result in invalid input data to the pipeline.
                    # ast.literal_eval() shows correct conversion, as validated in unit tests.
                    return [ast.literal_eval(s) for s in data]
                except (ValueError, SyntaxError):
                    return data
            else:
                raise MlflowException(
                    "An unsupported data type has been passed for Text Classification inference. "
                    "Only str, list of str, dict, and list of dict are supported."
                )
        else:
            raise MlflowException(
                "An unsupported data type has been passed for Text Classification inference. "
                "Only str, list of str, dict, and list of dict are supported."
            )

    def _parse_conversation_input(self, data) -> str:
        if isinstance(data, str):
            return data
        elif isinstance(data, list) and all(isinstance(elem, dict) for elem in data):
            return next(iter(data[0].values()))
        elif isinstance(data, dict):
            # The conversation pipeline can only accept a single string at a time
            return next(iter(data.values()))

    def _parse_input_for_table_question_answering(self, data):
        if "table" not in data:
            raise MlflowException(
                "The input dictionary must have the 'table' key.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif isinstance(data["table"], dict):
            data["table"] = json.dumps(data["table"])
            return data
        else:
            return data

    def _coerce_exploded_dict_to_single_dict(
        self, data: list[dict[str, Any]]
    ) -> dict[str, list[Any]]:
        """
        Parses the result of Pandas DataFrame.to_dict(orient="records") from pyfunc
        signature validation to coerce the output to the required format for a
        Pipeline that requires a single dict with list elements such as
        TableQuestionAnsweringPipeline.
        Example input:

        [
          {"answer": "We should order more pizzas to meet the demand."},
          {"answer": "The venue size should be updated to handle the number of guests."},
        ]

        Output:

        {
          "answer": [
              "We should order more pizzas to meet the demand.",
              "The venue size should be updated to handle the number of guests.",
          ]
        }
        """
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            collection = data.copy()
            parsed = collection[0]
            for coll in collection:
                for key, value in coll.items():
                    if key not in parsed:
                        raise MlflowException(
                            "Unable to parse the input. The keys within each "
                            "dictionary of the parsed input are not consistent"
                            "among the dictionaries.",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    if value != parsed[key]:
                        value_type = type(parsed[key])
                        if value_type == str:
                            parsed[key] = [parsed[key], value]
                        elif value_type == list:
                            if all(len(entry) == 1 for entry in value):
                                # This conversion is required solely for model serving.
                                # In the parsing logic that occurs internally, strings that
                                # contain single quotes `'` result in casting to a List[char]
                                # instead of a str type. Attempting to append a List[char]
                                # to a List[str] as would happen in the `else` block here
                                # results in the entire List being overwritten as `None` without
                                # an Exception being raised. By checking for single value entries
                                # and subsequently converting to list and extracting the first
                                # element reconstructs the original input string.
                                parsed[key].append([str(value)][0])
                            else:
                                parsed[key] = parsed[key].append(value)
                        else:
                            parsed[key] = value
            return parsed
        else:
            return data

    def _flatten_zero_shot_text_classifier_output_to_df(self, data):
        """
        Converts the output of sequences, labels, and scores to a Pandas DataFrame output.

        Example input:

        [{'sequence': 'My dog loves to eat spaghetti',
          'labels': ['happy', 'sad'],
          'scores': [0.9896970987319946, 0.010302911512553692]},
         {'sequence': 'My dog hates going to the vet',
          'labels': ['sad', 'happy'],
          'scores': [0.957074761390686, 0.042925238609313965]}]

        Output:

        pd.DataFrame in a fully normalized (flattened) format with each sequence, label, and score
        having a row entry.
        For example, here is the DataFrame output:

                                sequence labels    scores
        0  My dog loves to eat spaghetti  happy  0.989697
        1  My dog loves to eat spaghetti    sad  0.010303
        2  My dog hates going to the vet    sad  0.957075
        3  My dog hates going to the vet  happy  0.042925
        """
        if isinstance(data, list) and not all(isinstance(item, dict) for item in data):
            raise MlflowException(
                "Encountered an unknown return type from the pipeline type "
                f"{type(self.pipeline).__name__}. Expecting a List[Dict]",
                error_code=BAD_REQUEST,
            )
        if isinstance(data, dict):
            data = [data]

        flattened_data = []
        for entry in data:
            for label, score in zip(entry["labels"], entry["scores"]):
                flattened_data.append(
                    {"sequence": entry["sequence"], "labels": label, "scores": score}
                )
        return pd.DataFrame(flattened_data)

    def _strip_input_from_response_in_instruction_pipelines(
        self,
        input_data,
        output,
        output_key,
        flavor_config,
        include_prompt=True,
        collapse_whitespace=False,
    ):
        """
        Parse the output from instruction pipelines to conform with other text generator
        pipeline types and remove line feed characters and other confusing outputs
        """

        def extract_response_data(data_out):
            if all(isinstance(x, dict) for x in data_out):
                return [elem[output_key] for elem in data_out][0]
            elif all(isinstance(x, list) for x in data_out):
                return [elem[output_key] for coll in data_out for elem in coll]
            else:
                raise MlflowException(
                    "Unable to parse the pipeline output. Expected List[Dict[str,str]] or "
                    f"List[List[Dict[str,str]]] but got {type(data_out)} instead."
                )

        output = extract_response_data(output)

        def trim_input(data_in, data_out):
            # NB: the '\n\n' pattern is exclusive to specific InstructionalTextGenerationPipeline
            # types that have been loaded as a plain TextGenerator. The structure of these
            # pipelines will precisely repeat the input question immediately followed by 2 carriage
            # return statements, followed by the start of the response to the prompt. We only
            # want to left-trim these types of pipelines output values if the user has indicated
            # the removal action of the input prompt in the returned str or List[str] by applying
            # the optional model_config entry of `{"include_prompt": False}`.
            # By default, the prompt is included in the response.
            # Stripping out additional carriage returns (\n) is another additional optional flag
            # that can be set for these generator pipelines. It is off by default (False).
            if (
                not include_prompt
                and flavor_config[FlavorKey.INSTANCE_TYPE] in self._supported_custom_generator_types
                and data_out.startswith(data_in + "\n\n")
            ):
                # If the user has indicated to not preserve the prompt input in the response,
                # split the response output and trim the input prompt from the response.
                data_out = data_out[len(data_in) :].lstrip()
                if data_out.startswith("A:"):
                    data_out = data_out[2:].lstrip()

            # If the user has indicated to remove newlines and extra spaces from the generated
            # text, replace them with a single space.
            if collapse_whitespace:
                data_out = re.sub(r"\s+", " ", data_out).strip()
            return data_out

        if isinstance(input_data, list) and isinstance(output, list):
            return [trim_input(data_in, data_out) for data_in, data_out in zip(input_data, output)]
        elif isinstance(input_data, str) and isinstance(output, str):
            return trim_input(input_data, output)
        else:
            raise MlflowException(
                "Unknown data structure after parsing output. Expected str or List[str]. "
                f"Got {type(output)} instead."
            )

    def _sanitize_output(self, output, input_data):
        # Some pipelines and their underlying models leave leading or trailing whitespace.
        # This method removes that whitespace.
        import transformers

        if (
            not isinstance(self.pipeline, transformers.TokenClassificationPipeline)
            and isinstance(input_data, str)
            and isinstance(output, list)
        ):
            # Retrieve the first output for return types that are List[str] of only a single
            # element.
            output = output[0]
        if isinstance(output, str):
            return output.strip()
        elif isinstance(output, list):
            if all(isinstance(elem, str) for elem in output):
                cleaned = [text.strip() for text in output]
                # If the list has only a single string, return as string.
                return cleaned if len(cleaned) > 1 else cleaned[0]
            else:
                return [self._sanitize_output(coll, input_data) for coll in output]
        elif isinstance(output, dict) and all(
            isinstance(key, str) and isinstance(value, str) for key, value in output.items()
        ):
            return {k: v.strip() for k, v in output.items()}
        else:
            return output

    @staticmethod
    def _wrap_strings_as_list_if_scalar(output_data):
        """
        Wraps single string outputs in a list to support batch processing logic in serving.
        Scalar values are not supported for processing in batch logic as they cannot be coerced
        to DataFrame representations.
        """
        if isinstance(output_data, str):
            return [output_data]
        else:
            return output_data

    def _parse_lists_of_dict_to_list_of_str(self, output_data, target_dict_key) -> list[str]:
        """
        Parses the output results from select Pipeline types to extract specific values from a
        target key.
        Examples (with "a" as the `target_dict_key`):

        Input: [{"a": "valid", "b": "invalid"}, {"a": "another valid", "c": invalid"}]
        Output: ["valid", "another_valid"]

        Input: [{"a": "valid", "b": [{"a": "another valid"}, {"b": "invalid"}]},
                {"a": "valid 2", "b": [{"a": "another valid 2"}, {"c": "invalid"}]}]
        Output: ["valid", "another valid", "valid 2", "another valid 2"]
        """
        if isinstance(output_data, list):
            output_coll = []
            for output in output_data:
                if isinstance(output, dict):
                    for key, value in output.items():
                        if key == target_dict_key:
                            output_coll.append(output[target_dict_key])
                        elif isinstance(value, list) and all(
                            isinstance(elem, dict) for elem in value
                        ):
                            output_coll.extend(
                                self._parse_lists_of_dict_to_list_of_str(value, target_dict_key)
                            )
                elif isinstance(output, list):
                    output_coll.extend(
                        self._parse_lists_of_dict_to_list_of_str(output, target_dict_key)
                    )
            return output_coll
        elif target_dict_key:
            return output_data[target_dict_key]
        else:
            return output_data

    @staticmethod
    def _parse_feature_extraction_input(input_data):
        if isinstance(input_data, list) and isinstance(input_data[0], dict):
            return [list(data.values())[0] for data in input_data]
        else:
            return input_data

    @staticmethod
    def _parse_feature_extraction_output(output_data):
        """
        Parse the return type from a FeatureExtractionPipeline output. The mixed types for
        input are present depending on how the pyfunc is instantiated. For model serving usage,
        the returned type from MLServer will be a numpy.ndarray type, otherwise, the return
        within a manually executed pyfunc (i.e., for udf usage), the return will be a collection
        of nested lists.

        Examples:

        Input: [[[0.11, 0.98, 0.76]]] or np.array([0.11, 0.98, 0.76])
        Output: np.array([0.11, 0.98, 0.76])

        Input: [[[[0.1, 0.2], [0.3, 0.4]]]] or
            np.array([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
        Output: np.array([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
        """
        if isinstance(output_data, np.ndarray):
            return output_data
        else:
            return np.array(output_data[0][0])

    def _parse_tokenizer_output(self, output_data, target_set):
        """
        Parses the tokenizer pipeline output.

        Examples:

        Input: [{"entity": "PRON", "score": 0.95}, {"entity": "NOUN", "score": 0.998}]
        Output: "PRON,NOUN"

        Input: [[{"entity": "PRON", "score": 0.95}, {"entity": "NOUN", "score": 0.998}],
                [{"entity": "PRON", "score": 0.95}, {"entity": "NOUN", "score": 0.998}]]
        Output: ["PRON,NOUN", "PRON,NOUN"]
        """
        # NB: We're collapsing the results here to a comma separated string for each inference
        # input string. This is to simplify having to otherwise make extensive changes to
        # ColSpec in order to support schema enforcement of List[List[str]]
        if isinstance(output_data[0], list):
            return [self._parse_tokenizer_output(coll, target_set) for coll in output_data]
        else:
            # NB: Since there are no attributes accessible from the pipeline object that determine
            # what the characteristics of the return structure names are within the dictionaries,
            # Determine which one is present in the output to extract the correct entries.
            target = target_set.intersection(output_data[0].keys()).pop()
            return ",".join([coll[target] for coll in output_data])

    @staticmethod
    def _parse_list_of_multiple_dicts(output_data, target_dict_key):
        """
        Returns the first value of the `target_dict_key` that matches in the first dictionary in a
        list of dictionaries.
        """

        def fetch_target_key_value(data, key):
            if isinstance(data[0], dict):
                return data[0][key]
            return [item[0][key] for item in data]

        if isinstance(output_data[0], list):
            return [
                fetch_target_key_value(collection, target_dict_key) for collection in output_data
            ]
        else:
            return [output_data[0][target_dict_key]]

    def _parse_question_answer_input(self, data):
        """
        Parses the single string input representation for a question answer pipeline into the
        required dict format for a `question-answering` pipeline.
        """
        if isinstance(data, list):
            return [self._parse_question_answer_input(entry) for entry in data]
        elif isinstance(data, dict):
            expected_keys = {"question", "context"}
            if not expected_keys.intersection(set(data.keys())) == expected_keys:
                raise MlflowException(
                    f"Invalid keys were submitted. Keys must be exclusively {expected_keys}"
                )
            return data
        else:
            raise MlflowException(
                "An invalid type has been supplied. Must be either List[Dict[str, str]] or "
                f"Dict[str, str]. {type(data)} is not supported.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _parse_text2text_input(self, data):
        """
        Parses the mixed input types that can be submitted into a text2text Pipeline.
        Valid examples:

        Input:
            {"context": "abc", "answer": "def"}
        Output:
            "context: abc answer: def"
        Input:
            [{"context": "abc", "answer": "def"}, {"context": "ghi", "answer": "jkl"}]
        Output:
            ["context: abc answer: def", "context: ghi answer: jkl"]
        Input:
            "abc"
        Output:
            "abc"
        Input:
            ["abc", "def"]
        Output:
            ["abc", "def"]
        """
        if isinstance(data, dict) and all(isinstance(value, str) for value in data.values()):
            if all(isinstance(key, str) for key in data) and "inputs" not in data:
                # NB: Text2Text Pipelines require submission of text in a pseudo-string based dict
                # formatting.
                # As an example, for the input of:
                # data = {"context": "The sky is blue", "answer": "blue"}
                # This method will return the Pipeline-required format of:
                # "context: The sky is blue. answer: blue"
                return " ".join(f"{key}: {value}" for key, value in data.items())
            else:
                return list(data.values())
        elif isinstance(data, list) and all(isinstance(value, dict) for value in data):
            return [self._parse_text2text_input(entry) for entry in data]
        elif isinstance(data, str) or (
            isinstance(data, list) and all(isinstance(value, str) for value in data)
        ):
            return data
        else:
            raise MlflowException(
                f"An invalid type has been supplied: {_truncate_and_ellipsize(data, 100)} "
                f"(type: {type(data).__name__}). Please supply a Dict[str, str], str, List[str], "
                "or a List[Dict[str, str]] for a Text2Text Pipeline.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _parse_json_encoded_list(self, data, key_to_unpack):
        """
        Parses the complex input types for pipelines such as ZeroShotClassification in which
        the required input type is Dict[str, Union[str, List[str]]] wherein the list
        provided is encoded as JSON. This method unpacks that string to the required
        elements.
        """
        if isinstance(data, list):
            return [self._parse_json_encoded_list(entry, key_to_unpack) for entry in data]
        elif isinstance(data, dict):
            if key_to_unpack not in data:
                raise MlflowException(
                    "Invalid key in inference payload. The expected inference data key "
                    f"is: {key_to_unpack}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if isinstance(data[key_to_unpack], str):
                try:
                    return {
                        k: (json.loads(v) if k == key_to_unpack else v) for k, v in data.items()
                    }
                except json.JSONDecodeError:
                    return data
            elif isinstance(data[key_to_unpack], list):
                return data

    @staticmethod
    def _parse_json_encoded_dict_payload_to_dict(data, key_to_unpack):
        """
        Parses complex dict input types that have been json encoded. Pipelines like
        TableQuestionAnswering require such input types.
        """
        if isinstance(data, list):
            return [
                {
                    key: (
                        json.loads(value)
                        if key == key_to_unpack and isinstance(value, str)
                        else value
                    )
                    for key, value in entry.items()
                }
                for entry in data
            ]
        elif isinstance(data, dict):
            # This is to handle serving use cases as the DataFrame encapsulation converts
            # collections within rows to np.array type. In order to process this data through
            # the transformers.Pipeline API, we need to cast these arrays back to lists
            # and replace the single quotes with double quotes after extracting the
            # json-encoded `table` (a pandas DF) in order to convert it to a dict that
            # the TableQuestionAnsweringPipeline can accept and cast to a Pandas DataFrame.
            #
            # An example casting that occurs for this case when input to model serving is the
            # conversion of a user input of:
            #   '{"inputs": {"query": "What is the longest distance?",
            #                "table": {"Distance": ["1000", "10", "1"]}}}'
            # is converted to:
            #   [{'query': array('What is the longest distance?', dtype='<U29'),
            #     'table': array('{\'Distance\': [\'1000\', \'10\', \'1\']}', dtype='U<204')}]
            # which is an invalid input to the pipeline.
            # this method converts the input to:
            #   {'query': 'What is the longest distance?',
            #    'table': {'Distance': ['1000', '10', '1']}}
            # which is a valid input to the TableQuestionAnsweringPipeline.
            output = {}
            for key, value in data.items():
                if key == key_to_unpack:
                    if isinstance(value, np.ndarray):
                        output[key] = ast.literal_eval(value.item())
                    else:
                        output[key] = ast.literal_eval(value)
                else:
                    if isinstance(value, np.ndarray):
                        # This cast to np.ndarray occurs when more than one question is asked.
                        output[key] = value.item()
                    else:
                        # Otherwise, the entry does not need casting from a np.ndarray type to
                        # list as it is already a scalar string.
                        output[key] = value
            return output
        else:
            return {
                key: (
                    json.loads(value) if key == key_to_unpack and isinstance(value, str) else value
                )
                for key, value in data.items()
            }

    @staticmethod
    def _validate_str_or_list_str(data):
        if not isinstance(data, (str, list)):
            raise MlflowException(
                f"The input data is of an incorrect type. {type(data)} is invalid. "
                "Must be either string or List[str]",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif isinstance(data, list) and not all(isinstance(entry, str) for entry in data):
            raise MlflowException(
                "If supplying a list, all values must be of string type.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @staticmethod
    def _convert_cast_lists_from_np_back_to_list(data):
        """
        This handles the casting of dicts within lists from Pandas DF conversion within model
        serving back into the required Dict[str, List[str]] if this type matching occurs.
        Otherwise, it's a noop.
        """
        if not isinstance(data, list):
            # NB: applying a short-circuit return here to not incur runtime overhead with
            # type validation if the input is not a list
            return data
        elif not all(isinstance(value, dict) for value in data):
            return data
        else:
            parsed_data = []
            for entry in data:
                if all(isinstance(value, np.ndarray) for value in entry.values()):
                    parsed_data.append({key: value.tolist() for key, value in entry.items()})
                else:
                    parsed_data.append(entry)
            return parsed_data

    @staticmethod
    def is_base64_image(image):
        """Check whether input image is a base64 encoded"""

        try:
            b64_decoded_image = base64.b64decode(image)
            return (
                base64.b64encode(b64_decoded_image).decode("utf-8") == image
                or base64.encodebytes(b64_decoded_image).decode("utf-8") == image
            )
        except binascii.Error:
            return False

    def _convert_image_input(self, input_data):
        """
        Conversion utility for decoding the base64 encoded bytes data of a raw image file when
        parsed through model serving, if applicable. Direct usage of the pyfunc implementation
        outside of model serving will treat this utility as a noop.

        For reference, the expected encoding for input to Model Serving will be:

        import requests
        import base64

        response = requests.get("https://www.my.images/a/sound/file.jpg")
        encoded_image = base64.b64encode(response.content).decode("utf-8")

        inference_data = json.dumps({"inputs": [encoded_image]})

        or

        inference_df = pd.DataFrame(
        pd.Series([encoded_image], name="image_file")
        )
        split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
        split_json = json.dumps(split_dict)

        or

        records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
        records_json = json.dumps(records_dict)

        This utility will convert this JSON encoded, base64 encoded text back into bytes for
        input into the Image pipelines for inference.
        """

        def process_input_element(input_element):
            input_value = next(iter(input_element.values()))
            if isinstance(input_value, str) and not self.is_base64_image(input_value):
                self._validate_str_input_uri_or_file(input_value)
            return input_value

        if isinstance(input_data, list) and all(
            isinstance(element, dict) for element in input_data
        ):
            # Use a list comprehension for readability
            # the elimination of empty collection declarations
            return [process_input_element(element) for element in input_data]
        elif isinstance(input_data, str) and not self.is_base64_image(input_data):
            self._validate_str_input_uri_or_file(input_data)

        return input_data

    def _convert_audio_input(
        self, data: AudioInput | list[dict[int, list[AudioInput]]]
    ) -> AudioInput | list[AudioInput]:
        """
        Convert the input data into the format that the Transformers pipeline expects.

        Args:
            data: The input data to be converted. This can be one of the following:
                1. A single input audio data (bytes, numpy array, or a path or URI to an audio file)
                2. List of dictionaries, derived from Pandas DataFrame with `orient="records"`.
                   This is the outcome of the pyfunc signature validation for the audio input.
                   E.g. [{[0]: <audio data>}, {[1]: <audio data>}]

        Returns:
            A single or list of audio data.
        """
        if isinstance(data, list):
            data = [list(element.values())[0] for element in data]
            decoded = [self._decode_audio(audio) for audio in data]
            # Signature validation converts a single audio data into a list (via Pandas Series).
            # We have to unwrap it back not to confuse with batch processing.
            return decoded if len(decoded) > 1 else decoded[0]
        else:
            return self._decode_audio(data)

    def _decode_audio(self, audio: AudioInput) -> AudioInput:
        """
        Decode the audio data if it is base64 encoded bytes, otherwise no-op.
        """
        if isinstance(audio, str):
            # Input is an URI to the audio file to be processed.
            self._validate_str_input_uri_or_file(audio)
            return audio
        elif isinstance(audio, np.ndarray):
            # Input is a numpy array that contains floating point time series of the audio.
            return audio
        elif isinstance(audio, bytes):
            # Input is a bytes object. In model serving, the input audio data is b64encoded.
            # They are typically decoded before reaching here, but iff the inference payload
            # contains raw bytes in the key 'inputs', the upstream code will not decode the
            # bytes. Therefore, we need to decode the bytes here. For other cases like
            # 'dataframe_records' or 'dataframe_split', the bytes should be already decoded.
            if self.is_base64_audio(audio):
                return base64.b64decode(audio)
            else:
                return audio
        else:
            raise MlflowException(
                "Invalid audio data. Must be either bytes, str, or np.ndarray.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @staticmethod
    def is_base64_audio(audio: bytes) -> bool:
        """Check whether input audio is a base64 encoded"""
        try:
            return base64.b64encode(base64.b64decode(audio)) == audio
        except binascii.Error:
            return False

    @staticmethod
    def _validate_str_input_uri_or_file(input_str):
        """
        Validation of blob references to either audio or image files,
        if a string is input to the ``predict``
        method, perform validation of the string contents by checking for a valid uri or
        filesystem reference instead of surfacing the cryptic stack trace that is otherwise raised
        for an invalid uri input.
        """

        def is_uri(s):
            try:
                result = urlparse(s)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        valid_uri = os.path.isfile(input_str) or is_uri(input_str)

        if not valid_uri:
            if len(input_str) <= 20:
                data_str = f"Received: {input_str}"
            else:
                data_str = f"Received (truncated): {input_str[:20]}..."
            raise MlflowException(
                "An invalid string input was provided. String inputs to "
                "audio or image files must be either a file location or a uri."
                f"audio files must be either a file location or a uri. {data_str}",
                error_code=BAD_REQUEST,
            )

    def _format_prompt_template(self, input_data):
        """
        Wraps the input data in the specified prompt template. If no template is
        specified, or if the pipeline is an unsupported type, or if the input type
        is not a string or list of strings, then the input data is returned unchanged.
        """
        if not self.prompt_template:
            return input_data

        if self.pipeline.task not in _SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES:
            raise MlflowException(
                f"_format_prompt_template called on an unexpected pipeline type. "
                f"Expected one of: {_SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES}. "
                f"Received: {self.pipeline.task}"
            )

        if isinstance(input_data, str):
            return self.prompt_template.format(prompt=input_data)
        elif isinstance(input_data, list):
            # if every item is a string, then apply formatting to every item
            if all(isinstance(data, str) for data in input_data):
                return [self.prompt_template.format(prompt=data) for data in input_data]

        # throw for unsupported types
        raise MlflowException.invalid_parameter_value(
            "Prompt templating is only supported for data of type str or List[str]. "
            f"Got {type(input_data)} instead."
        )


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
    log_datasets=False,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    extra_tags=None,
):
    """
    This autologging integration is solely used for disabling spurious autologging of irrelevant
    sub-models that are created during the training and evaluation of transformers-based models.
    Autologging functionality is not implemented fully for the transformers flavor.
    """
    # A list of other flavors whose base autologging config would be automatically logged due to
    # training a model that would otherwise create a run and be logged internally within the
    # transformers-supported trainer calls.
    DISABLED_ANCILLARY_FLAVOR_AUTOLOGGING = ["sklearn", "tensorflow", "pytorch"]

    def train(original, *args, **kwargs):
        with disable_discrete_autologging(DISABLED_ANCILLARY_FLAVOR_AUTOLOGGING):
            return original(*args, **kwargs)

    with contextlib.suppress(ImportError):
        import setfit

        safe_patch(
            FLAVOR_NAME,
            (setfit.SetFitTrainer if Version(setfit.__version__).major < 1 else setfit.Trainer),
            "train",
            functools.partial(train),
            manage_run=False,
        )

    with contextlib.suppress(ImportError):
        import transformers

        classes = [transformers.Trainer, transformers.Seq2SeqTrainer]
        methods = ["train"]
        for clazz in classes:
            for method in methods:
                safe_patch(FLAVOR_NAME, clazz, method, functools.partial(train), manage_run=False)


def _get_prompt_template(model_path):
    if not os.path.exists(model_path):
        raise MlflowException(
            f'Could not find an "{MLMODEL_FILE_NAME}" configuration file at "{model_path}"',
            RESOURCE_DOES_NOT_EXIST,
        )

    model_conf = Model.load(model_path)
    if model_conf.metadata:
        return model_conf.metadata.get(FlavorKey.PROMPT_TEMPLATE)

    return None


def _validate_prompt_template(prompt_template):
    if prompt_template is None:
        return

    if not isinstance(prompt_template, str):
        raise MlflowException(
            f"Argument `prompt_template` must be a string, received {type(prompt_template)}",
            INVALID_PARAMETER_VALUE,
        )

    format_args = [
        tup[1] for tup in string.Formatter().parse(prompt_template) if tup[1] is not None
    ]

    # expect there to only be one format arg, and for that arg to be "prompt"
    if format_args != ["prompt"]:
        raise MlflowException.invalid_parameter_value(
            "Argument `prompt_template` must be a string with a single format arg, 'prompt'. "
            "For example: 'Answer the following question in a friendly tone. Q: {prompt}. A:'\n"
            f"Received {prompt_template}. "
        )
