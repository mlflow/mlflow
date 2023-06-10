import ast
import base64
import binascii
import contextlib
import functools
from functools import lru_cache
import json
import logging
import numpy as np
import os
import pathlib
import pandas as pd
import re
from typing import Union, List, Optional, Dict, Any, NamedTuple
from urllib.parse import urlparse

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import ModelInputExample, Model, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from mlflow.types.utils import _validate_input_dictionary_contains_only_strings_and_lists_of_strings
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.docstring_utils import (
    format_docstring,
    LOG_MODEL_PARAM_DOCS,
    docstring_version_compatibility_warning,
)
from mlflow.utils.environment import _find_duplicate_requirements
from mlflow.environment_variables import (
    MLFLOW_DEFAULT_PREDICTION_DEVICE,
    MLFLOW_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES,
    MLFLOW_HUGGINGFACE_USE_DEVICE_MAP,
    MLFLOW_HUGGINGFACE_DEVICE_MAP_STRATEGY,
    MLFLOW_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE,
    MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE,
)
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _CONDA_ENV_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _process_conda_env,
    _process_pip_requirements,
    _CONSTRAINTS_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
    _download_artifact_from_uri,
    _get_flavor_configuration,
    _get_flavor_configuration_from_uri,
    _add_code_from_conf_to_system_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "transformers"

_CARD_TEXT_FILE_NAME = "model_card.md"
_CARD_DATA_FILE_NAME = "model_card_data.yaml"
_COMPONENTS_BINARY_KEY = "components"
_FEATURE_EXTRACTOR_KEY = "feature_extractor"
_FEATURE_EXTRACTOR_TYPE_KEY = "feature_extractor_type"
_FRAMEWORK_KEY = "framework"
_IMAGE_PROCESSOR_KEY = "image_processor"
_IMAGE_PROCESSOR_TYPE_KEY = "image_processor_type"
_INFERENCE_CONFIG_BINARY_KEY = "inference_config.txt"
_INSTANCE_TYPE_KEY = "instance_type"
_MODEL_KEY = "model"
_MODEL_BINARY_KEY = "model_binary"
_MODEL_TYPE_KEY = "model_type"
_MODEL_BINARY_FILE_NAME = "model"
_MODEL_PATH_OR_NAME_KEY = "source_model_name"
_PIPELINE_MODEL_TYPE_KEY = "pipeline_model_type"
_PROCESSOR_KEY = "processor"
_PROCESSOR_TYPE_KEY = "processor_type"
_SUPPORTED_RETURN_TYPES = {"pipeline", "components"}
# The default device id for CPU is -1 and GPU IDs are ordinal starting at 0, as documented here:
# https://huggingface.co/transformers/v4.7.0/main_classes/pipelines.html
_TRANSFORMERS_DEFAULT_CPU_DEVICE_ID = -1
_TRANSFORMERS_DEFAULT_GPU_DEVICE_ID = 0
_TASK_KEY = "task"
_TOKENIZER_KEY = "tokenizer"
_TOKENIZER_TYPE_KEY = "tokenizer_type"
_TORCH_DTYPE_KEY = "torch_dtype"
_METADATA_PIPELINE_SCALAR_CONFIG_KEYS = {_FRAMEWORK_KEY}
_SUPPORTED_SAVE_KEYS = {_MODEL_KEY, _TOKENIZER_KEY, _FEATURE_EXTRACTOR_KEY, _IMAGE_PROCESSOR_KEY}

_logger = logging.getLogger(__name__)


def _model_packages(model) -> List[str]:
    """
    Determines which pip libraries should be included based on the base model engine
    type.

    :param model: The model instance to be saved in order to provide the required underlying
                  deep learning execution framework dependency requirements.
    :return: A list of strings representing the underlying engine-specific dependencies
    """
    engine = _get_engine_type(model)
    if engine == "torch":
        return ["torch", "torchvision", "accelerate"]
    else:
        return [engine]


@experimental
def get_default_pip_requirements(model) -> List[str]:
    """
    :param model: The model instance to be saved in order to provide the required underlying
                  deep learning execution framework dependency requirements. Note that this must
                  be the actual model instance and not a Pipeline.
    :return: A list of default pip requirements for MLflow Models that have been produced with the
             ``transformers`` flavor. Calls to :py:func:`save_model()` and :py:func:`log_model()`
             produce a pip environment that contain these requirements at a minimum.
    """

    from transformers import TFPreTrainedModel, FlaxPreTrainedModel, PreTrainedModel

    if not isinstance(model, (TFPreTrainedModel, FlaxPreTrainedModel, PreTrainedModel)):
        raise MlflowException(
            "The supplied model type is unsupported. The model must be one of: "
            "PreTrainedModel, TFPreTrainedModel, or FlaxPreTrainedModel",
            error_code=INVALID_PARAMETER_VALUE,
        )
    try:
        base_reqs = ["transformers", *_model_packages(model)]
        return [_get_pinned_requirement(module) for module in base_reqs]
    except Exception as e:
        dependencies = [
            _get_pinned_requirement(module)
            for module in ["transformers", "torch", "torchvision", "tensorflow"]
        ]
        _logger.warning(
            "Could not infer model execution engine type due to huggingface_hub not "
            "being installed or unable to connect in online mode. Adding full "
            f"dependency chain: {dependencies}. \nFailure cause: {str(e)}"
        )
        return dependencies


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
        if _MODEL_KEY not in transformers_model:
            raise MlflowException(
                f"The 'transformers_model' dictionary must have an entry for {_MODEL_KEY}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        model = transformers_model[_MODEL_KEY]
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


@experimental
def get_default_conda_env(model):
    """
    :return: The default Conda environment for MLflow Models produced with the ``transformers``
             flavor, based on the model instance framework type of the model to be logged.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(model))


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    transformers_model,
    path: str,
    processor=None,
    task: Optional[str] = None,
    model_card=None,
    inference_config: Optional[Dict[str, Any]] = None,
    code_paths: Optional[List[str]] = None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    conda_env=None,
    metadata: Dict[str, Any] = None,
    **kwargs,
) -> None:
    """
    Save a trained transformers model to a path on the local file system.

    :param transformers_model:
        A trained transformers `Pipeline` or a dictionary that maps required components of a
        pipeline to the named keys of ["model", "image_processor", "tokenizer",
        "feature_extractor"]. The `model` key in the dictionary must map to a value that inherits
        from `PreTrainedModel`, `TFPreTrainedModel`, or `FlaxPreTrainedModel`.
        All other component entries in the dictionary must support the defined task type that is
        associated with the base model type configuration.

        An example of supplying component-level parts of a transformers model is shown below:

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

        An example of submitting a `Pipeline` from a default pipeline instantiation:

        .. code-block:: python

          from transformers import pipeline

          qa_pipe = pipeline("question-answering", "csarron/mobilebert-uncased-squad-v2")

          with mlflow.start_run():
              mlflow.transformers.save_model(
                  transformers_model=qa_pipe,
                  path="path/to/save/model",
              )

    :param path: Local path destination for the serialized model to be saved.
    :param processor: An optional ``Processor`` subclass object. Some model architectures,
                      particularly multi-modal types, utilize Processors to combine text
                      encoding and image or audio encoding in a single entrypoint.

                      .. Note:: If a processor is supplied when saving a model, the
                                model will be unavailable for loading as a ``Pipeline`` or for
                                usage with pyfunc inference.

    :param task: The transformers-specific task type of the model. These strings are utilized so
                 that a pipeline can be created with the appropriate internal call architecture
                 to meet the needs of a given model. If this argument is not specified, the
                 pipeline utilities within the transformers library will be used to infer the
                 correct task type. If the value specified is not a supported type within the
                 version of transformers that is currently installed, an Exception will be thrown.
    :param model_card: An Optional `ModelCard` instance from `huggingface-hub`. If provided, the
                       contents of the model card will be saved along with the provided
                       `transformers_model`. If not provided, an attempt will be made to fetch
                       the card from the base pretrained model that is provided (or the one that is
                       included within a provided `Pipeline`).

                       .. Note:: In order for a ModelCard to be fetched (if not provided),
                                 the huggingface_hub package must be installed and the version
                                 must be >=0.10.0

    :param inference_config:
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
                task=task, tokenizer=AutoTokenizer.from_pretrained(architecture), model=architecture
            )

            # Validate that the overrides function
            prompts = ["Generative models are", "I'd like a coconut so that I can"]

            # validation of config prior to save or log
            inference_config = {
                "top_k": 2,
                "num_beams": 5,
                "max_length": 30,
                "temperature": 0.62,
                "top_p": 0.85,
                "repetition_penalty": 1.15,
            }

            # Verify that no exceptions are thrown
            sentence_pipeline(prompts, **inference_config)

            mlflow.transformers.save_model(
                transformers_model=sentence_pipeline,
                path="/path/for/model",
                task=task,
                inference_config=inference_config,
            )

    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: An MLflow model object that specifies the flavor that this model is being
                         added to.
    :param signature: A Model Signature object that describes the input and output Schema of the
                      model. The model signature can be inferred using `infer_signature` function
                      of `mlflow.models.signature`.
                      Example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
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
    :param input_example: An example of valid input that the model can accept. The example can be
                          used as a hint of what data to feed the model. The given example will be
                          converted to a `Pandas DataFrame` and then serialized to JSON using the
                          `Pandas` split-oriented format. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param conda_env: {{ conda_env }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.

    :param kwargs: Optional additional configurations for transformers serialization.
    :return: None
    """
    import transformers

    _validate_transformers_model_dict(transformers_model)

    if isinstance(transformers_model, dict):
        transformers_model = _TransformersModel.from_dict(**transformers_model)

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = pathlib.Path(path).absolute()

    _validate_and_prepare_target_save_path(str(path))

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, str(path))

    resolved_task = _get_or_infer_task_type(transformers_model, task)

    if not isinstance(transformers_model, transformers.Pipeline):
        built_pipeline = _build_pipeline_from_model_input(transformers_model, resolved_task)
    else:
        built_pipeline = transformers_model

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        input_example = _format_input_example_for_special_cases(input_example, built_pipeline)
        _save_example(mlflow_model, input_example, str(path))
    if metadata is not None:
        mlflow_model.metadata = metadata

    flavor_conf = _generate_base_flavor_configuration(built_pipeline, resolved_task)

    components = _record_pipeline_components(built_pipeline)

    if components:
        flavor_conf.update(**components)

    if processor:
        flavor_conf.update({_PROCESSOR_TYPE_KEY: _get_instance_type(processor)})

    # Save the model object
    built_pipeline.model.save_pretrained(
        save_directory=path.joinpath(_MODEL_BINARY_FILE_NAME),
        max_shard_size=MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE.get(),
    )

    # Save the components explicitly to the components directory
    if components:
        _save_components(
            root_path=path.joinpath(_COMPONENTS_BINARY_KEY),
            component_config=components,
            pipeline=built_pipeline,
            processor=processor,
            inference_config=inference_config,
        )

    # Get the model card from either the argument or the HuggingFace marketplace
    card_data = model_card if model_card is not None else _fetch_model_card(transformers_model)

    # If the card data can be acquired, save the text and the data separately
    _write_card_data(card_data, path)

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}

    # Only allow a subset of task types to have a pyfunc definition.
    # Currently supported types are NLP-based language tasks which have a pipeline definition
    # consisting exclusively of a Model and a Tokenizer.
    if _should_add_pyfunc_to_model(built_pipeline):
        # For pyfunc supported models, if a signature is not supplied, infer the signature
        # from the input_example if provided, otherwise, apply a generic signature.
        if not signature:
            mlflow_model.signature = _get_default_pipeline_signature(built_pipeline, input_example)

        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.transformers",
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_dir_subpath,
            **model_bin_kwargs,
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
    flavor_conf.update(**model_bin_kwargs)
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        transformers_version=transformers.__version__,
        code=code_dir_subpath,
        **flavor_conf,
    )
    mlflow_model.save(str(path.joinpath(MLMODEL_FILE_NAME)))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(transformers_model.model)
            inferred_reqs = infer_pip_requirements(str(path), FLAVOR_NAME, fallback=default_reqs)
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    if duplicates := _find_duplicate_requirements(pip_requirements):
        _logger.warning(
            "Duplicate packages are present within the pip requirements. Duplicate packages: "
            f"{duplicates}. Please manually specify the requirements by using the "
            "`pip_requirements` argument in order to prevent unexpected installation "
            "issues for this model."
        )

    with path.joinpath(_CONDA_ENV_FILE_NAME).open("w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(str(path.joinpath(_CONSTRAINTS_FILE_NAME)), "\n".join(pip_constraints))

    write_to(str(path.joinpath(_REQUIREMENTS_FILE_NAME)), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(str(path.joinpath(_PYTHON_ENV_FILE_NAME)))


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    transformers_model,
    artifact_path: str,
    processor=None,
    task: Optional[str] = None,
    model_card=None,
    inference_config: Optional[Dict[str, Any]] = None,
    code_paths: Optional[List[str]] = None,
    registered_model_name: str = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    conda_env=None,
    metadata: Dict[str, Any] = None,
    **kwargs,
):
    """
    Log a ``transformers`` object as an MLflow artifact for the current run.

    :param transformers_model:
        A trained transformers `Pipeline` or a dictionary that maps required components of a
        pipeline to the named keys of ["model", "image_processor", "tokenizer",
        "feature_extractor"]. The `model` key in the dictionary must map to a value that inherits
        from `PreTrainedModel`, `TFPreTrainedModel`, or `FlaxPreTrainedModel`.
        All other component entries in the dictionary must support the defined task type that is
        associated with the base model type configuration.

        An example of supplying component-level parts of a transformers model is shown below:

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
                  artifact_path="my_model",
              )

        An example of submitting a `Pipeline` from a default pipeline instantiation:

        .. code-block:: python

          from transformers import pipeline

          qa_pipe = pipeline("question-answering", "csarron/mobilebert-uncased-squad-v2")

          with mlflow.start_run():
              mlflow.transformers.log_model(
                  transformers_model=qa_pipe,
                  artifact_path="my_pipeline",
              )

    :param artifact_path: Local path destination for the serialized model to be saved.
    :param processor: An optional ``Processor`` subclass object. Some model architectures,
                  particularly multi-modal types, utilize Processors to combine text
                  encoding and image or audio encoding in a single entrypoint.

                  .. Note:: If a processor is supplied when logging a model, the
                            model will be unavailable for loading as a ``Pipeline`` or for usage
                            with pyfunc inference.

    :param task: The transformers-specific task type of the model. These strings are utilized so
                 that a pipeline can be created with the appropriate internal call architecture
                 to meet the needs of a given model. If this argument is not specified, the
                 pipeline utilities within the transformers library will be used to infer the
                 correct task type. If the value specified is not a supported type within the
                 version of transformers that is currently installed, an Exception will be thrown.
    :param model_card: An Optional `ModelCard` instance from `huggingface-hub`. If provided, the
                       contents of the model card will be saved along with the provided
                       `transformers_model`. If not provided, an attempt will be made to fetch
                       the card from the base pretrained model that is provided (or the one that is
                       included within a provided `Pipeline`).

                       .. Note:: In order for a ModelCard to be fetched (if not provided),
                                 the huggingface_hub package must be installed and the version
                                 must be >=0.10.0

    :param inference_config:
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
              task=task, tokenizer=AutoTokenizer.from_pretrained(architecture), model=architecture
          )

          # Validate that the overrides function
          prompts = ["Generative models are", "I'd like a coconut so that I can"]

          # validation of config prior to save or log
          inference_config = {
              "top_k": 2,
              "num_beams": 5,
              "max_length": 30,
              "temperature": 0.62,
              "top_p": 0.85,
              "repetition_penalty": 1.15,
          }

          # Verify that no exceptions are thrown
          sentence_pipeline(prompts, **inference_config)

          with mlflow.start_run():
              mlflow.transformers.log_model(
                  transformers_model=sentence_pipeline,
                  artifact_path="my_sentence_generator",
                  task=task,
                  inference_config=inference_config,
              )

    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param signature: A Model Signature object that describes the input and output Schema of the
                      model. The model signature can be inferred using `infer_signature` function
                      of `mlflow.models.signature`.
                      Example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        from mlflow.transformers import generate_signature_output
                        from transformers import pipeline

                        en_to_de = pipeline("translation_en_to_de")

                        data = "MLflow is great!"
                        output = generate_signature_output(en_to_de, data)
                        signature = infer_signature(data, output)

                        with mlflow.start_run() as run:
                            mlflow.transformers.log_model(
                                transformers_model=en_to_de,
                                artifact_path="english_to_german_translator",
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a ``Pandas DataFrame`` and
                          then serialized to json using the ``Pandas`` split-oriented format.
                          Bytes are base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version
                                   to finish being created and is in ``READY`` status.
                                   By default, the function waits for five minutes.
                                   Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param conda_env: {{ conda_env }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param kwargs: Additional arguments for :py:class:`mlflow.models.model.Model`
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.transformers,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        metadata=metadata,
        transformers_model=transformers_model,
        processor=processor,
        task=task,
        model_card=model_card,
        inference_config=inference_config,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
def load_model(model_uri: str, dst_path: str = None, return_type="pipeline", device=None, **kwargs):
    """
    Load a ``transformers`` object from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``mlflow-artifacts:/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to utilize for downloading the model artifact.
                     This directory must already exist if provided. If unspecified, a local output
                     path will be created.
    :param return_type: A return type modifier for the stored ``transformers`` object.
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
    :param device: The device on which to load the model. Default is None. Use 0 to
                   load to the default GPU.
    :param kwargs: Optional configuration options for loading of a ``transformers`` object.
                   For information on parameters and their usage, see
                   `transformers documentation <https://huggingface.co/docs/transformers/index>`_.
    :return: A ``transformers`` model instance or a dictionary of components
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

    if return_type == "pipeline" and _PROCESSOR_TYPE_KEY in flavor_config:
        raise MlflowException(
            "This model has been saved with a processor. Processor objects are "
            "not compatible with Pipelines. Please load this model by specifying "
            "the 'return_type'='components'.",
            error_code=BAD_REQUEST,
        )

    _add_code_from_conf_to_system_path(local_model_path, flavor_config)

    return _load_model(local_model_path, flavor_config, return_type, device, **kwargs)


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


def _try_load_model_with_device(model_instance, model_path, device, conf):
    load_model_conf = {}
    # Assume if torch_dtype was specified in the conf, then it must be with a
    # pipeline for which it's compatible.
    if _TORCH_DTYPE_KEY in conf:
        load_model_conf[_TORCH_DTYPE_KEY] = conf[_TORCH_DTYPE_KEY]

    try:
        load_model_conf["device"] = device
        model = model_instance.from_pretrained(model_path, **load_model_conf)
    except (ValueError, TypeError, NotImplementedError):
        _logger.warning("Could not specify device parameter for this pipeline type")
        load_model_conf.pop("device", None)
        model = model_instance.from_pretrained(model_path, **load_model_conf)
    return model


def _load_model(path: str, flavor_config, return_type: str, device=None, **kwargs):
    """
    Loads components from a locally serialized ``Pipeline`` object.
    """
    import transformers

    model_instance = getattr(transformers, flavor_config[_PIPELINE_MODEL_TYPE_KEY])
    local_path = pathlib.Path(path)
    model_path = local_path.joinpath(flavor_config.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME))
    conf = {
        "task": flavor_config[_TASK_KEY],
    }

    if device is None:
        if MLFLOW_DEFAULT_PREDICTION_DEVICE.get():
            try:
                device = int(MLFLOW_DEFAULT_PREDICTION_DEVICE.get())
            except ValueError:
                device = _TRANSFORMERS_DEFAULT_CPU_DEVICE_ID
        elif is_gpu_available():
            device = _TRANSFORMERS_DEFAULT_GPU_DEVICE_ID
    # Note that we don't set the device in the conf yet because device is
    # incompatible with device_map.
    accelerate_model_conf = {}
    if MLFLOW_HUGGINGFACE_USE_DEVICE_MAP.get():
        device_map_strategy = MLFLOW_HUGGINGFACE_DEVICE_MAP_STRATEGY.get()
        conf["device_map"] = device_map_strategy
        accelerate_model_conf["device_map"] = device_map_strategy
        # Cannot use device with device_map
        device = None

    if device is not None:
        conf["device"] = device
        accelerate_model_conf["device"] = device

    if _TORCH_DTYPE_KEY in flavor_config or _TORCH_DTYPE_KEY in kwargs:
        if _TORCH_DTYPE_KEY in kwargs:
            dtype_val = kwargs[_TORCH_DTYPE_KEY]
        else:
            dtype_val = _deserialize_torch_dtype_if_exists(flavor_config)
        conf[_TORCH_DTYPE_KEY] = dtype_val
        accelerate_model_conf[_TORCH_DTYPE_KEY] = dtype_val

    accelerate_model_conf["low_cpu_mem_usage"] = MLFLOW_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE.get()

    if not MLFLOW_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES.get():
        try:
            model = model_instance.from_pretrained(model_path, **accelerate_model_conf)
        except (ValueError, TypeError, NotImplementedError, ImportError):
            # NB: ImportError is caught here in the event that `accelerate` is not installed
            # on the system, which will raise if `low_cpu_mem_usage` is set or the argument
            # `device_map` is set and accelerate is not installed.
            model = _try_load_model_with_device(model_instance, model_path, device, conf)
    else:
        model = _try_load_model_with_device(model_instance, model_path, device, conf)

    conf["model"] = model

    if _PROCESSOR_TYPE_KEY in flavor_config:
        conf[_PROCESSOR_KEY] = _load_component(
            local_path, _PROCESSOR_KEY, flavor_config[_PROCESSOR_TYPE_KEY]
        )

    for component_key in flavor_config[_COMPONENTS_BINARY_KEY]:
        component_type_key = f"{component_key}_type"
        component_type = flavor_config[component_type_key]
        conf[component_key] = _load_component(local_path, component_key, component_type)

    for key in _METADATA_PIPELINE_SCALAR_CONFIG_KEYS:
        if key in flavor_config:
            conf[key] = flavor_config[key]

    if return_type == "pipeline":
        conf.update(**kwargs)
        return transformers.pipeline(**conf)
    elif return_type == "components":
        return conf


@lru_cache
def _torch_dype_mapping():
    """
    Memoized torch data type mapping from the torch primary datatypes for use in deserializing the
    saved pipeline parameter `torch_dtype`
    """
    try:
        import torch

        return {
            str(dtype): dtype
            for name, dtype in torch.__dict__.items()
            if isinstance(dtype, torch.dtype)
        }
    except ImportError as e:
        raise MlflowException(
            "Unable to determine if the value supplied by the argument "
            "torch_dtype is valid since torch is not installed.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


def _deserialize_torch_dtype_if_exists(flavor_config):
    """
    Convert the string-encoded `torch_dtype` pipeline argument back to the correct `torch.dtype`
    instance value for applying to a loaded pipeline instance.
    """

    return _torch_dype_mapping()[flavor_config["torch_dtype"]]


def _fetch_model_card(model_or_pipeline):
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

    model = model_or_pipeline.model

    if hasattr(hub, "ModelCard"):
        try:
            return hub.ModelCard.load(model.name_or_path)
        except Exception as e:
            _logger.warning(f"The model card could not be retrieved from the hub due to {e}")
    else:
        _logger.warning(
            f"The version of huggingface_hub that is installed does not provide "
            f"ModelCard functionality. You have version {hub.__version__} installed. "
            f"Update huggingface_hub to >= '0.10.0' to retrieve the ModelCard data."
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


def _build_pipeline_from_model_input(model, task: str):
    """
    Utility for generating a pipeline from component parts. If required components are not
    specified, use the transformers library pipeline component validation to force raising an
    exception. The underlying Exception thrown in transformers is verbose enough for diagnosis.
    """
    from transformers import pipeline

    pipeline_config = model.to_dict()
    pipeline_config.update({"task": task})
    try:
        return pipeline(**pipeline_config)
    except Exception as e:
        raise MlflowException(
            "The provided model configuration cannot be created as a Pipeline. "
            "Please verify that all required and compatible components are "
            "specified with the correct keys.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


def _record_pipeline_components(pipeline) -> Dict[str, Any]:
    """
    Utility for recording which components are present in either the generated pipeline iff the
    supplied save object is not a pipeline or the components of the supplied pipeline object.
    """
    components_conf = {}
    components = []
    for attr, key in [
        ("feature_extractor", _FEATURE_EXTRACTOR_TYPE_KEY),
        ("tokenizer", _TOKENIZER_TYPE_KEY),
        ("image_processor", _IMAGE_PROCESSOR_TYPE_KEY),
    ]:
        component = getattr(pipeline, attr, None)
        if component is not None:
            components_conf.update({key: _get_instance_type(component)})
            components.append(attr)
    if components:
        components_conf.update({_COMPONENTS_BINARY_KEY: components})
    return components_conf


def _save_components(
    root_path: pathlib.Path, component_config: Dict[str, Any], pipeline, processor, inference_config
):
    """
    Saves non-model pipeline components.
    """
    component_types = component_config.get(_COMPONENTS_BINARY_KEY, [])
    for component_name in component_types:
        component = getattr(pipeline, component_name)
        component.save_pretrained(root_path.joinpath(component_name))
    if processor:
        processor.save_pretrained(root_path.joinpath(_PROCESSOR_KEY))
    if inference_config:
        root_path.joinpath(_INFERENCE_CONFIG_BINARY_KEY).write_text(json.dumps(inference_config))


def _load_component(root_path: pathlib.Path, component_key: str, component_type):
    """
    Loads an individual component object from local disk.
    """
    import transformers

    components_dir = root_path.joinpath(_COMPONENTS_BINARY_KEY)
    component_path = components_dir.joinpath(component_key)
    component_instance = getattr(transformers, component_type)
    return component_instance.from_pretrained(component_path)


def _generate_base_flavor_configuration(
    pipeline,
    task: str,
) -> Dict[str, str]:
    """
    Generates the base flavor metadata needed for reconstructing a pipeline from saved
    components. This is important because the ``Pipeline`` class does not have a loader
    functionality. The serialization of a Pipeline saves the model, configurations, and
    metadata for ``FeatureExtractor``s, ``Processor``s, and ``Tokenizer``s exclusively.
    This function extracts key information from the submitted model object so that the precise
    instance types can be loaded correctly.
    """

    _validate_transformers_task_type(task)

    flavor_configuration = {
        _TASK_KEY: task,
        _INSTANCE_TYPE_KEY: _get_instance_type(pipeline),
        _MODEL_PATH_OR_NAME_KEY: _get_base_model_architecture(pipeline),
        _PIPELINE_MODEL_TYPE_KEY: _get_instance_type(pipeline.model),
    }

    # Extract and add to the configuration the scalar serializable arguments for pipeline args
    for arg_key in _METADATA_PIPELINE_SCALAR_CONFIG_KEYS:
        if entry := _get_scalar_argument_from_pipeline(pipeline, arg_key):
            flavor_configuration[arg_key] = entry

    # Extract a serialized representation of torch_dtype if provided
    if torch_dtype := _extract_torch_dtype_if_set(pipeline):
        flavor_configuration[_TORCH_DTYPE_KEY] = torch_dtype

    return flavor_configuration


def _get_scalar_argument_from_pipeline(pipeline, arg_key):
    """
    Retrieve provided pipeline arguments for the purposes of instantiating a pipeline object upon
    loading.
    """

    return getattr(pipeline, arg_key, None)


def _extract_torch_dtype_if_set(pipeline):
    """
    Extract the torch datatype argument if set and return as a string encoded value.
    """
    if torch_dtype := getattr(pipeline, _TORCH_DTYPE_KEY, None):
        return str(torch_dtype)


def _get_or_infer_task_type(model, task: Optional[str] = None) -> str:
    """
    Validates that a supplied task type is supported by the ``transformers`` library if supplied,
    else, if not supplied, infers the appropriate task type based on the model type.
    """
    if task:
        _validate_transformers_task_type(task)
    else:
        task = _infer_transformers_task_type(model)
    return task


def _infer_transformers_task_type(model) -> str:
    """
    Performs inference of the task type, used in generating a pipeline object based on the
    underlying model's intended use case. This utility relies on the definitions within the
    transformers pipeline construction utility functions.

    :param model: Either the model or the Pipeline object that the task will be extracted or
                  inferred from
    :return: The task type string
    """
    from transformers import Pipeline
    from transformers.pipelines import get_task

    if isinstance(model, Pipeline):
        return model.task
    elif isinstance(model, _TransformersModel):
        try:
            return get_task(model.model.name_or_path)
        except Exception as e:
            raise MlflowException(
                "The task type cannot be inferred from the submitted Pipeline or dictionary of "
                "model components. Please provide the task type explicitly when saving or logging "
                "this submitted Pipeline or dictionary of components.",
                error_code=BAD_REQUEST,
            ) from e
    else:
        raise MlflowException(
            f"The provided model type: {type(model)} is not supported. "
            "Supported model types are: Pipeline or a dictionary with specific named keys. "
            "Run `help(mlflow.transformers.save_model)` to see details of supported types.",
            error_code=BAD_REQUEST,
        )


def _validate_transformers_task_type(task: str) -> None:
    """
    Validates that a given ``task`` type is supported by the ``transformers`` library and has been
    registered in the hub.
    """
    from transformers.pipelines import get_supported_tasks

    valid_tasks = get_supported_tasks()

    if task not in valid_tasks and not task.startswith("translation"):
        raise MlflowException(
            f"The task provided is invalid. '{task}' is not a supported task. "
            f"Must be one of the registered tasks: {valid_tasks}",
            error_code=BAD_REQUEST,
        )


def _get_engine_type(model):
    """
    Determines the underlying execution engine for the model based on the 3 currently supported
    deep learning framework backends: ``tensorflow``, ``torch``, or ``flax``.
    """
    from transformers import PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel

    for cls in model.__class__.__mro__:
        if issubclass(cls, TFPreTrainedModel):
            return "tensorflow"
        elif issubclass(cls, PreTrainedModel):
            return "torch"
        elif issubclass(cls, FlaxPreTrainedModel):
            return "flax"


def _get_base_model_architecture(model_or_pipeline):
    """
    Extracts the base model architecture type from a submitted model.
    """
    from transformers import Pipeline

    if isinstance(model_or_pipeline, Pipeline):
        return model_or_pipeline.model.name_or_path
    else:
        return model_or_pipeline[_MODEL_KEY].name_or_path


def _get_instance_type(obj):
    """
    Utility for extracting the saved object type or, if the `base` argument is set to `True`,
    the base ABC type of the model.
    """
    return obj.__class__.__name__


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
        "ImageClassificationPipeline",
        "ImageSegmentationPipeline",
        "DepthEstimationPipeline",
        "ObjectDetectionPipeline",
        "VideoClassificationPipeline",
        "ZeroShotImageClassificationPipeline",
        "ZeroShotObjectDetectionPipeline",
        "ZeroShotAudioClassificationPipeline",
    ]

    impermissible_attrs = {"image_processor"}

    for attr in impermissible_attrs:
        if getattr(pipeline, attr, None) is not None:
            return False
    for model_type in exclusion_model_types:
        if hasattr(transformers, model_type):
            if isinstance(pipeline.model, getattr(transformers, model_type)):
                return False
    if type(pipeline).__name__ in exclusion_pipeline_types:
        return False
    return True


def _format_input_example_for_special_cases(input_example, pipeline):
    """
    Handles special formatting for specific types of Pipelines so that the displayed example
    reflects the correct example input structure that mirrors the behavior of the input parsing
    for pyfunc.
    """
    import transformers

    if (
        isinstance(pipeline, transformers.ZeroShotClassificationPipeline)
        and isinstance(input_example, dict)
        and isinstance(input_example["candidate_labels"], list)
    ):
        input_example["candidate_labels"] = json.dumps(input_example["candidate_labels"])
    return input_example


def _get_default_pipeline_signature(pipeline, example=None) -> ModelSignature:
    """
    Assigns a default ModelSignature for a given Pipeline type that has pyfunc support. These
    default signatures should only be generated and assigned when saving a model iff the user
    has not supplied a signature.
    For signature inference in some Pipelines that support complex input types, an input example
    is needed.
    """

    import transformers

    if example:
        try:
            inference_pyfunc = _TransformersWrapper(pipeline)
            return infer_signature(example, inference_pyfunc.predict(example))
        except Exception as e:
            _logger.warning(
                "Attempted to generate a signature for the saved model or pipeline "
                f"but encountered an error: {e}"
            )
    else:
        if isinstance(
            pipeline,
            (
                transformers.TokenClassificationPipeline,
                transformers.ConversationalPipeline,
                transformers.TranslationPipeline,
                transformers.FillMaskPipeline,
                transformers.TextGenerationPipeline,
                transformers.Text2TextGenerationPipeline,
            ),
        ):
            return ModelSignature(
                inputs=Schema([ColSpec("string")]), outputs=Schema([ColSpec("string")])
            )
        elif isinstance(pipeline, transformers.TextClassificationPipeline):
            return ModelSignature(
                inputs=Schema([ColSpec("string")]),
                outputs=Schema([ColSpec("string", name="label"), ColSpec("double", name="score")]),
            )
        elif isinstance(pipeline, transformers.ZeroShotClassificationPipeline):
            return ModelSignature(
                inputs=Schema(
                    [
                        ColSpec("string", name="sequences"),
                        ColSpec("string", name="candidate_labels"),
                        ColSpec("string", name="hypothesis_template"),
                    ]
                ),
                outputs=Schema(
                    [
                        ColSpec("string", name="sequence"),
                        ColSpec("string", name="labels"),
                        ColSpec("double", name="scores"),
                    ]
                ),
            )
        elif isinstance(pipeline, transformers.AutomaticSpeechRecognitionPipeline):
            return ModelSignature(
                inputs=Schema([ColSpec("binary")]),
                outputs=Schema([ColSpec("string")]),
            )
        elif isinstance(pipeline, transformers.AudioClassificationPipeline):
            return ModelSignature(
                inputs=Schema([ColSpec("binary")]),
                outputs=Schema([ColSpec("double", name="score"), ColSpec("string", name="label")]),
            )
        elif isinstance(
            pipeline,
            (
                transformers.TableQuestionAnsweringPipeline,
                transformers.QuestionAnsweringPipeline,
            ),
        ):
            column_1 = None
            column_2 = None
            if isinstance(pipeline, transformers.TableQuestionAnsweringPipeline):
                column_1 = "query"
                column_2 = "table"
            elif isinstance(pipeline, transformers.QuestionAnsweringPipeline):
                column_1 = "question"
                column_2 = "context"
            return ModelSignature(
                inputs=Schema(
                    [
                        ColSpec("string", name=column_1),
                        ColSpec("string", name=column_2),
                    ]
                ),
                outputs=Schema([ColSpec("string")]),
            )
        elif isinstance(pipeline, transformers.FeatureExtractionPipeline):
            return ModelSignature(
                inputs=Schema([ColSpec("string")]),
                outputs=Schema([TensorSpec(np.dtype("float64"), [-1], "double")]),
            )
        else:
            _logger.warning(
                "An unsupported Pipeline type was supplied for signature inference. "
                "Either provide an `input_example` or generate a signature manually "
                "via `infer_signature` if you would like to have a signature recorded "
                "in the MLmodel file."
            )


class _TransformersModel(NamedTuple):
    """
    Type validator class for models that are submitted as a dictionary for saving and logging.
    Usage of this class should always leverage the type-checking from the class method
    'from_dict()' instead of the instance-based configuration that is utilized with instantiating
    a NamedTuple instance (it uses '__new__()' instead of an '__init__()'  dunder method, making
    type validation on instantiation overly complex if we were to support that approach).
    """

    # NB: Assigning Any type here to eliminate local imports. Type validation is performed when
    # calling the `from_dict` class method.
    model: Any
    tokenizer: Any = None
    feature_extractor: Any = None
    image_processor: Any = None
    processor: Any = None

    def to_dict(self):
        dict_repr = self._asdict()
        # NB: due to breaking changes in APIs, newer pipeline-supported argument keys are not
        # backwards compatible. If there isn't an instance present, do not return an empty
        # key to value mapping.
        return {name: obj for name, obj in dict_repr.items() if obj}

    @staticmethod
    def _build_exception_msg(obj, obj_name, valid_types):
        type_msg = (
            "one of: " + ", ".join([valid_type.__name__ for valid_type in valid_types])
            if isinstance(valid_types, tuple)
            else valid_types.__name__
        )
        return (
            f"The {obj_name} type submitted is not compatible with the transformers flavor: "
            f"'{type(obj).__name__}'. "
            f"The allowed types must inherit from {type_msg}."
        )

    @classmethod
    def _validate_submitted_types(
        cls, model, tokenizer, feature_extractor, image_processor, processor
    ):
        from transformers import (
            PreTrainedModel,
            TFPreTrainedModel,
            FlaxPreTrainedModel,
            PreTrainedTokenizerBase,
            FeatureExtractionMixin,
            ImageFeatureExtractionMixin,
            ImageProcessingMixin,
            ProcessorMixin,
        )

        validation = [
            (model, "model", (PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel)),
            (tokenizer, "tokenizer", PreTrainedTokenizerBase),
            (
                feature_extractor,
                "feature_extractor",
                (
                    FeatureExtractionMixin,
                    ImageFeatureExtractionMixin,
                    ProcessorMixin,
                    ImageProcessingMixin,
                ),
            ),
            (image_processor, "image_processor", ImageProcessingMixin),
            (processor, "processor", ProcessorMixin),
        ]
        invalid_types = []

        for arg, name, types in validation:
            if arg and not isinstance(arg, types):
                invalid_types.append(cls._build_exception_msg(arg, name, types))
        if invalid_types:
            raise MlflowException("\n".join(invalid_types), error_code=BAD_REQUEST)

    @classmethod
    def from_dict(
        cls,
        model,
        tokenizer=None,
        feature_extractor=None,
        image_processor=None,
        processor=None,
        **kwargs,
    ):
        cls._validate_submitted_types(
            model, tokenizer, feature_extractor, image_processor, processor
        )

        return _TransformersModel(model, tokenizer, feature_extractor, image_processor, processor)


def _get_inference_config(local_path):
    """
    Load the inference config if it was provided for use in the `_TransformersWrapper` pyfunc
    Model wrapper.
    """
    config_path = local_path.joinpath("inference_config.txt")
    if config_path.exists():
        return json.loads(config_path.read_text())


def _load_pyfunc(path):
    """
    Loads the model as pyfunc model
    """
    local_path = pathlib.Path(path)
    flavor_configuration = _get_flavor_configuration(local_path, FLAVOR_NAME)
    inference_config = _get_inference_config(local_path.joinpath(_COMPONENTS_BINARY_KEY))
    return _TransformersWrapper(
        _load_model(str(local_path), flavor_configuration, "pipeline"),
        flavor_configuration,
        inference_config,
    )


@experimental
def generate_signature_output(pipeline, data, inference_config=None):
    """
    Utility for generating the response output for the purposes of extracting an output signature
    for model saving and logging. This function simulates loading of a saved model or pipeline
    as a ``pyfunc`` model without having to incur a write to disk.

    :param pipeline: A ``transformers`` pipeline object. Note that component-level or model-level
                     inputs are not permitted for extracting an output example.
    :param data: An example input that is compatible with the given pipeline
    :param inference_config: Any additional inference configuration, provided as kwargs, to inform
                             the format of the output type from a pipeline inference call.
    :return: The output from the ``pyfunc`` pipeline wrapper's ``predict`` method
    """
    import transformers

    if not isinstance(pipeline, transformers.Pipeline):
        raise MlflowException(
            f"The pipeline type submitted is not a valid transformers Pipeline. "
            f"The type {type(pipeline).__name__} is not supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return _TransformersWrapper(pipeline=pipeline, inference_config=inference_config).predict(data)


class _TransformersWrapper:
    def __init__(self, pipeline, flavor_config=None, inference_config=None):
        self.pipeline = pipeline
        self.flavor_config = flavor_config
        self.inference_config = inference_config or {}
        self._conversation = None
        # NB: Current special-case custom pipeline types that have not been added to
        # the native-supported transformers package but require custom parsing:
        # InstructionTextGenerationPipeline [Dolly] https://huggingface.co/databricks/dolly-v2-12b
        #   (and all variants)
        self._supported_custom_generator_types = {"InstructionTextGenerationPipeline"}

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

    def predict(self, data, device=None):
        if isinstance(data, pd.DataFrame):
            input_data = self._convert_pandas_to_dict(data)
        elif isinstance(data, dict):
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
        elif isinstance(data, str):
            input_data = data
        elif isinstance(data, bytes):
            input_data = data
        elif isinstance(data, np.ndarray):
            input_data = data
        else:
            raise MlflowException(
                "Input data must be either a pandas.DataFrame, a string, List[str], "
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

        predictions = self._predict(input_data, device)

        return predictions

    def _predict(self, data, device):
        import transformers

        # NB: the ordering of these conditional statements matters. TranslationPipeline and
        # SummarizationPipeline both inherit from TextGenerationPipeline (they are subclasses)
        # in which the return data structure from their __call__ implementation is modified.
        if isinstance(self.pipeline, transformers.TranslationPipeline):
            self._validate_str_or_list_str(data)
            output_key = "translation_text"
        elif isinstance(self.pipeline, transformers.SummarizationPipeline):
            self._validate_str_or_list_str(data)
            output_key = "summary_text"
        elif isinstance(self.pipeline, transformers.Text2TextGenerationPipeline):
            data = self._parse_text2text_input(data)
            output_key = "generated_text"
        elif isinstance(self.pipeline, transformers.TextGenerationPipeline):
            self._validate_str_or_list_str(data)
            output_key = "generated_text"
        elif isinstance(self.pipeline, transformers.QuestionAnsweringPipeline):
            data = self._parse_question_answer_input(data)
            output_key = "answer"
        elif isinstance(self.pipeline, transformers.FillMaskPipeline):
            self._validate_str_or_list_str(data)
            output_key = "token_str"
        elif isinstance(self.pipeline, transformers.TextClassificationPipeline):
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
        elif isinstance(self.pipeline, transformers.ConversationalPipeline):
            output_key = None
            if not self._conversation:
                self._conversation = transformers.Conversation()
            self._conversation.add_user_input(data)
        elif type(self.pipeline).__name__ in self._supported_custom_generator_types:
            self._validate_str_or_list_str(data)
            output_key = "generated_text"
        elif isinstance(self.pipeline, transformers.AutomaticSpeechRecognitionPipeline):
            if self.inference_config.get("return_timestamps", None) in ["word", "char"]:
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
        # formatting output), but if `include_prompt` is set to False in the `inference_config`
        # option during model saving, excess newline characters and the fed-in prompt will be
        # trimmed out from the start of the response.
        include_prompt = self.inference_config.pop("include_prompt", True)
        # Optional stripping out of `\n` for specific generator pipelines.
        collapse_whitespace = self.inference_config.pop("collapse_whitespace", False)
        if device is not None:
            self.inference_config["device"] = device

        data = self._convert_cast_lists_from_np_back_to_list(data)

        # Generate inference data with the pipeline object
        if isinstance(self.pipeline, transformers.ConversationalPipeline):
            conversation_output = self.pipeline(self._conversation)
            return conversation_output.generated_responses[-1]

        if isinstance(
            self.pipeline,
            (
                transformers.AutomaticSpeechRecognitionPipeline,
                transformers.AudioClassificationPipeline,
            ),
        ):
            try:
                raw_output = self.pipeline(data, **self.inference_config)
            except ValueError as e:
                if "Malformed soundfile" in str(e):
                    raise MlflowException(
                        "Failed to process the input audio data. Either the audio file is "
                        "corrupted or a uri was passed in without overriding the default model "
                        "signature. If submitting a string uri, please ensure that the model has "
                        "been saved with a signature that defines a string input type.",
                        error_code=INVALID_PARAMETER_VALUE,
                    ) from e
                raise
        elif isinstance(data, dict):
            raw_output = self.pipeline(**data, **self.inference_config)
        else:
            raw_output = self.pipeline(data, **self.inference_config)

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
        elif isinstance(self.pipeline, transformers.FeatureExtractionPipeline):
            return self._parse_feature_extraction_output(raw_output)
        elif isinstance(self.pipeline, transformers.FillMaskPipeline):
            output = self._parse_list_of_multiple_dicts(raw_output, output_key)
        elif isinstance(self.pipeline, transformers.ZeroShotClassificationPipeline):
            return self._flatten_zero_shot_text_classifier_output_to_df(raw_output)
        elif isinstance(self.pipeline, transformers.TokenClassificationPipeline):
            output = self._parse_tokenizer_output(raw_output, output_key)
        elif isinstance(
            self.pipeline, transformers.AutomaticSpeechRecognitionPipeline
        ) and self.inference_config.get("return_timestamps", None) in ["word", "char"]:
            output = json.dumps(raw_output)
        elif isinstance(
            self.pipeline,
            (transformers.AudioClassificationPipeline, transformers.TextClassificationPipeline),
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

        data = self._coerce_exploded_dict_to_single_dict(data)
        data = self._parse_input_for_table_question_answering(data)
        data = self._parse_conversation_input(data)
        if (
            isinstance(
                self.pipeline,
                (
                    transformers.FillMaskPipeline,
                    transformers.TextGenerationPipeline,
                    transformers.TranslationPipeline,
                    transformers.TextClassificationPipeline,
                    transformers.SummarizationPipeline,
                    transformers.TokenClassificationPipeline,
                ),
            )
            and isinstance(data, list)
            and all(isinstance(entry, dict) for entry in data)
        ):
            return [list(entry.values())[0] for entry in data]
        else:
            return data

    def _parse_conversation_input(self, data):
        import transformers

        if not isinstance(self.pipeline, transformers.ConversationalPipeline):
            return data
        elif isinstance(data, str):
            return data
        elif isinstance(data, list) and all(isinstance(elem, dict) for elem in data):
            return next(iter(data[0].values()))
        elif isinstance(data, dict):
            # The conversation pipeline can only accept a single string at a time
            return next(iter(data.values()))

    def _parse_input_for_table_question_answering(self, data):
        import transformers

        if not isinstance(self.pipeline, transformers.TableQuestionAnsweringPipeline):
            return data

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

    def _coerce_exploded_dict_to_single_dict(self, data):
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

        [
          "We should order more pizzas to meet the demand.",
          "The venue size should be updated to handle the number of guests.",
        ]

        """
        import transformers

        if not isinstance(
            self.pipeline,
            transformers.TableQuestionAnsweringPipeline,
        ):
            return data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
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
            # the optional inference_config entry of `{"include_prompt": False}`.
            # By default, the prompt is included in the response.
            # Stripping out additional carriage returns (\n) is another additional optional flag
            # that can be set for these generator pipelines. It is off by default (False).
            if (
                not include_prompt
                and flavor_config[_INSTANCE_TYPE_KEY] in self._supported_custom_generator_types
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

    def _parse_lists_of_dict_to_list_of_str(self, output_data, target_dict_key) -> List[str]:
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
                            output_coll.append(
                                self._parse_lists_of_dict_to_list_of_str(value, target_dict_key)[0]
                            )
                elif isinstance(output, list):
                    output_coll.append(
                        self._parse_lists_of_dict_to_list_of_str(output, target_dict_key)[0]
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
        if isinstance(output_data[0], list):
            return [collection[0][target_dict_key] for collection in output_data]
        else:
            return [output_data[0][target_dict_key]]

    def _parse_list_output_for_multiple_candidate_pipelines(self, output_data):
        # NB: This will not continue to parse nested lists. Pipelines do not output complex
        # types that are greater than 2 levels deep so there is no need for more complex
        # traversal for outputs.
        if isinstance(output_data, list) and len(output_data) < 1:
            raise MlflowException(
                "The output of the pipeline contains no data.", error_code=BAD_REQUEST
            )

        if isinstance(output_data[0], list):
            return [
                self._parse_list_output_for_multiple_candidate_pipelines(x) for x in output_data
            ]
        else:
            return output_data[0]

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
        elif isinstance(data, str):
            return data
        elif isinstance(data, list) and all(isinstance(value, str) for value in data):
            return data
        else:
            raise MlflowException(
                "An invalid type has been supplied. Please supply a Dict[str, str], str, "
                "List[str], or a List[Dict[str, str]] for a Text2Text Pipeline.",
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

    def _convert_audio_input(self, data):
        """
        Conversion utility for decoding the base64 encoded bytes data of a raw soundfile when
        parsed through model serving, if applicable. Direct usage of the pyfunc implementation
        outside of model serving will treat this utility as a noop.

        For reference, the expected encoding for input to Model Serving will be:

        import requests
        import base64

        response = requests.get("https://www.my.sound/a/sound/file.wav")
        encoded_audio = base64.b64encode(response.content).decode("ascii")

        inference_data = json.dumps({"inputs": [encoded_audio]})

        or

        inference_df = pd.DataFrame(
        pd.Series([encoded_audio], name="audio_file")
        )
        split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
        split_json = json.dumps(split_dict)

        or

        records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
        records_json = json.dumps(records_dict)

        This utility will convert this JSON encoded, base64 encoded text back into bytes for
        input into the AutomaticSpeechRecognitionPipeline for inference.
        """

        def is_base64(s):
            try:
                return base64.b64encode(base64.b64decode(s)) == s
            except binascii.Error:
                return False

        def decode_audio(encoded):
            if isinstance(encoded, str):
                # This is to support blob style passing of uri locations to process audio files
                # on disk or object store. Note that if a uri is passed, a signature *must be*
                # provided for serving to function as the default signature uses bytes.
                return encoded
            elif isinstance(encoded, bytes):
                # For input types 'dataframe_split' and 'dataframe_records', the encoding
                # conversion to bytes is handled.
                if not is_base64(encoded):
                    return encoded
                else:
                    # For input type 'inputs', explicit decoding of the b64encoded audio is needed.
                    return base64.b64decode(encoded)
            else:
                try:
                    return base64.b64decode(encoded)
                except binascii.Error as e:
                    raise MlflowException(
                        "The encoded soundfile that was passed has not been properly base64 "
                        "encoded. Please ensure that the raw sound bytes have been processed with "
                        "`base64.b64encode(<audio data bytes>).decode('ascii')`"
                    ) from e

        # The example input data that is processed by this logic is from the pd.DataFrame
        # conversion that happens within serving wherein the bytes input data is cast to
        # a pd.DataFrame(pd.Series([raw_bytes])) and then cast to JSON serializable data in the
        # format:
        # {[0]: [{[0]: <audio data>}]}
        # In the inputs format, due to the modification of how types are not enforced, the
        # logic that is present in processing `records` and `split` format orientation when casting
        # back to dictionary does not do the automatic decoding of the data from base64 encoded
        # back to bytes. This is the reason for the conditional logic within `decode_audio` based
        # on whether the bytes data is base64 encoded or standard bytes format.
        # The output of the conversion present in the conditional structural validation below is
        # to return the only input format that the audio transcription pipeline permits:
        # a bytes input of a single element.
        if isinstance(data, list) and all(isinstance(element, dict) for element in data):
            encoded_audio = list(data[0].values())[0]
            if isinstance(encoded_audio, str):
                self._validate_str_input_uri_or_file(encoded_audio)
            return decode_audio(encoded_audio)
        elif isinstance(data, str):
            self._validate_str_input_uri_or_file(data)
        return data

    @staticmethod
    def _validate_str_input_uri_or_file(input_str):
        """
        Validation of blob references to audio files, if a string is input to the ``predict``
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
            raise MlflowException(
                "An invalid string input was provided. String inputs to "
                "audio files must be either a file location or a uri.",
                error_code=BAD_REQUEST,
            )


@experimental
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
):  # pylint: disable=unused-argument
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
        with mlflow.utils.autologging_utils.disable_discrete_autologging(
            DISABLED_ANCILLARY_FLAVOR_AUTOLOGGING
        ):
            return original(*args, **kwargs)

    with contextlib.suppress(ImportError):
        import setfit

        safe_patch(
            FLAVOR_NAME, setfit.SetFitTrainer, "train", functools.partial(train), manage_run=False
        )

    with contextlib.suppress(ImportError):
        import transformers

        classes = [transformers.Trainer, transformers.Seq2SeqTrainer]
        methods = ["train"]
        for clazz in classes:
            for method in methods:
                safe_patch(FLAVOR_NAME, clazz, method, functools.partial(train), manage_run=False)
