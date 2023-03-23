import logging
import pathlib

from typing import Union, List, Optional, Dict, Any
import yaml
import warnings

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import ModelInputExample, Model, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
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
    _get_flavor_configuration_from_uri,
    _add_code_from_conf_to_system_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "transformers"
_PIPELINE_BINARY_KEY = "pipeline"
_PIPELINE_BINARY_FILE_NAME = "pipeline.tr"
_COMPONENTS_BINARY_KEY = "components"
_TOKENIZER_KEY = "tokenizer"
_FEATURE_EXTRACTOR_KEY = "feature_extractor"
_IMAGE_PROCESSOR_KEY = "image_processor"
_PROCESSOR_KEY = "processor"
_TOKENIZER_TYPE_KEY = "tokenizer_type"
_FEATURE_EXTRACTOR_TYPE_KEY = "feature_extractor_type"
_IMAGE_PROCESSOR_TYPE_KEY = "image_processor_type"
_PROCESSOR_TYPE_KEY = "processor_type"
_CARD_TEXT_FILE_NAME = "model_card_text.txt"
_CARD_DATA_FILE_NAME = "model_card_data.yaml"
_TASK_KEY = "task"
_LOGGED_TYPE_KEY = "logged_type"
_INSTANCE_TYPE_KEY = "instance_type"
_PIPELINE_MODEL_TYPE_KEY = "pipeline_model_type"
_MODEL_PATH_OR_NAME_KEY = "source_model_name"

_logger = logging.getLogger(__name__)


def _model_engine_type(model) -> List[str]:
    """
    Determines which pip libraries should be included based on the base model engine
    type.
    In the ``transformers`` library, all TensorFlow versions of specific model implementations have
    their class names begin with "TF" (for instance, "TFMobileBertForSequenceClassification" is the
    TensorFlow version of the MobileBertForSequenceClassification model) while the
    PyTorch version of the same architecture omits the "TF" ("MobileBertForSequenceClassification").

    :param model: The model instance to be saved in order to provide the required underlying
    deep learning execution framework dependency requirements.
    :return: A string representing the engine type of the model being used (one of 'tf', 'flax', or
        'torch').
    """
    engine = _get_engine_type(model)
    if engine == "torch":
        return ["torch", "torchvision"]
    else:
        return [engine]


def get_default_pip_requirements(model) -> List[str]:
    """
    :param model: The model instance to be saved in order to provide the required underlying
    deep learning execution framework dependency requirements. Note that this must be the actual
    model instance and not a Pipeline.
    :return: A list of default pip requirements for MLflow Models that have been produced with the
    ``transformers`` flavor. Calls to :py:func:`save_model()` and :py:func:`log_model()` produce
    a pip environment that contain these requirements at a minimum.
    """
    from transformers import TFPreTrainedModel, FlaxPreTrainedModel, PreTrainedModel

    if not isinstance(model, (TFPreTrainedModel, FlaxPreTrainedModel, PreTrainedModel)):
        raise MlflowException(
            "The supplied model type is unsupported. The model must be one of: PreTrainedModel, "
            "TFPreTrainedModel, or FlaxPreTrainedModel"
        )
    base_reqs = ["transformers"]
    base_reqs.extend(_model_engine_type(model))
    return [_get_pinned_requirement(module) for module in base_reqs]


def get_default_conda_env(
    model,
) -> Union[None, Dict[str, Union[str, List[Union[str, Dict[str, List[str]]]]]]]:
    """
    :return: The default Conda environment for MLflow Models produced with the ``transformers``
    flavor, based on the model instance framework type of the model to be logged.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(model))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    transformers_model,
    path: str,
    processor=None,
    task: Optional[str] = None,
    model_card=None,
    conda_env: Optional[
        Union[str, Dict[str, Union[str, List[Union[str, Dict[str, List[str]]]]]]]
    ] = None,
    code_paths: Optional[List[str]] = None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    metadata: Dict[str, Any] = None,
    **kwargs,
) -> None:
    """
    Save a trained transformers model to a path on the local file system.

    :param transformers_model: A trained transformers `Pipeline`, `PreTrainedModel`,
                               `TFPreTrainedModel`, or `FlaxPreTrainedModel` that can be loaded
                               using the `from_pretrained` method of the respective class.
    :param path: Local path destination for the serialized model to be saved.
    :param processor: An optional ``Processor`` subclass object. Some model architectures,
                      particularly multi-modal types, utilize Processors to combine text
                      encoding and image or audio encoding in a single entrypoint.

                      .. Note:: If a processor is supplied with the saving a model, the
                                model will be unavailable for loading as a ``Pipeline`` or used
                                for pyfunc inference.

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
    :param conda_env: A dictionary specifying a conda environment to be created for running the
                      model. The environment can be specified as a YAML file, a string in YAML or
                      JSON format, or a dictionary.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: An MLflow model object that specifies the flavor that this model is being
                         added to.
    :param signature: A Model Signature object that describes the input and output Schema of the
                      model. The model signature can be inferred using `infer_signature` function
                      of `mlflow.models.signature`.
    :param input_example: An example of valid input that the model can accept. The example can be
                          used as a hint of what data to feed the model. The given example will be
                          converted to a `Pandas DataFrame` and then serialized to JSON using the
                          `Pandas` split-oriented format. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.

    :param kwargs: Optional additional configurations for transformers serialization.
    :return: None
    """
    import transformers

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = pathlib.Path(path).absolute()

    _validate_and_prepare_target_save_path(str(path))

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, str(path))

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, str(path))
    if metadata is not None:
        mlflow_model.metadata = metadata

    resolved_task = _get_or_infer_task_type(transformers_model, task)

    # build the pipeline object if a model is passed
    if not isinstance(transformers_model, transformers.Pipeline):
        built_pipeline = _build_pipeline_from_model_input(transformers_model, resolved_task)
    else:
        built_pipeline = transformers_model

    flavor_conf = _generate_base_flavor_configuration(built_pipeline, resolved_task)

    components = _record_pipeline_components(built_pipeline)

    if components:
        flavor_conf.update(**components)

    if processor:
        warnings.warn(
            "The model being saved contains a Processor component. Processors are not "
            "compatible or available to be used if loading as pyfunc or used in a "
            "spark_udf."
        )
        flavor_conf.update({_PROCESSOR_TYPE_KEY: _get_instance_type(processor, False)})

    # Save the pipeline object
    built_pipeline.save_pretrained(save_directory=str(path.joinpath(_PIPELINE_BINARY_FILE_NAME)))

    # Save the components explicitly to the components directory
    _save_components(
        root_path=path.joinpath(_COMPONENTS_BINARY_KEY),
        component_config=components,
        pipeline=built_pipeline,
        processor=processor,
    )

    # Get the model card from either the argument or the HuggingFace marketplace
    card_data = model_card if model_card is not None else _fetch_model_card(transformers_model)

    # If the card data can be acquired, save the text and the data separately
    if card_data:
        write_to(str(path.joinpath(_CARD_TEXT_FILE_NAME)), card_data.text)
        with path.joinpath(_CARD_DATA_FILE_NAME).open("w") as file:
            yaml.safe_dump(card_data.data.to_dict(), stream=file, default_flow_style=False)

    model_bin_kwargs = {_PIPELINE_BINARY_KEY: _PIPELINE_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.transformers",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
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
            if isinstance(transformers_model, transformers.Pipeline):
                default_reqs = get_default_pip_requirements(transformers_model.model)
            else:
                default_reqs = get_default_pip_requirements(transformers_model)
            inferred_reqs = infer_pip_requirements(str(path), FLAVOR_NAME, fallback=default_reqs)
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
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


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    transformers_model,
    artifact_path: str,
    processor=None,
    task: Optional[str] = None,
    model_card=None,
    conda_env: Optional[
        Union[str, Dict[str, Union[str, List[Union[str, Dict[str, List[str]]]]]]]
    ] = None,
    code_paths: Optional[List[str]] = None,
    registered_model_name: str = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    metadata: Dict[str, Any] = None,
    **kwargs,
):
    """
    Log a ``transformers`` object as an MLflow artifact for the current run.

    :param transformers_model: A trained ``transformers`` ``Pipeline``, ``PreTrainedModel``,
                               ``TFPreTrainedModel``, or ``FlaxPreTrainedModel`` that can be loaded
                               using the `from_pretrained` method of the respective class.
    :param path: Local path destination for the serialized model to be saved.
    :param processor: An optional ``Processor`` subclass object. Some model architectures,
                  particularly multi-modal types, utilize Processors to combine text
                  encoding and image or audio encoding in a single entrypoint.

                  .. Note:: If a processor is supplied with the saving a model, the
                            model will be unavailable for loading as a ``Pipeline`` or used
                            for pyfunc inference.

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
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param signature: :py:class:`Model Signature <mlflow.models.ModelSignature>` describes model
                      input and output :py:class:`Schema <mlflow.types.Schema>`. The model
                      signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        # TODO: fill out example once pyfunc support for dtypes is done

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
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def load_model(model_uri: str, dst_path: str = None, return_type="pipeline", **kwargs):
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
                        {
                         "model": BertForSequenceClassification,
                         "tokenizer": BertTokenizerFast
                         }

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
                        saving.
                        Default: "pipeline".
    :param kwargs: Optional configuration options for loading of a ``transformers`` object.
                   For information on parameters and their usage, see
                   `transformers documentation <https://huggingface.co/docs/transformers/index>`_.

    :return: A ``transformers`` model instance or a dictionary of components
    """

    import transformers

    model_uri = str(model_uri)

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)

    flavor_config = _get_flavor_configuration_from_uri(model_uri, FLAVOR_NAME, _logger)

    if return_type == "pipeline" and flavor_config.get(_PROCESSOR_TYPE_KEY, None):
        raise MlflowException(
            "This model has been saved with a processor. Processor objects are "
            "not compatible with Pipelines. Please load this model by specifying "
            "the 'return_type'='components'."
        )

    _add_code_from_conf_to_system_path(local_model_path, flavor_config)

    components = _load_model(local_model_path, flavor_config)
    if return_type == "pipeline":
        components.update(**kwargs)
        return transformers.pipeline(**components)
    elif return_type == "components":
        return components
    else:
        raise MlflowException(
            f"The specified return_type mode '{return_type}' is unsupported. "
            "Please select one of: 'pipeline' or 'components'."
        )


def _load_model(path: str, flavor_config):
    """
    Loads components from a locally serialized ``Pipeline`` object.
    """
    import transformers

    local_path = pathlib.Path(path)
    pipeline_path = local_path.joinpath(
        flavor_config.get(_PIPELINE_BINARY_KEY, _PIPELINE_BINARY_FILE_NAME)
    )

    model_instance = getattr(transformers, flavor_config[_PIPELINE_MODEL_TYPE_KEY])
    conf = {
        "task": flavor_config[_TASK_KEY],
        "model": model_instance.from_pretrained(pipeline_path),
    }

    if flavor_config.get(_PROCESSOR_TYPE_KEY, None):
        conf[_PROCESSOR_KEY] = _load_component(
            local_path, _PROCESSOR_KEY, flavor_config[_PROCESSOR_TYPE_KEY]
        )

    for component_key in flavor_config[_COMPONENTS_BINARY_KEY]:
        component_type_key = f"{component_key}_type"
        component_type = flavor_config[component_type_key]
        conf[component_key] = _load_component(local_path, component_key, component_type)
    return conf


def _fetch_model_card(model_or_pipeline):
    """
    Attempts to retrieve the model card for the specified model architecture iff the
    `huggingface_hub` library is installed. If a card cannot be found in the registry or
    the library is not installed, returns None.
    """
    from transformers import Pipeline

    # Attempt to import the huggingface-hub library in order to fetch the model card.
    try:
        from huggingface_hub import ModelCard
    except ImportError:
        return None

    model = (
        model_or_pipeline.model if isinstance(model_or_pipeline, Pipeline) else model_or_pipeline
    )

    return ModelCard.load(model.name_or_path)


def _build_pipeline_from_model_input(model, task: str):
    """
    Utility for generating a pipeline from component parts.
    """
    from transformers import PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel

    if not isinstance(model, (PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel)):
        raise MlflowException(
            f"The provided model is not the correct type. The type provided is: {type(model)}. "
            f"The model must inherit from one of: PreTrainedModel, TFPreTrainedModel, "
            f"FlaxPreTrainedModel"
        )

    from transformers import AutoTokenizer, pipeline

    model_architecture_name = model.name_or_path

    pipeline_config = {"task": task, "model": model}

    try:
        pipeline_config["tokenizer"] = AutoTokenizer.from_pretrained(model_architecture_name)
    except (KeyError, OSError):
        # A KeyError (or OSError if reading from cache) indicates that the model does not
        # support a tokenizer and instead requires either a ``FeatureExtractor`` or an
        # ``ImageProcessor``
        pass
    pipeline_config.update(**_configure_extractors(model_architecture_name))

    return pipeline(**pipeline_config)


def _configure_extractors(architecture):
    """
    Performs a hierarchy-based acquisition of an appropriate processor or
    feature extractor based on a named and registered model type within the HuggingFace repo.
    In order to support as many use cases as possible, all available processors and extractors
    are loaded.
    """
    import transformers

    extractor_types = {
        "image_processor": "AutoImageProcessor",
        "feature_extractor": "AutoFeatureExtractor",
    }
    extractors = {}
    for extractor_type, loader in extractor_types.items():
        try:
            instance = getattr(transformers, loader)
            extractors[extractor_type] = instance.from_pretrained(architecture)
        except (KeyError, OSError):
            pass
    return extractors


def _record_pipeline_components(pipeline) -> Optional[Dict[str, Any]]:
    """
    Utility for recording which components are present in either the generated pipeline iff the
    supplied save object is not a pipeline or the components of the supplied pipeline object.
    """
    components_conf = {}
    components = []
    if pipeline.feature_extractor:
        feature_extractor = pipeline.feature_extractor
        components_conf.update(
            {_FEATURE_EXTRACTOR_TYPE_KEY: _get_instance_type(feature_extractor, False)}
        )
        components.append(_FEATURE_EXTRACTOR_KEY)
    if pipeline.tokenizer:
        tokenizer = pipeline.tokenizer
        components_conf.update({_TOKENIZER_TYPE_KEY: _get_instance_type(tokenizer, False)})
        components.append(_TOKENIZER_KEY)
    if pipeline.image_processor:
        image_processor = pipeline.image_processor
        components_conf.update(
            {_IMAGE_PROCESSOR_TYPE_KEY: _get_instance_type(image_processor, False)}
        )
        components.append(_IMAGE_PROCESSOR_KEY)
    if components:
        components_conf.update({_COMPONENTS_BINARY_KEY: components})
    return components_conf


def _save_components(
    root_path: pathlib.Path, component_config: Dict[str, Any], pipeline, processor
):
    """
    Saves non-model pipeline components explicitly to a separate directory path for compatibility
    with certain older model types. In earlier versions of ``transformers``, inferred feature
    extractor types could be resolved to their correct processor types. This approach ensures
    compatibility with the expected pipeline configuration argument assignments in later versions
    of the ``Pipeline`` class.
    """
    component_types = component_config["components"]
    for component_name in component_types:
        component = getattr(pipeline, component_name)
        save_path = str(root_path.joinpath(component_name))
        component.save_pretrained(save_path)
    if processor:
        processor_path = str(root_path.joinpath(_PROCESSOR_KEY))
        processor.save_pretrained(processor_path)


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
    model,
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
    from transformers import Pipeline

    _validate_transformers_task_type(task)

    flavor_configuration = {
        _TASK_KEY: task,
        _LOGGED_TYPE_KEY: _get_instance_type(model, True),
        _INSTANCE_TYPE_KEY: _get_instance_type(model, False),
        _MODEL_PATH_OR_NAME_KEY: _get_base_model_architecture(model),
    }

    # If the object to be saved is a Pipeline, record the model type within the pipeline
    if isinstance(model, Pipeline):
        flavor_configuration.update(
            {_PIPELINE_MODEL_TYPE_KEY: _get_instance_type(model.model, False)}
        )
    return flavor_configuration


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
    from transformers import PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel, Pipeline
    from transformers.pipelines import get_task

    if isinstance(model, Pipeline):
        return model.task
    elif isinstance(model, (PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel)):
        # transformers provides guard conditions for potentially invalid entries
        return get_task(model.name_or_path)
    else:
        raise MlflowException(
            f"The provided model type: {type(model)} is not supported. "
            "Supported model types are: PreTrainedModel or Pipeline."
        )


def _validate_transformers_task_type(task: str) -> None:
    """
    Validates that a given ``task`` type is supported by the ``transformers`` library.
    """
    from transformers.pipelines import get_supported_tasks

    valid_tasks = get_supported_tasks()

    if task not in valid_tasks:
        raise MlflowException(
            f"The task provided is invalid. '{task}' is not a supported task. "
            f"Must be one of: {valid_tasks}"
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
        return model_or_pipeline.name_or_path


def _get_instance_type(model, base: bool):
    """
    Utility for extracting the saved object type or, if the `base` argument is set to `True`,
    the base ABC type of the model.
    """
    if base:
        from transformers import PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel, Pipeline

        def _get_model_base_class(model):
            pipeline_abc = Pipeline
            tf_abc = TFPreTrainedModel
            torch_abc = PreTrainedModel
            flax_abc = FlaxPreTrainedModel
            for cls in model.__class__.__mro__:
                if issubclass(cls, tf_abc):
                    return tf_abc.__name__
                elif issubclass(cls, torch_abc):
                    return torch_abc.__name__
                elif issubclass(cls, flax_abc):
                    return flax_abc.__name__
                elif issubclass(cls, pipeline_abc):
                    return pipeline_abc.__name__

        return _get_model_base_class(model)
    else:
        return model.__class__.__name__
