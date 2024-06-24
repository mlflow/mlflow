import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.transformers.llm_inference_utils import (
    _LLM_INFERENCE_TASK_EMBEDDING,
    _LLM_V1_EMBEDDING_INPUT_KEY,
    postprocess_output_for_llm_v1_embedding_task,
)
from mlflow.types.llm import (
    EMBEDDING_MODEL_INPUT_SCHEMA,
    EMBEDDING_MODEL_OUTPUT_SCHEMA,
)
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
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
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _download_artifact_from_uri,
    _get_flavor_configuration_from_uri,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "sentence_transformers"
_TRANSFORMER_SOURCE_MODEL_NAME_KEY = "source_model_name"
_TRANSFORMER_MODEL_TYPE_KEY = "pipeline_model_type"

SENTENCE_TRANSFORMERS_DATA_PATH = "model.sentence_transformer"
_INFERENCE_CONFIG_PATH = "inference_config"

# Patterns to extract HuggingFace model repository name from the local snapshot path.
# The path format would be like /path/to/{username}_{modelname}, where user name can
# only contain number, letters, and dashes.
_LOCAL_SNAPSHOT_PATH_PATTERN = re.compile(r"/([0-9a-zA-Z-]+)_([^\/]+)/$")

_logger = logging.getLogger(__name__)


@experimental
def get_default_pip_requirements() -> List[str]:
    """
    Retrieves the set of minimal dependencies for the ``sentence_transformers`` flavor.

    Returns:
        A list of default pip requirements for MLflow Models that have been produced with the
        ``sentence-transformers`` flavor. Calls to :py:func:`save_model()` and
        :py:func:`log_model()` produce a pip environment that contain these
        requirements at a minimum.
    """
    base_reqs = ["sentence-transformers", "transformers", "torch"]
    return [_get_pinned_requirement(module) for module in base_reqs]


@experimental
def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced with the
        ``sentence_transformers`` flavor.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@experimental
def _verify_task_and_update_metadata(
    task: str, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if task not in [_LLM_INFERENCE_TASK_EMBEDDING]:
        raise MlflowException.invalid_parameter_value(
            f"Received invalid parameter value for `task` argument {task}. Task type could "
            f"only be {_LLM_INFERENCE_TASK_EMBEDDING}"
        )
    if metadata is None:
        metadata = {}
    if "task" in metadata and metadata["task"] != task:
        raise MlflowException.invalid_parameter_value(
            f"Received invalid parameter value for `task` argument {task}. Task type is "
            f"inconsistent with the task value from metadata {metadata['task']}"
        )
    metadata["task"] = task
    return metadata


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    path: str,
    task: Optional[str] = None,
    inference_config: Optional[Dict[str, Any]] = None,
    code_paths: Optional[List[str]] = None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    conda_env=None,
    metadata: Optional[Dict[str, Any]] = None,
    example_no_conversion: bool = False,
) -> None:
    """
    .. note::

        Saving Sentence Transformers models with custom code (i.e. models that require
        ``trust_remote_code=True``) is supported in MLflow 2.12.0 and above.


    Save a trained ``sentence-transformers`` model to a path on the local file system.

    Args:
        model: A trained ``sentence-transformers`` model.
        path: Local path destination for the serialized model to be saved.
        task: MLflow inference task type for ``sentence-transformers`` model. Candidate task type
            is `llm/v1/embeddings`.
        inference_config:
            A dict of valid inference parameters that can be applied to a ``sentence-transformer``
            model instance during inference.
            These arguments are used exclusively for the case of loading the model as a ``pyfunc``
            Model or for use in Spark.
            These values are not applied to a returned model from a call to
            ``mlflow.sentence_transformers.load_model()``
        code_paths: {{ code_paths }}
        mlflow_model: An MLflow model object that specifies the flavor that this model is being
            added to.
        signature: an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            class that describes the model's inputs and outputs. If not specified but an
            ``input_example`` is supplied, a signature will be automatically inferred
            based on the supplied input example and model. If both ``signature`` and
            ``input_example`` are not specified or the automatic signature inference
            fails, a default signature will be adopted. To prevent a signature from being
            adopted, set ``signature`` to ``False``. To manually infer a model signature,
            call :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
            with valid model inputs and valid model outputs.
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        conda_env: {{ conda_env }}
        metadata: {{ metadata }}
        example_no_conversion: {{ example_no_conversion }}
    """
    import sentence_transformers

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = pathlib.Path(path).absolute()
    model_data_path = path.joinpath(SENTENCE_TRANSFORMERS_DATA_PATH)

    _validate_and_prepare_target_save_path(str(path))

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, str(path))

    if task is not None:
        signature = ModelSignature(
            inputs=EMBEDDING_MODEL_INPUT_SCHEMA, outputs=EMBEDDING_MODEL_OUTPUT_SCHEMA
        )
    elif signature is None and input_example is not None:
        wrapped_model = _SentenceTransformerModelWrapper(model)
        signature = _infer_signature_from_input_example(
            input_example, wrapped_model, no_conversion=example_no_conversion
        )
    elif signature is None:
        signature = _get_default_signature()
    elif signature is False:
        signature = None

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, str(path), no_conversion=example_no_conversion)
    if metadata is not None:
        mlflow_model.metadata = metadata
    model_config = None
    if task is not None:
        mlflow_model.metadata = _verify_task_and_update_metadata(task, mlflow_model.metadata)
        model_config = {"task": _LLM_INFERENCE_TASK_EMBEDDING}

    model.save(str(model_data_path))

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.sentence_transformers",
        data=SENTENCE_TRANSFORMERS_DATA_PATH,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        model_config=model_config,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        sentence_transformers_version=sentence_transformers.__version__,
        code=code_dir_subpath,
        **_get_transformers_model_metadata(model),
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(str(path.joinpath(MLMODEL_FILE_NAME)))

    if inference_config:
        path.joinpath(_INFERENCE_CONFIG_PATH).write_text(json.dumps(inference_config))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
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


def _get_transformers_model_metadata(model) -> Dict[str, str]:
    """
    Extract metadata about the underlying Transformers model, such as the model class name
    and the repository id.

    Args:
        model: A SentenceTransformer model instance.

    Returns:
        A dictionary containing metadata about the Transformers model.
    """
    from sentence_transformers.models import Transformer

    # NB: We assume the SentenceTransformer model contains only up to one Transformer model.
    for module in model.modules():
        if isinstance(module, Transformer):
            model_instance = module.auto_model
            return {
                _TRANSFORMER_SOURCE_MODEL_NAME_KEY: _get_transformers_model_name(
                    model_instance.name_or_path
                ),
                _TRANSFORMER_MODEL_TYPE_KEY: model_instance.__class__.__name__,
            }
    return {}


def _get_transformers_model_name(model_name_or_path):
    """
    Extract the Transformers model name from name_or_path attribute of a Transformer model.

    Normally the name_or_path attribute just points to the model name, but in Sentence
    Transformers < 2.3.0, the library loads the Transformers model after local snapshot
    download, so the name_or_path attribute points to the local filepath.
    https://github.com/UKPLab/sentence-transformers/commit/9db0f205adcf315d16961fea7e9e6906cb950d43
    """
    if m := _LOCAL_SNAPSHOT_PATH_PATTERN.search(model_name_or_path):
        return f"{m.group(1)}/{m.group(2)}"
    return model_name_or_path


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path: str,
    task: Optional[str] = None,
    inference_config: Optional[Dict[str, Any]] = None,
    code_paths: Optional[List[str]] = None,
    registered_model_name: Optional[str] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    conda_env=None,
    metadata: Optional[Dict[str, Any]] = None,
    example_no_conversion: bool = False,
):
    """
    .. note::

        Logging Sentence Transformers models with custom code (i.e. models that require
        ``trust_remote_code=True``) is supported in MLflow 2.12.0 and above.

    Log a ``sentence_transformers`` model as an MLflow artifact for the current run.

    .. code-block:: python

        # An example of using log_model for a sentence-transformers model and architecture:

        from sentence_transformers import SentenceTransformer
        import mlflow

        model = SentenceTransformer("all-MiniLM-L6-v2")
        data = "MLflow is awesome!"
        signature = mlflow.models.infer_signature(
            model_input=data,
            model_output=model.encode(data),
        )

        with mlflow.start_run():
            mlflow.sentence_transformers.log_model(
                model=model,
                artifact_path="sbert_model",
                signature=signature,
                input_example=data,
            )



    Args:
        model: A trained ``sentence-transformers`` model.
        artifact_path: Local path destination for the serialized model to be saved.
        task: MLflow inference task type for ``sentence-transformers`` model. Candidate task type
            is `llm/v1/embeddings`.
        inference_config:
            A dict of valid overrides that can be applied to a ``sentence-transformer`` model
            instance during inference.
            These arguments are used exclusively for the case of loading the model as a ``pyfunc``
            Model or for use in Spark.
            These values are not applied to a returned model from a call to
            ``mlflow.sentence_transformers.load_model()``
        code_paths: {{ code_paths }}
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            class that describes the model's inputs and outputs. If not specified but an
            ``input_example`` is supplied, a signature will be automatically inferred
            based on the supplied input example and model. If both ``signature`` and
            ``input_example`` are not specified or the automatic signature inference
            fails, a default signature will be adopted. To prevent a signature from being
            adopted, set ``signature`` to ``False``. To manually infer a model signature,
            call :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
            with valid model inputs and valid model outputs.
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        conda_env: {{ conda_env }}
        metadata: {{ metadata }}
        example_no_conversion: {{ example_no_conversion }}
    """
    if task is not None:
        metadata = _verify_task_and_update_metadata(task, metadata)

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.sentence_transformers,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        metadata=metadata,
        model=model,
        inference_config=inference_config,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        example_no_conversion=example_no_conversion,
    )


def _get_load_kwargs():
    import sentence_transformers

    load_kwargs = {}
    # The trust_remote_code is supported since Sentence Transformers 2.3.0
    if Version(sentence_transformers.__version__) >= Version("2.3.0"):
        # Always set trust_remote_code=True because we save the entire repository files in
        # the model artifacts, so there is no risk of running untrusted code unless the logged
        # artifact is modified by a malicious actor, which is much more broader security
        # concern that even cannot be prevented by setting trust_remote_code=False.
        load_kwargs["trust_remote_code"] = True
    return load_kwargs


def _load_pyfunc(path, model_config: Optional[Dict[str, Any]] = None):
    """
    Load PyFunc implementation for SentenceTransformer. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``sentence_transformer`` flavor.
    """
    import sentence_transformers

    load_kwargs = _get_load_kwargs()
    model = sentence_transformers.SentenceTransformer(path, **load_kwargs)
    model_config = model_config or {}
    task = model_config.get("task", None)
    return _SentenceTransformerModelWrapper(model, task)


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
def load_model(model_uri: str, dst_path: Optional[str] = None):
    """
    Load a ``sentence_transformers`` object from a local file or a run.

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

    Returns:
        A ``sentence_transformers`` model instance
    """

    import sentence_transformers

    model_uri = str(model_uri)

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)

    local_model_dir = pathlib.Path(local_model_path).joinpath(SENTENCE_TRANSFORMERS_DATA_PATH)

    flavor_config = _get_flavor_configuration_from_uri(model_uri, FLAVOR_NAME, _logger)

    _add_code_from_conf_to_system_path(local_model_path, flavor_config)

    load_kwargs = _get_load_kwargs()
    return sentence_transformers.SentenceTransformer(str(local_model_dir), **load_kwargs)


def _get_default_signature():
    """
    Generates a default signature for the ``sentence_transformers`` flavor to be applied if not
    set or overridden by supplying the `signature` argument to `log_model` or `save_model`.
    """
    return ModelSignature(
        inputs=Schema([ColSpec("string")]),
        outputs=Schema([TensorSpec(np.dtype("float64"), [-1])]),
    )


class _SentenceTransformerModelWrapper:
    def __init__(self, model, task=None):
        self.model = model
        self.task = task

    def predict(self, sentences, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            sentences: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        # When the input is a single string or a dictionary, it is transformed into a DataFrame
        # with one column and row, but the encode function does not accept DataFrame input
        convert_output_to_llm_v1_format = False
        if type(sentences) == pd.DataFrame:
            # Wrap the output to OpenAI format only when the input is dict `{"input": ... }`
            if self.task and list(sentences.columns)[0] == _LLM_V1_EMBEDDING_INPUT_KEY:
                convert_output_to_llm_v1_format = True
            sentences = sentences.iloc[:, 0]
            if type(sentences[0]) == list:
                sentences = sentences[0]

        # The encode API has additional parameters that we can add as kwargs.
        # See https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
        if params:
            try:
                output_data = self.model.encode(sentences, **params)
            except TypeError as e:
                raise MlflowException.invalid_parameter_value(
                    "Received invalid parameter value for `params` argument"
                ) from e
        else:
            output_data = self.model.encode(sentences)

        if convert_output_to_llm_v1_format:
            output_data = postprocess_output_for_llm_v1_embedding_task(
                sentences, output_data, self.model.tokenizer
            )
        return output_data
