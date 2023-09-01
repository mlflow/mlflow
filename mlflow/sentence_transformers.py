import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
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
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _download_artifact_from_uri,
    _get_flavor_configuration_from_uri,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "sentence_transformers"
SENTENCE_TRANSFORMERS_DATA_PATH = "model.sentence_transformer"
_INFERENCE_CONFIG_PATH = "inference_config"

_logger = logging.getLogger(__name__)


@experimental
def get_default_pip_requirements() -> List[str]:
    """
    Retrieves the set of minimal dependencies for the ``sentence_transformers`` flavor.

    :return: A list of default pip requirements for MLflow Models that have been produced with the
             ``sentence-transformers`` flavor. Calls to :py:func:`save_model()` and
             :py:func:`log_model()` produce a pip environment that contain these
             requirements at a minimum.
    """
    base_reqs = ["sentence-transformers", "transformers", "torch"]
    return [_get_pinned_requirement(module) for module in base_reqs]


@experimental
def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced with the
             ``sentence_transformers`` flavor.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    path: str,
    inference_config: Optional[Dict[str, Any]] = None,
    code_paths: Optional[List[str]] = None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    conda_env=None,
    metadata: Dict[str, Any] = None,
) -> None:
    """
    Save a trained ``sentence-transformers`` model to a path on the local file system.

    :param model: A trained ``sentence-transformers`` model.
    :param path: Local path destination for the serialized model to be saved.
    :param inference_config:
        A dict of valid inference parameters that can be applied to a ``sentence-transformer``
        model instance during inference.
        These arguments are used exclusively for the case of loading the model as a ``pyfunc``
        Model or for use in Spark.
        These values are not applied to a returned model from a call to
        ``mlflow.sentence_transformers.load_model()``
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: An MLflow model object that specifies the flavor that this model is being
                         added to.
    :param signature: an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      class that describes the model's inputs and outputs. If not specified but an
                      ``input_example`` is supplied, a signature will be automatically inferred
                      based on the supplied input example and model. If both ``signature`` and
                      ``input_example`` are not specified or the automatic signature inference
                      fails, a default signature will be adopted. To prevent a signature from being
                      adopted, set ``signature`` to ``False``. To manually infer a model signature,
                      call :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
                      with valid model inputs and valid model outputs.
    :param input_example: {{ input_example }}
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param conda_env: {{ conda_env }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.

    :return: None
    """
    import sentence_transformers

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = pathlib.Path(path).absolute()
    model_data_path = path.joinpath(SENTENCE_TRANSFORMERS_DATA_PATH)

    _validate_and_prepare_target_save_path(str(path))

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, str(path))

    if signature is None and input_example is not None:
        wrapped_model = _SentenceTransformerModelWrapper(model)
        signature = _infer_signature_from_input_example(input_example, wrapped_model)
    elif signature is None:
        signature = _get_default_signature()
    elif signature is False:
        signature = None

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, str(path))
    if metadata is not None:
        mlflow_model.metadata = metadata

    model.save(str(model_data_path))

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.sentence_transformers",
        data=SENTENCE_TRANSFORMERS_DATA_PATH,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        sentence_transformers_version=sentence_transformers.__version__,
        code=code_dir_subpath,
    )
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


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path: str,
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
):
    """
    Log a ``sentence_transformers`` model as an MLflow artifact for the current run.

    :param model: A trained ``sentence-transformers`` model.
    :param artifact_path: Local path destination for the serialized model to be saved.
    :param inference_config:
        A dict of valid overrides that can be applied to a ``sentence-transformer`` model instance
        during inference.
        These arguments are used exclusively for the case of loading the model as a ``pyfunc``
        Model or for use in Spark.
        These values are not applied to a returned model from a call to
        ``mlflow.sentence_transformers.load_model()``
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param signature: an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      class that describes the model's inputs and outputs. If not specified but an
                      ``input_example`` is supplied, a signature will be automatically inferred
                      based on the supplied input example and model. If both ``signature`` and
                      ``input_example`` are not specified or the automatic signature inference
                      fails, a default signature will be adopted. To prevent a signature from being
                      adopted, set ``signature`` to ``False``. To manually infer a model signature,
                      call :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
                      with valid model inputs and valid model outputs.
    :param input_example: {{ input_example }}
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param conda_env: {{ conda_env }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.

    """
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
    )


def _load_pyfunc(path):
    """
    Load PyFunc implementation for SentenceTransformer. Called by ``pyfunc.load_model``.
    :param path: Local filesystem path to the MLflow Model with the ``sentence_transformer`` flavor.
    """
    import sentence_transformers

    model = sentence_transformers.SentenceTransformer.load(path)
    return _SentenceTransformerModelWrapper(model)


@experimental
@docstring_version_compatibility_warning(integration_name=FLAVOR_NAME)
def load_model(model_uri: str, dst_path: str = None):
    """
    Load a ``sentence_transformers`` object from a local file or a run.

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
    :return: A ``sentence_transformers`` model instance
    """

    import sentence_transformers

    model_uri = str(model_uri)

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)

    local_model_dir = pathlib.Path(local_model_path).joinpath(SENTENCE_TRANSFORMERS_DATA_PATH)

    flavor_config = _get_flavor_configuration_from_uri(model_uri, FLAVOR_NAME, _logger)

    _add_code_from_conf_to_system_path(local_model_path, flavor_config)

    return sentence_transformers.SentenceTransformer.load(local_model_dir)


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
    def __init__(self, model):
        self.model = model

    def predict(self, sentences, params: Optional[Dict[str, Any]] = None):
        """
        :param sentences: Model input data.
        :param params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        :return: Model predictions.
        """
        # When the input is a single string, it is transformed into a DataFrame with one column
        # and row, but the encode function does not accept DataFrame input
        if type(sentences) == pd.DataFrame:
            sentences = sentences[0]

        # The encode API has additional parameters that we can add as kwargs.
        # See https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
        if params:
            try:
                return self.model.encode(sentences, **params)
            except TypeError as e:
                raise MlflowException.invalid_parameter_value(
                    "Received invalid parameter value for `params` argument"
                ) from e
        return self.model.encode(sentences)
