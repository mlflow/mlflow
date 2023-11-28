"""
The ``mlflow.spacy`` module provides an API for logging and loading spaCy models.
This module exports spacy models with the following flavors:

spaCy (native) format
    This is the main flavor that can be loaded back into spaCy.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference, this
    flavor is created only if spaCy's model pipeline has at least one
    `TextCategorizer <https://spacy.io/api/textcategorizer>`_.
"""
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
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
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "spacy"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("spacy")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    spacy_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """Save a spaCy model to a path on the local file system.

    Args:
        spacy_model: spaCy model to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                    containing file dependencies). These files are *prepended* to the system
                    path when the model is loaded.
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                   describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                   The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                   from datasets with valid model input (e.g. the training dataset with target
                   column omitted) and valid model output (e.g. model predictions generated on
                   the training dataset), for example:

                   .. code-block:: python

                      from mlflow.models import infer_signature

                      train = df.drop_column("target_label")
                      predictions = ...  # compute model predictions
                      signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """
    import spacy

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    model_data_subpath = "model.spacy"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    # Save spacy-model
    spacy_model.to_disk(path=model_data_path)
    # Save the pyfunc flavor if at least one text categorizer in spaCy pipeline
    if any(
        isinstance(pipe_component[1], spacy.pipeline.TextCategorizer)
        for pipe_component in spacy_model.pipeline
    ):
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.spacy",
            data=model_data_subpath,
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_dir_subpath,
        )
    else:
        _logger.warning(
            "Generating only the spacy flavor for the provided spacy model. This means the model "
            "can be loaded back via `mlflow.spacy.load_model`, but cannot be loaded back using "
            "pyfunc APIs like `mlflow.pyfunc.load_model` or via the `mlflow models` CLI commands. "
            "MLflow will only generate the pyfunc flavor for spacy models containing a pipeline "
            "component that is an instance of spacy.pipeline.TextCategorizer."
        )

    mlflow_model.add_flavor(
        FLAVOR_NAME, spacy_version=spacy.__version__, data=model_data_subpath, code=code_dir_subpath
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    spacy_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """Log a spaCy model as an MLflow artifact for the current run.

    Args:
        spacy_model: spaCy model to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                    containing file dependencies). These files are *prepended* to the system
                    path when the model is loaded.
        registered_model_name: If given, create a model version under
                               ``registered_model_name``, also creating a registered model if one
                               with the given name does not exist.

        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                   describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                   The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                   from datasets with valid model input (e.g. the training dataset with target
                   column omitted) and valid model output (e.g. model predictions generated on
                   the training dataset), for example:

                   .. code-block:: python

                      from mlflow.models import infer_signature

                      train = df.drop_column("target_label")
                      predictions = ...  # compute model predictions
                      signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
        kwargs: kwargs to pass to ``spacy.save_model`` method.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.spacy,
        registered_model_name=registered_model_name,
        spacy_model=spacy_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def _load_model(path):
    import spacy

    path = os.path.abspath(path)
    return spacy.load(path)


class _SpacyModelWrapper:
    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def predict(
        self, dataframe, params: Optional[Dict[str, Any]] = None  # pylint: disable=unused-argument
    ):
        """Only works for predicting using text categorizer.
        Not suitable for other pipeline components (e.g: parser)

        Args:
            dataframe: pandas dataframe containing texts to be categorized
                       expected shape is (n_rows,1 column)
            params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        Returns:
            dataframe with predictions
        """
        if len(dataframe.columns) != 1:
            raise MlflowException("Shape of input dataframe must be (n_rows, 1column)")

        return pd.DataFrame(
            {"predictions": dataframe.iloc[:, 0].apply(lambda text: self.spacy_model(text).cats)}
        )


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``spacy`` flavor.
    """
    return _SpacyModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """Load a spaCy model from a local file (if ``run_id`` is ``None``) or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
                  This directory must already exist. If unspecified, a local output
                  path will be created.

    Returns:
        A spaCy loaded model
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    # Flavor configurations for models saved in MLflow version <= 0.8.0 may not contain a
    # `data` key; in this case, we assume the model artifact path to be `model.spacy`
    spacy_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.spacy"))
    return _load_model(path=spacy_model_file_path)
