"""
The ``mlflow.pmdarima`` module provides an API for logging and loading ``pmdarima`` models.
This module exports univariate ``pmdarima`` models in the following formats:

Pmdarima format
    Serialized instance of a ``pmdarima`` model using pickle.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and for batch auditing
    of historical forecasts.

.. _Pmdarima:
    http://alkaline-ml.com/pmdarima/
"""
import os
import logging
import pickle
import warnings
import pandas as pd
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _CONDA_ENV_FILE_NAME,
    _process_pip_requirements,
    _process_conda_env,
    _CONSTRAINTS_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS


FLAVOR_NAME = "pmdarima"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.pmd"
_MODEL_TYPE_KEY = "model_type"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment that,
             at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("pmdarima")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    pmdarima_model,
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
    """
    Save a pmdarima ``ARIMA`` model or ``Pipeline`` object to a path on the local file system.

    :param pmdarima_model: pmdarima ``ARIMA`` or ``Pipeline`` model that has been ``fit`` on a
                           temporal series.
    :param path: Local path destination for the serialized model (in pickle format) is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param signature: :py:class:`Model Signature <mlflow.models.ModelSignature>` describes model
                      input and output :py:class:`Schema <mlflow.types.Schema>`. The model
                      signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        model = pmdarima.auto_arima(data)
                        predictions = model.predict(n_periods=30, return_conf_int=False)
                        signature = infer_signature(data, predictions)

                      .. Warning:: if utilizing confidence interval generation in the ``predict``
                        method of a ``pmdarima`` model (``return_conf_int=True``), the signature
                        will not be inferred due to the complex tuple return type when using the
                        native ``ARIMA.predict()`` API. ``infer_schema`` will function correctly
                        if using the ``pyfunc`` flavor of the model, though.

    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a ``Pandas DataFrame`` and
                          then serialized to json using the ``Pandas`` split-oriented format.
                          Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """
    import pmdarima

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_path = os.path.join(path, _MODEL_BINARY_FILE_NAME)
    _save_model(pmdarima_model, model_data_path)

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.pmdarima",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
    )
    flavor_conf = {
        _MODEL_TYPE_KEY: pmdarima_model.__class__.__name__,
        **model_bin_kwargs,
    }
    mlflow_model.add_flavor(
        FLAVOR_NAME, pmdarima_version=pmdarima.__version__, code=code_dir_subpath, **flavor_conf
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    pmdarima_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Log a ``pmdarima`` ``ARIMA`` or ``Pipeline`` object as an MLflow artifact for the current run.

    :param pmdarima_model: pmdarima ``ARIMA`` or ``Pipeline`` model that has been ``fit`` on a
                           temporal series.
    :param artifact_path: Run-relative artifact path to save the model instance to.
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

                        from mlflow.models.signature import infer_signature

                        model = pmdarima.auto_arima(data)
                        predictions = model.predict(n_periods=30, return_conf_int=False)
                        signature = infer_signature(data, predictions)

                      .. Warning:: if utilizing confidence interval generation in the ``predict``
                        method of a ``pmdarima`` model (``return_conf_int=True``), the signature
                        will not be inferred due to the complex tuple return type when using the
                        native ``ARIMA.predict()`` API. ``infer_schema`` will function correctly
                        if using the ``pyfunc`` flavor of the model, though.

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
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.pmdarima,
        registered_model_name=registered_model_name,
        pmdarima_model=pmdarima_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def load_model(model_uri, dst_path=None):
    """
    Load a ``pmdarima`` ``ARIMA`` model or ``Pipeline`` object from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``mlflow-artifacts:/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A ``pmdarima`` model instance
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    pmdarima_model_file_path = os.path.join(
        local_model_path, flavor_conf.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME)
    )

    return _load_model(pmdarima_model_file_path)


def _save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _load_model(path):
    with open(path, "rb") as pickled_model:
        model = pickle.load(pickled_model)
    return model


def _load_pyfunc(path):
    return _PmdarimaModelWrapper(_load_model(path))


class _PmdarimaModelWrapper:
    def __init__(self, pmdarima_model):
        import pmdarima

        self.pmdarima_model = pmdarima_model
        self._pmdarima_version = pmdarima.__version__

    def predict(self, dataframe) -> pd.DataFrame:
        df_schema = dataframe.columns.values.tolist()

        if len(dataframe) > 1:
            raise MlflowException(
                f"The provided prediction pd.DataFrame contains {len(dataframe)} rows. "
                "Only 1 row should be supplied.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        attrs = dataframe.to_dict(orient="index").get(0)
        n_periods = attrs.get("n_periods", None)

        if not n_periods:
            raise MlflowException(
                f"The provided prediction configuration pd.DataFrame columns ({df_schema}) do not "
                "contain the required column `n_periods` for specifying future prediction periods "
                "to generate.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if not isinstance(n_periods, int):
            raise MlflowException(
                f"The provided `n_periods` value {n_periods} must be an integer."
                f"provided type: {type(n_periods)}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # NB Any model that is trained with exogenous regressor elements will need to provide
        # `X` entries as a 2D array structure to the predict method.
        exogenous_regressor = attrs.get("X", None)

        if exogenous_regressor and Version(self._pmdarima_version) < Version("1.8.0"):
            warnings.warn(
                "An exogenous regressor element was provided in column 'X'. This is "
                "supported only in pmdarima version >= 1.8.0. Installed version: "
                f"{self._pmdarima_version}"
            )

        return_conf_int = attrs.get("return_conf_int", False)
        alpha = attrs.get("alpha", 0.05)

        if not isinstance(n_periods, int):
            raise MlflowException(
                "The prediction DataFrame must contain a column `n_periods` with "
                "an integer value for number of future periods to predict.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if Version(self._pmdarima_version) >= Version("1.8.0"):
            raw_predictions = self.pmdarima_model.predict(
                n_periods=n_periods,
                X=exogenous_regressor,
                return_conf_int=return_conf_int,
                alpha=alpha,
            )
        else:
            raw_predictions = self.pmdarima_model.predict(
                n_periods=n_periods,
                return_conf_int=return_conf_int,
                alpha=alpha,
            )

        if return_conf_int:
            ci_low, ci_high = list(zip(*raw_predictions[1]))
            predictions = pd.DataFrame.from_dict(
                {"yhat": raw_predictions[0], "yhat_lower": ci_low, "yhat_upper": ci_high}
            )
        else:
            predictions = pd.DataFrame.from_dict({"yhat": raw_predictions})

        return predictions
