"""The ``flavor`` module provides an example for a custom model flavor for ``sktime`` library.

This module exports ``sktime`` models in the following formats:

sktime (native) format
    This is the main flavor that can be loaded back into sktime, which relies on pickle
    internally to serialize a model.
mlflow.pyfunc
    Produced for use by generic pyfunc-based deployment tools and batch inference.

    The `pyfunc` flavor of the model supports sktime predict methods `predict`,
    `predict_interval`, `predict_proba`, `predict_quantiles`, `predict_var`.

    The interface for utilizing a sktime model loaded as a `pyfunc` type for
    generating forecasts requires passing an exogenous regressor as Pandas
    DataFrame to the `pyfunc.predict()` method (an empty DataFrame must be
    passed if no exogenous regressor is used). The configuration of predict
    methods and parameter values passed to the predict methods is defined by
    a dictionary to be saved as an attribute of the fitted sktime model
    instance. If no prediction configuration is defined `pyfunc.predict()`
    will return output from sktime `predict` method. Note that for `pyfunc`
    flavor the forecasting horizon `fh` must be passed to the fit method.

    Predict methods and parameter values for `pyfunc` flavor can be defined
    in two ways: `Dict[str, dict]` if parameter values are passed to
    `pyfunc.predict()`, for example
    `{"predict_method": {"predict": {}, "predict_interval": {"coverage": [0.1, 0.9]}}`.
    `Dict[str, list]`, with default parameters in predict method, for example
    `{"predict_method": ["predict", "predict_interval"}` (Note: when including
    `predict_proba` method the former appraoch must be followed as `quantiles`
    parameter has to be provided by the user). If no prediction config is defined
    `pyfunc.predict()` will return output from sktime `predict()` method.
"""
import logging
import os
import pickle
import flavor

import pandas as pd
import yaml

import sktime
from sktime.utils.multiindex import flatten_multiindex

from mlflow import pyfunc
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

FLAVOR_NAME = "flavor"

PYFUNC_PREDICT_CONF = "pyfunc_predict_conf"
PYFUNC_PREDICT_CONF_KEY = "predict_method"
SKTIME_PREDICT = "predict"
SKTIME_PREDICT_INTERVAL = "predict_interval"
SKTIME_PREDICT_PROBA = "predict_proba"
SKTIME_PREDICT_QUANTILES = "predict_quantiles"
SKTIME_PREDICT_VAR = "predict_var"
SUPPORTED_SKTIME_PREDICT_METHODS = [
    SKTIME_PREDICT,
    SKTIME_PREDICT_INTERVAL,
    SKTIME_PREDICT_PROBA,
    SKTIME_PREDICT_QUANTILES,
    SKTIME_PREDICT_VAR,
]

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"
SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE,
]

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """Create list of default pip requirements for MLflow Models.

    Returns
    -------
    list of default pip requirements for MLflow Models produced by this flavor.
    Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
    that, at a minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("sktime")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """Return default Conda environment for MLflow Models.

    Returns
    -------
    The default Conda environment for MLflow Models produced by calls to
    :func:`save_model()` and :func:`log_model()`
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


def save_model(
    sktime_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_PICKLE,
):
    """Save a sktime model to a path on the local file system.

    Parameters
    ----------
    sktime_model :
        Fitted sktime model object.
    path : str
        Local path where the model is to be saved.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    mlflow_model: mlflow.models.Model, optional (default=None)
        mlflow.models.Model configuration to which to add the python_function flavor.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models.signature import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)

        .. Warning:: if performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a sktime model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though.
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["sktime", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                "{supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "model.pkl"
    model_data_path = os.path.join(path, model_data_subpath)
    _save_model(sktime_model, model_data_path, serialization_format=serialization_format)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="flavor",
        model_path=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sktime_version=sktime.__version__,
        serialization_format=serialization_format,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            include_cloudpickle = serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
            default_reqs = get_default_pip_requirements(include_cloudpickle)
            default_reqs = sorted(default_reqs)
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


def log_model(
    sktime_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature=None,
    input_example=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_PICKLE,
    **kwargs,
):
    """
    Log a sktime model as an MLflow artifact for the current run.

    Parameters
    ----------
    sktime_model : fitted sktime model
        Fitted sktime model object.
    artifact_path : str
        Run-relative artifact path to save the model to.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    registered_model_name : str, optional (default=None)
        If given, create a model version under ``registered_model_name``, also creating
        a registered model if one with the given name does not exist.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models.signature import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)

        .. Warning:: if performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a sktime model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though. The ``pyfunc`` flavor of the model supports sktime
          predict methods ``predict``, ``predict_interval``, ``predict_quantiles``
          and ``predict_var`` while ``predict_proba`` and ``predict_residuals`` are
          currently not supported.
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    await_registration_for : int, optional (default=None)
        Number of seconds to wait for the model version to finish being created and is
        in ``READY`` status. By default, the function waits for five minutes. Specify 0
        or None to skip waiting.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["sktime", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    kwargs:
        Additional arguments for :py:class:`mlflow.models.model.Model`

    Returns
    -------
    A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
    metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=flavor,
        registered_model_name=registered_model_name,
        sktime_model=sktime_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        serialization_format=serialization_format,
        **kwargs,
    )


def load_model(model_uri, dst_path=None):
    """
    Load a sktime model from a local file or a run.

    Parameters
    ----------
    model_uri : str
        The location, in URI format, of the MLflow model. For example:

                    - ``/Users/me/path/to/local/model``
                    - ``relative/path/to/local/model``
                    - ``s3://my_bucket/path/to/model``
                    - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                    - ``mlflow-artifacts:/path/to/model``

        For more information about supported URI schemes, see
        `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
        artifact-locations>`_.
    dst_path : str, optional (default=None)
        The local filesystem path to which to download the model artifact.This
        directory must already exist. If unspecified, a local output path will
        be created.

    Returns
    -------
    A sktime model instance.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    sktime_model_file_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    serialization_format = flavor_conf.get("serialization_format", SERIALIZATION_FORMAT_PICKLE)
    return _load_model(path=sktime_model_file_path, serialization_format=serialization_format)


def _save_model(model, path, serialization_format):
    with open(path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            cloudpickle.dump(model, out)
        else:
            raise MlflowException(
                message="Unrecognized serialization format: "
                "{serialization_format}".format(serialization_format=serialization_format),
                error_code=INTERNAL_ERROR,
            )


def _load_model(path, serialization_format):
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                "{supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    with open(path, "rb") as pickled_model:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(pickled_model)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(pickled_model)


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Parameters
    ----------
    path : str
        Local filesystem path to the MLflow Model with the sktime flavor.

    """
    if os.path.isfile(path):
        serialization_format = SERIALIZATION_FORMAT_PICKLE
        _logger.warning("Loading procedure in older versions of MLflow using pickle.load()")
    else:
        try:
            sktime_flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
            serialization_format = sktime_flavor_conf.get(
                "serialization_format", SERIALIZATION_FORMAT_PICKLE
            )
        except MlflowException:
            _logger.warning(
                "Could not find sktime flavor configuration during model "
                "loading process. Assuming 'pickle' serialization format."
            )
            serialization_format = SERIALIZATION_FORMAT_PICKLE

        pyfunc_flavor_conf = _get_flavor_configuration(
            model_path=path, flavor_name=pyfunc.FLAVOR_NAME
        )
        path = os.path.join(path, pyfunc_flavor_conf["model_path"])

    return _SktimeModelWrapper(_load_model(path, serialization_format=serialization_format))


class _SktimeModelWrapper:
    def __init__(self, sktime_model):
        self.sktime_model = sktime_model

    def predict(self, X):
        X = None if X.empty else X
        raw_predictions = {}

        if not hasattr(self.sktime_model, "pyfunc_predict_conf"):
            raw_predictions[SKTIME_PREDICT] = self.sktime_model.predict(X=X)

        else:
            if not isinstance(self.sktime_model.pyfunc_predict_conf, dict):
                raise MlflowException(
                    f"Attribute {PYFUNC_PREDICT_CONF} must be of type dict.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if PYFUNC_PREDICT_CONF_KEY not in self.sktime_model.pyfunc_predict_conf:
                raise MlflowException(
                    f"Attribute {PYFUNC_PREDICT_CONF} must contain "
                    f"a dictionary key {PYFUNC_PREDICT_CONF_KEY}.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if isinstance(self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY], list):
                predict_methods = self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY]
                predict_params = False
            elif isinstance(self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY], dict):
                predict_methods = list(
                    self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY].keys()
                )
                predict_params = True
            else:
                raise MlflowException(
                    "Dictionary value must be of type dict or list.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if not set(predict_methods).issubset(set(SUPPORTED_SKTIME_PREDICT_METHODS)):
                raise MlflowException(
                    f"The provided {PYFUNC_PREDICT_CONF_KEY} values must be "
                    f"a subset of {SUPPORTED_SKTIME_PREDICT_METHODS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if SKTIME_PREDICT in predict_methods:
                raw_predictions[SKTIME_PREDICT] = self.sktime_model.predict(X=X)

            if SKTIME_PREDICT_INTERVAL in predict_methods:
                if predict_params:
                    coverage = (
                        0.9
                        if "coverage"
                        not in self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                            SKTIME_PREDICT_INTERVAL
                        ]
                        else self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                            SKTIME_PREDICT_INTERVAL
                        ]["coverage"]
                    )
                else:
                    coverage = 0.9

                raw_predictions[SKTIME_PREDICT_INTERVAL] = self.sktime_model.predict_interval(
                    X=X, coverage=coverage
                )

            if SKTIME_PREDICT_PROBA in predict_methods:
                if not isinstance(
                    self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY], dict
                ):
                    raise MlflowException(
                        f"Method {SKTIME_PREDICT_PROBA} requires passing a dictionary.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                if (
                    "quantiles"
                    not in self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                        SKTIME_PREDICT_PROBA
                    ]
                ):
                    raise MlflowException(
                        f"Method {SKTIME_PREDICT_PROBA} requires passing " f"quantile values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                quantiles = self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                    SKTIME_PREDICT_PROBA
                ]["quantiles"]
                marginal = (
                    True
                    if "marginal"
                    not in self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                        SKTIME_PREDICT_PROBA
                    ]
                    else self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                        SKTIME_PREDICT_PROBA
                    ]["marginal"]
                )

                y_pred_dist = self.sktime_model.predict_proba(X=X, marginal=marginal)
                y_pred_dist_quantiles = pd.DataFrame(y_pred_dist.quantile(quantiles))
                y_pred_dist_quantiles.columns = [f"Quantiles_{q}" for q in quantiles]
                y_pred_dist_quantiles.index = y_pred_dist.parameters["loc"].index

                raw_predictions[SKTIME_PREDICT_PROBA] = y_pred_dist_quantiles

            if SKTIME_PREDICT_QUANTILES in predict_methods:
                if predict_params:
                    alpha = (
                        None
                        if "alpha"
                        not in self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                            SKTIME_PREDICT_QUANTILES
                        ]
                        else self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                            SKTIME_PREDICT_QUANTILES
                        ]["alpha"]
                    )
                else:
                    alpha = None
                raw_predictions[SKTIME_PREDICT_QUANTILES] = self.sktime_model.predict_quantiles(
                    X=X, alpha=alpha
                )

            if SKTIME_PREDICT_VAR in predict_methods:
                if predict_params:
                    cov = (
                        False
                        if "cov"
                        not in self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                            SKTIME_PREDICT_VAR
                        ]
                        else self.sktime_model.pyfunc_predict_conf[PYFUNC_PREDICT_CONF_KEY][
                            SKTIME_PREDICT_VAR
                        ]["cov"]
                    )
                else:
                    cov = False
                raw_predictions[SKTIME_PREDICT_VAR] = self.sktime_model.predict_var(X=X, cov=cov)

        for k, v in raw_predictions.items():
            if hasattr(v, "columns") and isinstance(v.columns, pd.MultiIndex):
                raw_predictions[k].columns = flatten_multiindex(v)

        if len(raw_predictions) > 1:
            predictions = pd.concat(
                list(raw_predictions.values()),
                axis=1,
                keys=list(raw_predictions.keys()),
            )
            predictions.columns = flatten_multiindex(predictions)
        else:
            predictions = raw_predictions[list(raw_predictions.keys())[0]]

        return predictions
