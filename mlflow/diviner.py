"""
The ``mlflow.diviner`` module provides an API for logging, saving and loading ``diviner`` models.
Diviner wraps several popular open source time series forecasting libraries in a unified API that
permits training, back-testing cross validation, and forecasting inference for groups of related
series.
This module exports groups of univariate ``diviner`` models in the following formats:

Diviner format
    Serialized instance of a ``diviner`` model type using native diviner serializers.
    (e.g., "GroupedProphet" or "GroupedPmdarima")
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and for batch auditing
    of historical forecasts.

.. _Diviner:
    https://databricks-diviner.readthedocs.io/en/latest/index.html
"""
import logging
import pathlib
import yaml
import pandas as pd
from typing import Tuple, List
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
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
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _validate_and_copy_code_paths,
    _get_flavor_configuration,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.models.utils import _save_example
from mlflow.utils.requirements_utils import _get_pinned_requirement


FLAVOR_NAME = "diviner"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.div"
_MODEL_TYPE_KEY = "model_type"
_FLAVOR_KEY = "flavors"

_logger = logging.getLogger(__name__)


@experimental
def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced with the ``Diviner``
             flavor. Calls to :py:func:`save_model()` and :py:func:`log_model()` produce a pip
             environment that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("diviner")]


@experimental
def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced with the ``Diviner`` flavor
             that is produced by calls to :py:func:`save_model()` and :py:func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    diviner_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save a ``Diviner`` model object to a path on the local file system.

    :param diviner_model: ``Diviner`` model that has been ``fit`` on a grouped temporal
                          ``DataFrame``.
    :param path: Local path destination for the serialized model is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` the flavor that this model is being added to.
    :param signature: :py:class:`Model Signature <mlflow.models.ModelSignature>` describes model
                      input and output :py:class:`Schema <mlflow.types.Schema>`. The model
                      signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: py

                        from mlflow.models.signature import infer_signature

                        model = diviner.GroupedProphet().fit(data, ("region", "state"))
                        predictions = model.predict(prediction_config)
                        signature = infer_signature(data, predictions)

    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a ``Pandas DataFrame`` and
                          then serialized to json using the ``Pandas`` split-oriented format.
                          Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    import diviner

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = pathlib.Path(path).absolute()

    _validate_and_prepare_target_save_path(str(path))

    # NB: When moving to native pathlib implementations, path encoding as string will not be needed.
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, str(path))

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, str(path))

    diviner_model.save(str(path.joinpath(_MODEL_BINARY_FILE_NAME)))

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.diviner",
        env=_CONDA_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
    )
    flavor_conf = {_MODEL_TYPE_KEY: diviner_model.__class__.__name__, **model_bin_kwargs}
    mlflow_model.add_flavor(
        FLAVOR_NAME, diviner_version=diviner.__version__, code=code_dir_subpath, **flavor_conf
    )
    mlflow_model.save(str(path.joinpath(MLMODEL_FILE_NAME)))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                str(path), FLAVOR_NAME, fallback=default_reqs
            )
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
def load_model(model_uri, dst_path=None):
    """
    Load a ``Diviner`` object from a local file or a run.

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

    :return: A ``Diviner`` model instance
    """

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)

    return _load_model(local_model_path)


def _get_diviner_instance_type(path) -> str:
    local_path = pathlib.Path(path)
    diviner_model_info_path = local_path.joinpath(MLMODEL_FILE_NAME)
    diviner_model_info = yaml.safe_load(diviner_model_info_path.read_text())
    return diviner_model_info.get(_FLAVOR_KEY).get(FLAVOR_NAME).get(_MODEL_TYPE_KEY)


def _load_model(path):

    import diviner

    local_path = pathlib.Path(path)

    flavor_conf = _get_flavor_configuration(model_path=str(local_path), flavor_name=FLAVOR_NAME)

    _add_code_from_conf_to_system_path(str(local_path), flavor_conf)

    diviner_model_path = local_path.joinpath(
        flavor_conf.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME)
    )

    diviner_instance = getattr(diviner, _get_diviner_instance_type(path))
    return diviner_instance.load(str(diviner_model_path))


def _load_pyfunc(path):
    local_path = pathlib.Path(path)
    # NB: reverting the dir walk that happens with pyfunc's loading implementation
    if local_path.is_file():
        local_path = local_path.parent
    return _DivinerModelWrapper(_load_model(local_path))


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    diviner_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log a ``Diviner`` object as an MLflow artifact for the current run.

    :param diviner_model: ``Diviner`` model that has been ``fit`` on a grouped temporal
                          ``DataFrame``.
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

                      .. code-block:: py

                        from mlflow.models.signature import infer_signature

                        auto_arima_obj = AutoARIMA(out_of_sample_size=60, maxiter=100)
                        base_auto_arima = GroupedPmdarima(model_template=auto_arima_obj).fit(
                            df=training_data,
                            group_key_columns=("region", "state"),
                            y_col="y",
                            datetime_col="ds",
                            silence_warnings=True,
                        )
                        predictions = model.predict(n_periods=30, alpha=0.05, return_conf_int=True)
                        signature = infer_signature(data, predictions)

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
    :param kwargs: Additional arguments for :py:class:`mlflow.models.model.Model`
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.diviner,
        registered_model_name=registered_model_name,
        diviner_model=diviner_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


class _DivinerModelWrapper:
    def __init__(self, diviner_model):
        self.diviner_model = diviner_model

    def predict(self, dataframe) -> pd.DataFrame:
        """
        A method that allows a pyfunc implementation of this flavor to generate forecasted values
        from the end of a trained Diviner model's training series per group.

        The implementation here encapsulates a config-based switch of method calling. In short:
          *  If the ``DataFrame`` supplied to this method contains a column ``groups`` whose
             first row of data is of type List[tuple[str]] (containing the series-identifying
             group keys that were generated to identify a single underlying model during training),
             the caller will resolve to the method ``predict_groups()`` in each of the underlying
             wrapped libraries (i.e., ``GroupedProphet.predict_groups()``).
          *  If the ``DataFrame`` supplied does not contain the column name ``groups``, then the
             specific forecasting method that is primitive-driven (for ``GroupedProphet``, the
             ``predict()`` method mirrors that of ``Prophet``'s, requiring a ``DataFrame``
             submitted with explicit datetime values per group which is not a tenable
             implementation for pyfunc or RESTful serving) is utilized. For ``GroupedProphet``,
             this is the ``.forecast()`` method, while for ``GroupedPmdarima``, this is the
             ``.predict()`` method.

        :param dataframe: A ``pandas.DataFrame`` that contains the required configuration for the
                          appropriate ``Diviner`` type.

                          For example, for ``GroupedProphet.forecast()``:

                          - horizon : int
                          - frequency: str

                          predict_conf = pd.DataFrame({"horizon": 30, "frequency": "D"}, index=[0])
                          forecast = pyfunc.load_pyfunc(model_uri=model_path).predict(predict_conf)

                          Will generate 30 days of forecasted values for each group that the model
                          was trained on.
        :return: A Pandas DataFrame containing the forecasted values for each group key that was
                 either trained or declared as a subset with a ``groups`` entry in the ``dataframe``
                 configuration argument.
        """

        from diviner import GroupedProphet, GroupedPmdarima

        schema = dataframe.columns.values.tolist()

        conf = dataframe.to_dict(orient="index").get(0)

        # required parameter extraction and validation
        horizon = conf.get("horizon", None)
        n_periods = conf.get("n_periods", None)
        if n_periods and horizon and n_periods != horizon:
            raise MlflowException(
                "The provided prediction configuration contains both `n_periods` and `horizon` "
                "with different values. Please provide only one of these integer values.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        else:
            if not n_periods and horizon:
                n_periods = horizon

        if not n_periods:
            raise MlflowException(
                "The provided prediction configuration Pandas DataFrame does not contain either "
                "the `n_periods` or `horizon` columns. At least one of these must be specified "
                f"with a valid integer value. Configuration schema: {schema}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if not isinstance(n_periods, int):
            raise MlflowException(
                "The `n_periods` column contains invalid data. Supplied type must be an integer. "
                f"Type supplied: {type(n_periods)}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        frequency = conf.get("frequency", None)

        if isinstance(self.diviner_model, GroupedProphet) and not frequency:
            raise MlflowException(
                "Diviner's GroupedProphet model requires a `frequency` value to be submitted in "
                "Pandas date_range format. The submitted configuration Pandas DataFrame does not "
                f"contain a `frequency` column. Configuration schema: {schema}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        predict_col = conf.get("predict_col", None)
        predict_groups = conf.get("groups", None)

        if predict_groups and not isinstance(predict_groups, List):
            raise MlflowException(
                "Specifying a group subset for prediction requires groups to be defined as a "
                f"[List[(Tuple|List)[<group_keys>]]. Submitted group type: {type(predict_groups)}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # NB: json serialization of a tuple converts the tuple to a List. Diviner requires a
        # List of Tuples to be input to the group_prediction API. This conversion is for utilizing
        # the pyfunc flavor through the serving API.
        if predict_groups and not isinstance(predict_groups[0], Tuple):
            predict_groups = [tuple(group) for group in predict_groups]

        if isinstance(self.diviner_model, GroupedProphet):
            # We're wrapping two different endpoints to Diviner here for the pyfunc implementation.
            # Since we're limited by a single endpoint, we can address redirecting to the
            # method ``predict_groups()`` which will allow for a subset of groups to be forecasted
            # if the prediction configuration DataFrame contains a List[tuple[str]]] in the
            # ``groups`` column. If this column is not present, all groups will be used to generate
            # forecasts, utilizing the less computationally complex method ``forecast``.
            if not predict_groups:
                prediction_df = self.diviner_model.forecast(horizon=n_periods, frequency=frequency)
            else:
                group_kwargs = {k: v for k, v in conf.items() if k in {"predict_col", "on_error"}}
                prediction_df = self.diviner_model.predict_groups(
                    groups=predict_groups, horizon=n_periods, frequency=frequency, **group_kwargs
                )

            if predict_col is not None:
                prediction_df.rename(columns={"yhat": predict_col}, inplace=True)

        elif isinstance(self.diviner_model, GroupedPmdarima):
            # As above, we're redirecting the prediction request to one of two different methods
            # for ``Diviner``'s pmdarima implementation. If the ``groups`` column is present with
            # a list of tuples of keys to lookup, ``predict_groups()`` will be used. Otherwise,
            # the standard ``predict()`` method will be called to generate forecasts for all groups
            # that were trained on.
            restricted_keys = {"n_periods", "horizon", "frequency", "groups"}

            predict_conf = {k: v for k, v in conf.items() if k not in restricted_keys}
            if not predict_groups:
                prediction_df = self.diviner_model.predict(n_periods=n_periods, **predict_conf)
            else:
                prediction_df = self.diviner_model.predict_groups(
                    groups=predict_groups, n_periods=n_periods, **predict_conf
                )
        else:
            raise MlflowException(
                f"The Diviner model instance type '{type(self.diviner_model)}' is not supported "
                f"in version {mlflow.__version__} of MLflow.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return prediction_df
