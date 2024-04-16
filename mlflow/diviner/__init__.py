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
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
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
from mlflow.utils.file_utils import (
    get_total_file_size,
    shutil_copytree_without_file_permissions,
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _get_flavor_configuration_from_uri,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import dbfs_hdfs_uri_to_fuse_path, generate_tmp_dfs_path

FLAVOR_NAME = "diviner"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.div"
_MODEL_TYPE_KEY = "model_type"
_FLAVOR_KEY = "flavors"
_SPARK_MODEL_INDICATOR = "fit_with_spark"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced with the ``Diviner``
        flavor. Calls to :py:func:`save_model()` and :py:func:`log_model()` produce a pip
        environment that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("diviner")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced with the ``Diviner`` flavor
        that is produced by calls to :py:func:`save_model()` and :py:func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


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
    metadata=None,
    **kwargs,
):
    """Save a ``Diviner`` model object to a path on the local file system.

    Args:
        diviner_model: ``Diviner`` model that has been ``fit`` on a grouped temporal
            ``DataFrame``.
        path: Local path destination for the serialized model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` the flavor that this model is being added to.
        signature: :py:class:`Model Signature <mlflow.models.ModelSignature>` describes model
            input and output :py:class:`Schema <mlflow.types.Schema>`. The model
            signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                model = diviner.GroupedProphet().fit(data, ("region", "state"))
                predictions = model.predict(prediction_config)
                signature = infer_signature(data, predictions)

        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.

        kwargs: Optional configurations for Spark DataFrame storage iff the model has
            been fit in Spark.
            Current supported options:
            - `partition_by` for setting a (or several) partition columns as a list of \
            column names. Must be a list of strings of grouping key column(s).
            - `partition_count` for setting the number of part files to write from a \
            repartition per `partition_by` group. The default part file count is 200.
            - `dfs_tmpdir` for specifying the DFS temporary location where the model will \
            be stored while copying from a local file system to a Spark-supported "dbfs:/" \
            scheme.

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
    if metadata is not None:
        mlflow_model.metadata = metadata

    fit_with_spark = _save_diviner_model(diviner_model, path, **kwargs)
    flavor_conf = {_SPARK_MODEL_INDICATOR: fit_with_spark}

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.diviner",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
    )
    flavor_conf.update({_MODEL_TYPE_KEY: diviner_model.__class__.__name__}, **model_bin_kwargs)
    mlflow_model.add_flavor(
        FLAVOR_NAME, diviner_version=diviner.__version__, code=code_dir_subpath, **flavor_conf
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
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


def _save_diviner_model(diviner_model, path, **kwargs) -> bool:
    """
    Saves a Diviner model to the specified path. If the model was fit by using a Pandas DataFrame
    for the training data submitted to `fit`, directly save the Diviner model object.
    If the Diviner model was fit by using a Spark DataFrame, save the model components separately.
    The metadata and ancillary files to write (JSON and Pandas DataFrames) are written directly
    to a fuse mount location, which the Spark DataFrame that contains the individual serialized
    Diviner model objects is written by using the 'dbfs:' scheme path that Spark recognizes.
    """
    save_path = str(path.joinpath(_MODEL_BINARY_FILE_NAME))

    if getattr(diviner_model, "_fit_with_spark", False):
        # Validate that the path is a relative path early in order to fail fast prior to attempting
        # to write the (large) DataFrame to a tmp DFS path first and raise a path validation
        # Exception within MLflow when attempting to copy the temporary write files from DFS to
        # the file system path provided.
        if not os.path.isabs(path):
            raise MlflowException(
                "The save path provided must be a relative path. "
                f"The path submitted, '{path}' is an absolute path."
            )

        # Create a temporary DFS location to write the Spark DataFrame containing the models to.
        tmp_path = generate_tmp_dfs_path(kwargs.get("dfs_tmpdir", MLFLOW_DFS_TMP.get()))

        # Save the model Spark DataFrame to the temporary DFS location
        diviner_model._save_model_df_to_path(tmp_path, **kwargs)

        diviner_data_path = os.path.abspath(save_path)

        tmp_fuse_path = dbfs_hdfs_uri_to_fuse_path(tmp_path)
        shutil.move(src=tmp_fuse_path, dst=diviner_data_path)

        # Save the model metadata to the path location
        diviner_model._save_model_metadata_components_to_path(path=diviner_data_path)
        return True
    diviner_model.save(save_path)
    return False


def _load_model_fit_in_spark(local_model_path: str, flavor_conf, **kwargs):
    """
    Loads a Diviner model that has been fit (and saved) in the Spark variant.
    """
    # NB: To load the model DataFrame (which is a Spark DataFrame), Spark requires that the file
    # partitions are in DFS. In order to facilitate this, the model DataFrame (saved as parquet)
    # will be copied to a temporary DFS location. The remaining files can be read directly from
    # the local file system path, which is handled within the Diviner APIs.
    import diviner

    dfs_temp_directory = generate_tmp_dfs_path(kwargs.get("dfs_tmpdir", MLFLOW_DFS_TMP.get()))
    dfs_fuse_directory = dbfs_hdfs_uri_to_fuse_path(dfs_temp_directory)
    os.makedirs(dfs_fuse_directory)
    shutil_copytree_without_file_permissions(src_dir=local_model_path, dst_dir=dfs_fuse_directory)

    diviner_instance = getattr(diviner, flavor_conf[_MODEL_TYPE_KEY])
    load_directory = os.path.join(dfs_fuse_directory, flavor_conf[_MODEL_BINARY_KEY])

    return diviner_instance.load(load_directory)


def load_model(model_uri, dst_path=None, **kwargs):
    """Load a ``Diviner`` object from a local file or a run.

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
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist if provided. If unspecified, a local output
            path will be created.
        kwargs: Optional configuration options for loading of a Diviner model. For models
            that have been fit and saved using Spark, if a specific DFS temporary directory
            is desired for loading of Diviner models, use the keyword argument
            `"dfs_tmpdir"` to define the loading temporary path for the model during loading.

    Returns:
        A ``Diviner`` model instance.
    """
    model_uri = str(model_uri)

    flavor_conf = _get_flavor_configuration_from_uri(model_uri, FLAVOR_NAME, _logger)

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)

    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)

    if flavor_conf.get(_SPARK_MODEL_INDICATOR, False):
        return _load_model_fit_in_spark(local_model_path, flavor_conf, **kwargs)

    return _load_model(local_model_path, flavor_conf)


def _load_model(path, flavor_conf):
    """
    Loads a Diviner model instance that was not fit using Spark from a file system location.
    """
    import diviner

    local_path = pathlib.Path(path)

    diviner_model_path = local_path.joinpath(
        flavor_conf.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME)
    )

    diviner_instance = getattr(diviner, flavor_conf[_MODEL_TYPE_KEY])
    return diviner_instance.load(str(diviner_model_path))


def _load_pyfunc(path):
    local_path = pathlib.Path(path)

    # NB: reverting the dir walk that happens with pyfunc's loading implementation
    if local_path.is_file():
        local_path = local_path.parent

    flavor_conf = _get_flavor_configuration(local_path, FLAVOR_NAME)

    if flavor_conf.get(_SPARK_MODEL_INDICATOR):
        raise MlflowException(
            "The model being loaded was fit in Spark. Diviner models fit in "
            "Spark do not support loading as pyfunc."
        )

    return _DivinerModelWrapper(_load_model(local_path, flavor_conf))


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
    metadata=None,
    **kwargs,
):
    """Log a ``Diviner`` object as an MLflow artifact for the current run.

    Args:
        diviner_model: ``Diviner`` model that has been ``fit`` on a grouped temporal ``DataFrame``.
        artifact_path: Run-relative artifact path to save the model instance to.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: :py:class:`Model Signature <mlflow.models.ModelSignature>` describes model
            input and output :py:class:`Schema <mlflow.types.Schema>`. The model
            signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python
              :caption: Example

              from mlflow.models import infer_signature

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

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version
            to finish being created and is in ``READY`` status.
            By default, the function waits for five minutes.
            Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.

        kwargs: Additional arguments for :py:class:`mlflow.models.model.Model`
            Additionally, for models that have been fit in Spark, the following supported
            configuration options are available to set.
            Current supported options:
            - `partition_by` for setting a (or several) partition columns as a list of \
            column names. Must be a list of strings of grouping key column(s).
            - `partition_count` for setting the number of part files to write from a \
            repartition per `partition_by` group. The default part file count is 200.
            - `dfs_tmpdir` for specifying the DFS temporary location where the model will \
            be stored while copying from a local file system to a Spark-supported "dbfs:/" \
            scheme.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
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
        metadata=metadata,
        **kwargs,
    )


class _DivinerModelWrapper:
    def __init__(self, diviner_model):
        self.diviner_model = diviner_model

    def predict(self, dataframe, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """A method that allows a pyfunc implementation of this flavor to generate forecasted values
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

        Args:
            dataframe: A ``pandas.DataFrame`` that contains the required configuration for the
                appropriate ``Diviner`` type.

                For example, for ``GroupedProphet.forecast()``:

                - horizon : int
                - frequency: str

                predict_conf = pd.DataFrame({"horizon": 30, "frequency": "D"}, index=[0])
                forecast = pyfunc.load_pyfunc(model_uri=model_path).predict(predict_conf)

                Will generate 30 days of forecasted values for each group that the model
                was trained on.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            A Pandas DataFrame containing the forecasted values for each group key that was
            either trained or declared as a subset with a ``groups`` entry in the ``dataframe``
            configuration argument.
        """

        from diviner import GroupedPmdarima, GroupedProphet

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
