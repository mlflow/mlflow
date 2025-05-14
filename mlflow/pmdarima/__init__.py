"""
The ``mlflow.pmdarima`` module provides an API for logging and loading ``pmdarima`` models.
This module exports univariate ``pmdarima`` models in the following formats:

Pmdarima format
    Serialized instance of a ``pmdarima`` model using pickle.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and for batch auditing
    of historical forecasts.

 .. code-block:: python
    :caption: Example

    import pandas as pd
    import mlflow
    import mlflow.pyfunc
    import pmdarima
    from pmdarima import auto_arima


    # Define a custom model class
    class PmdarimaWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.model = context.artifacts["model"]

        def predict(self, context, model_input):
            return self.model.predict(n_periods=model_input.shape[0])


    # Specify locations of source data and the model artifact
    SOURCE_DATA = "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
    ARTIFACT_PATH = "model"

    # Read data and recode columns
    sales_data = pd.read_csv(SOURCE_DATA)
    sales_data.rename(columns={"y": "sales", "ds": "date"}, inplace=True)

    # Split the data into train/test
    train_size = int(0.8 * len(sales_data))
    train, _ = sales_data[:train_size], sales_data[train_size:]

    # Create the model
    model = pmdarima.auto_arima(train["sales"], seasonal=True, m=12)

    # Log the model
    with mlflow.start_run():
        wrapper = PmdarimaWrapper()
        mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            artifacts={"model": mlflow.pyfunc.model_to_dict(model)},
        )


.. _Pmdarima:
    http://alkaline-ml.com/pmdarima/
"""

import logging
import os
import pickle
import warnings
from typing import Any, Optional

import pandas as pd
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
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
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "pmdarima"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.pmd"
_MODEL_TYPE_KEY = "model_type"


_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor. Calls to
        :func:`save_model()` and :func:`log_model()` produce a pip environment that, at a minimum,
        contains these requirements.
    """

    return [_get_pinned_requirement("pmdarima")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
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

    Args:
        pmdarima_model: pmdarima ``ARIMA`` or ``Pipeline`` model that has been ``fit`` on a
            temporal series.
        path: Local path destination for the serialized model (in pickle format) is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            class that describes the model's inputs and outputs. If not specified but an
            ``input_example`` is supplied, a signature will be automatically inferred
            based on the supplied input example and model. To disable automatic signature
            inference when providing an input example, set ``signature`` to ``False``.
            To manually infer a model signature, call
            :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
            with valid model inputs, such as a training dataset with the target column
            omitted, and valid model outputs, like model predictions made on the training
            dataset, for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                model = pmdarima.auto_arima(data)
                predictions = model.predict(n_periods=30, return_conf_int=False)
                signature = infer_signature(data, predictions)

            .. Warning:: if utilizing confidence interval generation in the ``predict``
                method of a ``pmdarima`` model (``return_conf_int=True``), the signature
                will not be inferred due to the complex tuple return type when using the
                native ``ARIMA.predict()`` API. ``infer_schema`` will function correctly
                if using the ``pyfunc`` flavor of the model, though.
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}

    .. code-block:: python
        :caption: Example

        import pandas as pd
        import mlflow
        import pmdarima

        # Specify locations of source data and the model artifact
        SOURCE_DATA = "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
        ARTIFACT_PATH = "model"

        # Read data and recode columns
        sales_data = pd.read_csv(SOURCE_DATA)
        sales_data.rename(columns={"y": "sales", "ds": "date"}, inplace=True)

        # Split the data into train/test
        train_size = int(0.8 * len(sales_data))
        train, test = sales_data[:train_size], sales_data[train_size:]

        with mlflow.start_run():
            # Create the model
            model = pmdarima.auto_arima(train["sales"], seasonal=True, m=12)

            # Save the model to the specified path
            mlflow.pmdarima.save_model(model, "model")
    """

    import pmdarima

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = _PmdarimaModelWrapper(pmdarima_model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
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
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
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
    artifact_path: Optional[str] = None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    name: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, Any]] = None,
    model_type: Optional[str] = None,
    step: int = 0,
    model_id: Optional[str] = None,
    **kwargs,
):
    """
    Logs a ``pmdarima`` ``ARIMA`` or ``Pipeline`` object as an MLflow artifact for the current run.

    Args:
        pmdarima_model: pmdarima ``ARIMA`` or ``Pipeline`` model that has been ``fit`` on a
            temporal series.
        artifact_path: Deprecated. Use `name` instead.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            class that describes the model's inputs and outputs. If not specified but an
            ``input_example`` is supplied, a signature will be automatically inferred
            based on the supplied input example and model. To disable automatic signature
            inference when providing an input example, set ``signature`` to ``False``.
            To manually infer a model signature, call
            :py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets
            with valid model inputs, such as a training dataset with the target column
            omitted, and valid model outputs, like model predictions made on the training
            dataset, for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                model = pmdarima.auto_arima(data)
                predictions = model.predict(n_periods=30, return_conf_int=False)
                signature = infer_signature(data, predictions)

            .. Warning:: if utilizing confidence interval generation in the ``predict``
                method of a ``pmdarima`` model (``return_conf_int=True``), the signature
                will not be inferred due to the complex tuple return type when using the
                native ``ARIMA.predict()`` API. ``infer_schema`` will function correctly
                if using the ``pyfunc`` flavor of the model, though.

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version
            to finish being created and is in ``READY`` status.
            By default, the function waits for five minutes.
            Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        kwargs: Additional arguments for :py:class:`mlflow.models.model.Model`

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import pandas as pd
        import mlflow
        from mlflow.models import infer_signature
        import pmdarima
        from pmdarima.metrics import smape

        # Specify locations of source data and the model artifact
        SOURCE_DATA = "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
        ARTIFACT_PATH = "model"

        # Read data and recode columns
        sales_data = pd.read_csv(SOURCE_DATA)
        sales_data.rename(columns={"y": "sales", "ds": "date"}, inplace=True)

        # Split the data into train/test
        train_size = int(0.8 * len(sales_data))
        train, test = sales_data[:train_size], sales_data[train_size:]

        with mlflow.start_run():
            # Create the model
            model = pmdarima.auto_arima(train["sales"], seasonal=True, m=12)

            # Calculate metrics
            prediction = model.predict(n_periods=len(test))
            metrics = {"smape": smape(test["sales"], prediction)}

            # Infer signature
            input_sample = pd.DataFrame(train["sales"])
            output_sample = pd.DataFrame(model.predict(n_periods=5))
            signature = infer_signature(input_sample, output_sample)

            # Log model
            mlflow.pmdarima.log_model(model, name=ARTIFACT_PATH, signature=signature)
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
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
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


def load_model(model_uri, dst_path=None):
    """
    Load a ``pmdarima`` ``ARIMA`` model or ``Pipeline`` object from a local file or a run.

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
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A ``pmdarima`` model instance

    .. code-block:: python
        :caption: Example

        import pandas as pd
        import mlflow
        from mlflow.models import infer_signature
        import pmdarima
        from pmdarima.metrics import smape

        # Specify locations of source data and the model artifact
        SOURCE_DATA = "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
        ARTIFACT_PATH = "model"

        # Read data and recode columns
        sales_data = pd.read_csv(SOURCE_DATA)
        sales_data.rename(columns={"y": "sales", "ds": "date"}, inplace=True)

        # Split the data into train/test
        train_size = int(0.8 * len(sales_data))
        train, test = sales_data[:train_size], sales_data[train_size:]

        with mlflow.start_run():
            # Create the model
            model = pmdarima.auto_arima(train["sales"], seasonal=True, m=12)

            # Calculate metrics
            prediction = model.predict(n_periods=len(test))
            metrics = {"smape": smape(test["sales"], prediction)}

            # Infer signature
            input_sample = pd.DataFrame(train["sales"])
            output_sample = pd.DataFrame(model.predict(n_periods=5))
            signature = infer_signature(input_sample, output_sample)

            # Log model
            input_example = input_sample.head()
            mlflow.pmdarima.log_model(
                model, name=ARTIFACT_PATH, signature=signature, input_example=input_example
            )

            # Get the model URI for loading
            model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

        # Load the model
        loaded_model = mlflow.pmdarima.load_model(model_uri)
        # Forecast for the next 60 days
        forecast = loaded_model.predict(n_periods=60)
        print(f"forecast: {forecast}")

    .. code-block:: text
        :caption: Output

        forecast:
        234    382452.397246
        235    380639.458720
        236    359805.611219
        ...
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
        return pickle.load(pickled_model)


def _load_pyfunc(path):
    return _PmdarimaModelWrapper(_load_model(path))


class _PmdarimaModelWrapper:
    def __init__(self, pmdarima_model):
        import pmdarima

        self.pmdarima_model = pmdarima_model
        self._pmdarima_version = pmdarima.__version__

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.pmdarima_model

    def predict(self, dataframe, params: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
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
