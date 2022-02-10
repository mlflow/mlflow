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
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _get_flavor_configuration
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
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    :param pmdarima_model:
    :param path:
    :param conda_env:
    :param mlflow_model:
    :param signature:
    :param input_example:
    :param pip_requirements:
    :param extra_pip_requirements:
    """
    import pmdarima

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(f"Path '{path}' already exists")
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_path = os.path.join(path, _MODEL_BINARY_FILE_NAME)
    _save_model(pmdarima_model, model_data_path)

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model, loader_module="mlflow.pmdarima", env=_CONDA_ENV_FILE_NAME, **model_bin_kwargs
    )
    flavor_conf = {
        _MODEL_TYPE_KEY: pmdarima_model.__class__.__name__,
        **model_bin_kwargs,
    }
    mlflow_model.add_flavor(FLAVOR_NAME, pmdarima_version=pmdarima.__version__, **flavor_conf)
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


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    pmdarima_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    :param pmdarima_model:
    :param artifact_path:
    :param conda_env:
    :param registered_model_name:
    :param signature:
    :param input_example:
    :param await_registration_for:
    :param pip_requirements:
    :param extra_pip_requirements:
    :param kwargs:

    :return:
    """

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.pmdarima,
        registered_model_name=registered_model_name,
        pmdarima_model=pmdarima_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def load_model(model_uri, dst_path=None):
    """
    :param model_uri:
    :param dst_path:
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
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
        self.pmdarima_model = pmdarima_model

    def predict(self, dataframe):

        attrs = dataframe.to_dict(orient="index").get(0)
        n_periods = attrs.get("n_periods", None)

        # NB Any model that is trained with exogeneous regressor elements will need to provide
        # `X` entries as a 2D array structure to the predict method.
        X = attrs.get("X", None)

        return_conf_int = attrs.get("return_conf_int", False)
        alpha = attrs.get("alpha", 0.05)

        if not isinstance(n_periods, int):
            raise MlflowException(
                "The prediction DataFrame must contain a column `n_periods` with "
                "an integer value for number of future periods to predict."
            )

        return self.pmdarima_model.predict(
            n_periods=n_periods, X=X, return_conf_int=return_conf_int, alpha=alpha
        )
