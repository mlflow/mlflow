import logging
import os

import cloudpickle

from mlflow.models import Model
from mlflow.models.dependencies_schemas import _get_dependencies_schema_from_model
from mlflow.models.model import _update_active_model_id_based_on_mlflow_model
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
)

_DEFAULT_MODEL_PATH = "data/model.pkl"
_logger = logging.getLogger(__name__)


def _set_dependency_schema_to_tracer(model_path, callbacks):
    """
    Set dependency schemas from the saved model metadata to the tracer
    to propagate it to inference traces.
    """
    from mlflow.dspy.callback import MlflowCallback

    tracer = next((cb for cb in callbacks if isinstance(cb, MlflowCallback)), None)
    if tracer is None:
        return

    model = Model.load(model_path)
    tracer.set_dependencies_schema(_get_dependencies_schema_from_model(model))


def _load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    mlflow_model = Model.load(local_model_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name="dspy")

    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    model_path = flavor_conf.get("model_path", _DEFAULT_MODEL_PATH)
    with open(os.path.join(local_model_path, model_path), "rb") as f:
        loaded_wrapper = cloudpickle.load(f)

    _set_dependency_schema_to_tracer(local_model_path, loaded_wrapper.dspy_settings["callbacks"])
    _update_active_model_id_based_on_mlflow_model(mlflow_model)
    return loaded_wrapper


@trace_disabled  # Suppress traces for internal calls while loading model
def load_model(model_uri, dst_path=None):
    """
    Load a Dspy model from a run.

    This function will also set the global dspy settings `dspy.settings` by the saved settings.

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
        An `dspy.module` instance, representing the dspy model.
    """
    import dspy

    wrapper = _load_model(model_uri, dst_path)

    # Set the global dspy settings for reproducing the model's behavior when the model is
    # loaded via `mlflow.dspy.load_model`. Note that for the model to be loaded as pyfunc,
    # settings will be set in the wrapper's `predict` method via local context to avoid the
    # "dspy.settings can only be changed by the thread that initially configured it" error
    # in Databricks model serving.
    dspy.settings.configure(**wrapper.dspy_settings)

    return wrapper.model


def _load_pyfunc(path):
    return _load_model(path)
