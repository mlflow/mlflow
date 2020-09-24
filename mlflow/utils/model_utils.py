import logging
import os
from distutils.version import LooseVersion

import mlflow.utils.cloudpickle
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.uri import append_to_uri_path


_logger = logging.getLogger(__name__)


def _get_flavor_configuration(model_path, flavor_name):
    """
    Obtains the configuration for the specified flavor from the specified
    MLflow model path. If the model does not contain the specified flavor,
    an exception will be thrown.

    :param model_path: The path to the root directory of the MLflow model for which to load
                       the specified flavor configuration.
    :param flavor_name: The name of the flavor configuration to load.
    :return: The flavor configuration as a dictionary.
    """
    model_configuration_path = os.path.join(model_path, MLMODEL_FILE_NAME)
    if not os.path.exists(model_configuration_path):
        raise MlflowException(
            'Could not find an "{model_file}" configuration file at "{model_path}"'.format(
                model_file=MLMODEL_FILE_NAME, model_path=model_path
            ),
            RESOURCE_DOES_NOT_EXIST,
        )

    model_conf = Model.load(model_configuration_path)
    if flavor_name not in model_conf.flavors:
        raise MlflowException(
            'Model does not have the "{flavor_name}" flavor'.format(flavor_name=flavor_name),
            RESOURCE_DOES_NOT_EXIST,
        )
    conf = model_conf.flavors[flavor_name]
    return conf


def _get_flavor_configuration_from_uri(model_uri, flavor_name):
    """
    Obtains the configuration for the specified flavor from the specified
    MLflow model uri. If the model does not contain the specified flavor,
    an exception will be thrown.

    :param model_uri: The path to the root directory of the MLflow model for which to load
                       the specified flavor configuration.
    :param flavor_name: The name of the flavor configuration to load.
    :return: The flavor configuration as a dictionary.
    """
    try:
        ml_model_file = _download_artifact_from_uri(
            artifact_uri=append_to_uri_path(model_uri, MLMODEL_FILE_NAME)
        )
    except Exception as ex:
        raise MlflowException(
            'Failed to download an "{model_file}" model file from "{model_uri}": {ex}'.format(
                model_file=MLMODEL_FILE_NAME, model_uri=model_uri, ex=ex
            ),
            RESOURCE_DOES_NOT_EXIST,
        )
    model_conf = Model.load(ml_model_file)
    if flavor_name not in model_conf.flavors:
        raise MlflowException(
            'Model does not have the "{flavor_name}" flavor'.format(flavor_name=flavor_name),
            RESOURCE_DOES_NOT_EXIST,
        )
    return model_conf.flavors[flavor_name]


def _get_cloudpickle_module_for_deserialization(mlflow_version=None):
    """
    Gets the cloudpickle module used for deserializing MLflow Model
    artifacts that were persisted using cloudpickle serialization format. Versions
    of MLflow > 1.11.0 use an inlined version of cloudpickle, while older versions
    use a standalone version of cloudpickle installed from PyPI/Anaconda

    :param mlflow_version: The string version of MLflow for which to obtain a deserialization
                           model. If unspecified, we assume a version of MLflow <= 1.11.0
                           because models persisted in these versions do not contain version
                           information.
    """
    if not mlflow_version or LooseVersion(mlflow_version) < LooseVersion("1.11.0"):
        try:
            import cloudpickle

            return cloudpickle
        except ImportError:
            _logger.warning(
                "Failed to import a standalone installation of cloudpickle for compatibility with"
                " models saved in MLflow version <= '1.11.0'. Using an inlined installation of"
                " cloudpickle with version '%s' instead. " % mlflow.utils.cloudpickle.__version___
            )

    return mlflow.utils.cloudpickle
