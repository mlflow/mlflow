import urllib.parse

from mlflow.environment_variables import MLFLOW_UC_OSS_TOKEN
from mlflow.exceptions import MlflowException
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import (
    _DATABRICKS_UNITY_CATALOG_SCHEME,
)


def get_oss_host_creds(server_uri=None):
    """
    Retrieve the host credentials for the OSS server.

    Args:
        server_uri (str): The URI of the server.

    Returns:
        MlflowHostCreds: The host credentials for the OSS server.
    """
    parsed_uri = urllib.parse.urlparse(server_uri)

    if parsed_uri.scheme != "uc":
        raise MlflowException("The scheme of the server_uri should be 'uc'")

    if parsed_uri.path == _DATABRICKS_UNITY_CATALOG_SCHEME:
        return get_databricks_host_creds(parsed_uri.path)
    return MlflowHostCreds(host=parsed_uri.path, token=MLFLOW_UC_OSS_TOKEN.get())
