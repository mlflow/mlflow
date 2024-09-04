import urllib.parse

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
    # new_uri is the parsed_uri WITHOUT the scheme
    new_uri = parsed_uri.path
    new_parsed_uri = urllib.parse.urlparse(new_uri)
    # checking if the server_uri scheme is "uc"
    if parsed_uri.scheme == "uc":
        if parsed_uri.path == _DATABRICKS_UNITY_CATALOG_SCHEME:
            return get_databricks_host_creds(parsed_uri.path)
        else:
            return MlflowHostCreds(
                host=f"{new_parsed_uri.scheme}://{new_parsed_uri.netloc}",
            )
    else:
        raise MlflowException("The scheme of the server_uri is should be 'uc'")
