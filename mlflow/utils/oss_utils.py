import urllib.parse

from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import (
    _DATABRICKS_UNITY_CATALOG_SCHEME,
)


def get_oss_host_creds(server_uri=None):
    parsed_uri = urllib.parse.urlparse(server_uri)
    new_uri = parsed_uri.path
    new_parsed_uri = urllib.parse.urlparse(new_uri)
    if parsed_uri.scheme == "uc":
        if parsed_uri.path == _DATABRICKS_UNITY_CATALOG_SCHEME:
            return get_databricks_host_creds(parsed_uri.path)
        else:
            return MlflowHostCreds(
                host=f"{new_parsed_uri.scheme}://{new_parsed_uri.netloc}",
            )
