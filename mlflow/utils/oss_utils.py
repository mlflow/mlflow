import urllib.parse

from mlflow.utils.rest_utils import MlflowHostCreds

from mlflow.utils.uri import (
    _DATABRICKS_UNITY_CATALOG_SCHEME,
)
from mlflow.utils.databricks_utils import get_databricks_host_creds


def get_oss_host_creds(server_uri=None):
    parsed_uri = urllib.parse.urlparse(server_uri)
    new_uri = parsed_uri.path
    new_parsed_uri = urllib.parse.urlparse(new_uri)
    print("getosshostcreds test", parsed_uri.scheme)
    print("path", parsed_uri.path)
    if parsed_uri.scheme == "uc":
        print("new parsed uri scheme", new_parsed_uri.scheme, "total parse", new_parsed_uri)
        if new_parsed_uri.scheme == _DATABRICKS_UNITY_CATALOG_SCHEME:
            print("we are abt to call db host creds")
            db_host = get_databricks_host_creds(parsed_uri.path)
            print("HOST:", db_host.host)
            return db_host
        else:
            return MlflowHostCreds(
                host=f"{new_parsed_uri.scheme}://{new_parsed_uri.netloc}",
            )