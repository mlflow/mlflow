import urllib.parse

from mlflow.utils.rest_utils import MlflowHostCreds


def get_oss_host_creds(server_uri=None):
    parsed_uri = urllib.parse.urlparse(server_uri)
    if parsed_uri.scheme == "uc":
        return MlflowHostCreds(
            host=server_uri
        )  # This is a dummy implementation. The actual implementation will require properly parsing the URI
