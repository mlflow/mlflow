import urllib.parse

from mlflow.utils.rest_utils import MlflowHostCreds


def get_oss_host_creds(server_uri=None):
    parsed_uri = urllib.parse.urlparse(server_uri)
    if parsed_uri.scheme == "uc":
        new_uri = server_uri.replace("uc:", "")
        new_parsed_uri = urllib.parse.urlparse(new_uri)
        return MlflowHostCreds(
            host=f"{new_parsed_uri.scheme}://{new_parsed_uri.netloc}",
        )
