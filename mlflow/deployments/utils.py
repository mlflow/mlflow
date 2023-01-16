import urllib
from mlflow.exceptions import MlflowException


def parse_target_uri(target_uri):
    """Parse out the deployment target from the provided target uri"""
    parsed = urllib.parse.urlparse(target_uri)
    if not parsed.scheme:
        if parsed.path:
            # uri = 'target_name' (without :/<path>)
            return parsed.path
        raise MlflowException(
            "Not a proper deployment URI: %s. " % target_uri
            + "Deployment URIs must be of the form 'target' or 'target:/suffix'"
        )
    return parsed.scheme
