import urllib.parse

import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri, is_databricks_uri


def is_using_databricks_registry(uri):
    profile_uri = get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
    return is_databricks_uri(profile_uri)


def _improper_model_uri_msg(uri):
    return (
        "Not a proper models:/ URI: %s. " % uri
        + "Models URIs must be of the form 'models:/<model_name>/<version or stage>'."
    )


def _get_model_version_from_stage(client, name, stage):
    latest = client.get_latest_versions(name, [stage])
    if len(latest) == 0:
        raise MlflowException(
            "No versions of model with name '{name}' and "
            "stage '{stage}' found".format(name=name, stage=stage)
        )
    return latest[0].version


def _parse_model_uri(uri):
    """
    Returns (name, version, stage). Since a models:/ URI can only have one of {version, stage},
    it will return (name, version, None) or (name, None, stage).
    """
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "models":
        raise MlflowException(_improper_model_uri_msg(uri))

    path = parsed.path
    if not path.startswith("/") or len(path) <= 1:
        raise MlflowException(_improper_model_uri_msg(uri))
    parts = path[1:].split("/")

    if len(parts) != 2 or parts[0].strip() == "":
        raise MlflowException(_improper_model_uri_msg(uri))

    if parts[1].isdigit():
        return parts[0], int(parts[1]), None
    else:
        return parts[0], None, parts[1]


def get_model_name_and_version(client, models_uri):
    (model_name, model_version, model_stage) = _parse_model_uri(models_uri)
    if model_stage is not None:
        model_version = _get_model_version_from_stage(client, model_name, model_stage)
    return model_name, str(model_version)
